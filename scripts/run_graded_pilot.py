"""Run empirical pilot on graded near-lossless compression and classifier impact.

Evaluates 120 stratified images across 41 compression variants:
- 30 Graded near-lossless: (delta_core, delta_band) in {(0,2), (0,4), (1,3), (2,4), (3,6)}
  x band_px in {4, 8} x bg_quality in {10, 30, 50}
- 6 Lossless-dilated: band_px in {4, 8} x bg_quality in {10, 30, 50}
- 5 DWT-only references: quality in {10, 20, 30, 50, 70}

Uses CUDA GPU (RTX 2050) for batch classifier inference.
Saves results to results/graded_pilot.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import load_model, predict_proba
from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RESULTS_DIR,
    WORK_DIR,
)
from medcomp.dwt_codec import dwt_encode
from medcomp.graded_codec import dilate_mask, graded_encode, make_zones
from medcomp.io_utils import load_image
from medcomp.metrics import masked_psnr

OUT_CSV = RESULTS_DIR / "graded_pilot.csv"
PRED_MASK_DIR = WORK_DIR / "pred" / "mask"

# 41 Variants definition
DELTA_PAIRS = [(0, 2), (0, 4), (1, 3), (2, 4), (3, 6)]
BAND_PIXELS = [4, 8]
BG_QUALITIES = [10, 30, 50]
DWT_QUALITIES = [10, 20, 30, 50, 70]

CSV_COLUMNS = [
    "stem",
    "source",
    "label",
    "fold",
    "has_true_mask",
    "family",
    "variant",
    "delta_core",
    "delta_band",
    "band_px",
    "bg_quality",
    "total_bytes",
    "bytes_header",
    "bytes_mask",
    "bytes_core",
    "bytes_band",
    "bytes_bg",
    "bpp",
    "max_err_core",
    "max_err_band",
    "max_err_bg",
    "prob_orig",
    "prob_dec",
    "flip",
    "abs_prob_diff",
    "max_err_true_lung",
    "p99_err_true_lung",
    "frac_err_gt_2",
    "frac_err_gt_4",
    "psnr_lung_true",
    "frac_lung_in_core",
    "frac_lung_in_band",
    "frac_lung_in_bg",
]


def build_variant_specs() -> list[dict[str, Any]]:
    """Generate configuration metadata for all 41 compression variants."""
    specs: list[dict[str, Any]] = []

    # 1. 30 Graded variants
    for dc, db in DELTA_PAIRS:
        for bp in BAND_PIXELS:
            fam = f"graded_d{dc}_{db}_bp{bp}"
            for q in BG_QUALITIES:
                specs.append({
                    "type": "graded",
                    "family": fam,
                    "variant": f"{fam}_q{q}",
                    "delta_core": dc,
                    "delta_band": db,
                    "band_px": bp,
                    "bg_quality": q,
                })

    # 2. 6 Lossless-dilated variants
    for bp in BAND_PIXELS:
        fam = f"lossless_dilated_bp{bp}"
        for q in BG_QUALITIES:
            specs.append({
                "type": "lossless_dilated",
                "family": fam,
                "variant": f"{fam}_q{q}",
                "delta_core": 0,
                "delta_band": 0,
                "band_px": bp,
                "bg_quality": q,
            })

    # 3. 5 DWT-only references
    for q in DWT_QUALITIES:
        specs.append({
            "type": "dwt_only",
            "family": "dwt_only",
            "variant": f"dwt_q{q}",
            "delta_core": -1,
            "delta_band": -1,
            "band_px": 0,
            "bg_quality": q,
        })

    return specs


ALL_VARIANT_SPECS = build_variant_specs()


def get_pilot_sample(manifest_df: pd.DataFrame, n_samples: int = 120, seed: int = 42) -> pd.DataFrame:
    """Select stratified pilot sample across source and label."""
    df = manifest_df.copy()
    df["stratum"] = df["source"].astype(str) + "_" + df["label"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(df, df["stratum"]))
    sample_df = df.iloc[train_idx].copy().sort_values("stem").reset_index(drop=True)
    return sample_df


def process_image_variants(
    item: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    """Compress an image across all 41 variants on a worker process.

    Returns:
        tuple: (item metadata, list of variant result records without classifier probs, list of recon images)
    """
    stem = str(item["stem"])
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    pred_mask_path = PRED_MASK_DIR / f"{stem}.png"
    true_mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

    img = load_image(img_path)
    pred_mask = load_image(pred_mask_path) > 127
    has_true = true_mask_path.is_file()
    true_mask = (load_image(true_mask_path) > 127) if has_true else None

    h, w = img.shape
    num_pixels = float(h * w)

    records: list[dict[str, Any]] = []
    recon_images: list[np.ndarray] = []

    for spec in ALL_VARIANT_SPECS:
        v_type = spec["type"]
        fam = spec["family"]
        var_name = spec["variant"]
        q = spec["bg_quality"]
        bp = spec["band_px"]
        dc = spec["delta_core"]
        db = spec["delta_band"]

        if v_type == "graded":
            params = {
                "delta_core": dc,
                "delta_band": db,
                "band_px": bp,
                "bg_quality": q,
                "bg_method": "dwt",
                "fill": "inpaint",
            }
            comp_bytes, recon, parts = graded_encode(img, pred_mask, params)
            zone_map = make_zones(pred_mask, bp)

            # Zone errors
            c_mask = (zone_map == 1)
            b_mask = (zone_map == 2)
            bg_mask = (zone_map == 0)

            err_c = np.abs(img[c_mask].astype(int) - recon[c_mask].astype(int)) if np.any(c_mask) else np.array([0])
            err_b = np.abs(img[b_mask].astype(int) - recon[b_mask].astype(int)) if np.any(b_mask) else np.array([0])
            err_bg = np.abs(img[bg_mask].astype(int) - recon[bg_mask].astype(int)) if np.any(bg_mask) else np.array([0])

            max_err_c = int(np.max(err_c))
            max_err_b = int(np.max(err_b))
            max_err_bg = int(np.max(err_bg))

            assert max_err_c <= dc, f"Stem {stem} {var_name}: Core error {max_err_c} > delta_core {dc}"
            assert max_err_b <= db, f"Stem {stem} {var_name}: Band error {max_err_b} > delta_band {db}"

        elif v_type == "lossless_dilated":
            dilated_core = dilate_mask(pred_mask, bp)
            params = {
                "delta_core": 0,
                "delta_band": 0,
                "band_px": 0,
                "bg_quality": q,
                "bg_method": "dwt",
                "fill": "inpaint",
            }
            comp_bytes, recon, parts = graded_encode(img, dilated_core, params)
            zone_map = np.zeros_like(pred_mask, dtype=np.uint8)
            zone_map[dilated_core] = 1

            c_mask = (zone_map == 1)
            bg_mask = (zone_map == 0)

            err_c = np.abs(img[c_mask].astype(int) - recon[c_mask].astype(int)) if np.any(c_mask) else np.array([0])
            err_bg = np.abs(img[bg_mask].astype(int) - recon[bg_mask].astype(int)) if np.any(bg_mask) else np.array([0])

            max_err_c = int(np.max(err_c))
            max_err_b = 0
            max_err_bg = int(np.max(err_bg))

            assert max_err_c == 0, f"Stem {stem} {var_name}: Lossless dilated core error {max_err_c} != 0"

        else:  # dwt_only
            comp_bytes, recon = dwt_encode(img, quality=q)
            parts = {
                "header": 11,
                "mask": 0,
                "core": 0,
                "band": 0,
                "background": len(comp_bytes) - 11,
            }
            zone_map = np.zeros_like(pred_mask, dtype=np.uint8)
            all_err = np.abs(img.astype(int) - recon.astype(int))
            max_err_c = -1
            max_err_b = -1
            max_err_bg = int(np.max(all_err))

        total_bytes = len(comp_bytes)
        bpp = (total_bytes * 8.0) / num_pixels

        # True lung statistics (if available)
        if has_true and true_mask is not None and np.any(true_mask):
            true_err = np.abs(img[true_mask].astype(int) - recon[true_mask].astype(int))
            max_err_tl = float(np.max(true_err))
            p99_err_tl = float(np.percentile(true_err, 99))
            frac_gt_2 = float(np.mean(true_err > 2))
            frac_gt_4 = float(np.mean(true_err > 4))

            p_val = masked_psnr(img, recon, true_mask, region="roi")
            psnr_tl = float(p_val) if not np.isinf(p_val) else 100.0

            n_tl = float(np.sum(true_mask))
            frac_core = float(np.sum(true_mask & (zone_map == 1)) / n_tl)
            frac_band = float(np.sum(true_mask & (zone_map == 2)) / n_tl)
            frac_bg = float(np.sum(true_mask & (zone_map == 0)) / n_tl)
        else:
            max_err_tl = np.nan
            p99_err_tl = np.nan
            frac_gt_2 = np.nan
            frac_gt_4 = np.nan
            psnr_tl = np.nan
            frac_core = np.nan
            frac_band = np.nan
            frac_bg = np.nan

        rec: dict[str, Any] = {
            "stem": stem,
            "source": str(item["source"]),
            "label": int(item["label"]),
            "fold": int(item["fold"]),
            "has_true_mask": 1 if has_true else 0,
            "family": fam,
            "variant": var_name,
            "delta_core": dc,
            "delta_band": db,
            "band_px": bp,
            "bg_quality": q,
            "total_bytes": total_bytes,
            "bytes_header": parts["header"],
            "bytes_mask": parts["mask"],
            "bytes_core": parts["core"],
            "bytes_band": parts["band"],
            "bytes_bg": parts["background"],
            "bpp": round(bpp, 6),
            "max_err_core": max_err_c,
            "max_err_band": max_err_b,
            "max_err_bg": max_err_bg,
            "max_err_true_lung": max_err_tl,
            "p99_err_true_lung": p99_err_tl,
            "frac_err_gt_2": frac_gt_2,
            "frac_err_gt_4": frac_gt_4,
            "psnr_lung_true": psnr_tl,
            "frac_lung_in_core": frac_core,
            "frac_lung_in_band": frac_band,
            "frac_lung_in_bg": frac_bg,
        }

        records.append(rec)
        recon_images.append(recon)

    return item, records, recon_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graded near-lossless compression pilot.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU worker processes (default: 4).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to run.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for classifier evaluation on RTX 2050!")
    device = torch.device("cuda")
    print(f"CUDA Device: {torch.cuda.get_device_name(device)} with mixed precision.")

    manifest_path = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    # 1. Stratified pilot sample of 120 images
    sample_df = get_pilot_sample(manifest_df, n_samples=120, seed=42)
    if args.limit is not None and args.limit > 0:
        sample_df = sample_df.iloc[:args.limit].copy()
        print(f"Limiting execution to first {len(sample_df)} images.")

    print(f"Target pilot images: {len(sample_df)}")

    # 2. Classifier OOF reference probabilities
    oof_path = RESULTS_DIR / "classifier_oof.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Classifier OOF predictions not found: {oof_path}")
    oof_df = pd.read_csv(oof_path)
    oof_probs = dict(zip(oof_df["stem"], oof_df["prob_tb"]))

    # 3. Load 5 fold models onto GPU
    print("Loading 5 fold models onto GPU...")
    fold_models = [load_model(k, device=device) for k in range(5)]
    for m in fold_models:
        m.eval()

    # 4. Resume check
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_stems: set[str] = set()
    write_header = not OUT_CSV.is_file()

    expected_variants = len(ALL_VARIANT_SPECS)  # 41
    if OUT_CSV.is_file():
        existing_df = pd.read_csv(OUT_CSV)
        if not existing_df.empty and "stem" in existing_df.columns:
            counts = existing_df["stem"].value_counts()
            completed_stems = set(counts[counts >= expected_variants].index)
            print(f"Resuming: found {len(completed_stems)} completed images in {OUT_CSV.name}")

    pending_items = [
        row.to_dict()
        for _, row in sample_df.iterrows()
        if str(row["stem"]) not in completed_stems
    ]
    print(f"Pending images to process: {len(pending_items)}")

    if not pending_items:
        print("All requested pilot images are already processed!")
        return

    csv_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    t_start = time.time()
    n_done = 0

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for item, records, recon_images in tqdm(
                executor.map(process_image_variants, pending_items),
                total=len(pending_items),
                desc="Graded Pilot",
            ):
                stem = str(item["stem"])
                fold = int(item["fold"])
                prob_orig = float(oof_probs[stem])
                model = fold_models[fold]

                # Batched GPU prediction for all 41 variant reconstructions
                with torch.inference_mode(), torch.autocast("cuda"):
                    probs_dec = predict_proba(model, recon_images, batch_size=64, device=device)

                # Merge probabilities and write rows
                for rec, p_dec in zip(records, probs_dec):
                    prob_dec = float(p_dec)
                    flip = int((prob_orig >= 0.5) != (prob_dec >= 0.5))
                    abs_diff = abs(prob_orig - prob_dec)

                    rec["prob_orig"] = round(prob_orig, 6)
                    rec["prob_dec"] = round(prob_dec, 6)
                    rec["flip"] = flip
                    rec["abs_prob_diff"] = round(abs_diff, 6)

                    writer.writerow(rec)

                csv_file.flush()
                n_done += 1

                if n_done % 5 == 0 or n_done == len(pending_items):
                    elapsed = time.time() - t_start
                    sec_per_img = elapsed / n_done
                    print(f"[{n_done}/{len(pending_items)}] Speed: {sec_per_img:.2f} s/image. Projected total: {sec_per_img * len(sample_df):.1f} s")

    finally:
        csv_file.close()

    total_time = time.time() - t_start
    print(f"Finished {n_done} images in {total_time:.2f} seconds ({total_time / max(1, n_done):.2f} s/img). Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
