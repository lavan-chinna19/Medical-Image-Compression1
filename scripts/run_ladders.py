"""Execute rate-distortion ladder compression across all 800 medical images.

Evaluates 3 compression families across all ladder rungs:
1. dwt_only: qualities in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90] (11 rungs)
2. graded_d2_4_bp8: delta_core=2, delta_band=4, band_px=8, bg_quality in [10, 20, 30, 50, 70, 90] (6 rungs)
3. lossless_dilated_bp8: dilated core band_px=8, bg_quality in [30, 50, 70, 90] (4 rungs)

Total rungs per image: 21.

Evaluates out-of-fold steering classifier (ResNet18) and independent judge classifier (EfficientNet-B0)
predictions on original and reconstructed images.

Outputs:
- results/ladders.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import load_model as load_steering_model, predict_proba as predict_steering
from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    RESULTS_DIR,
    WORK_DIR,
)
from medcomp.dwt_codec import dwt_decode, dwt_encode
from medcomp.graded_codec import dilate_mask, graded_decode, graded_encode
from medcomp.io_utils import load_image, save_image
from medcomp.judge import load_judge_model, predict_proba_judge

LADDERS_CSV = RESULTS_DIR / "ladders.csv"
PRED_MASK_DIR = WORK_DIR / "pred" / "mask"
DECODED_CACHE_DIR = WORK_DIR / "decoded_ladders"

DWT_QUALITIES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
GRADED_BG_QUALITIES = [10, 20, 30, 50, 70, 90]
LOSSLESS_BG_QUALITIES = [30, 50, 70, 90]

CSV_HEADER = [
    "stem",
    "source",
    "label",
    "fold",
    "family",
    "quality",
    "rung",
    "total_bytes",
    "bpp",
    "prob_orig_steering",
    "prob_orig_judge",
    "prob_steering",
    "prob_judge",
]


def build_ladder_specs() -> list[dict[str, Any]]:
    """Build specification list for all 21 ladder rungs."""
    specs: list[dict[str, Any]] = []

    # 1. dwt_only: 11 rungs
    for q in DWT_QUALITIES:
        specs.append({
            "family": "dwt_only",
            "quality": q,
            "rung": f"dwt_q{q}",
        })

    # 2. graded_d2_4_bp8: 6 rungs
    for q in GRADED_BG_QUALITIES:
        specs.append({
            "family": "graded_d2_4_bp8",
            "quality": q,
            "rung": f"graded_d2_4_bp8_q{q}",
            "delta_core": 2,
            "delta_band": 4,
            "band_px": 8,
        })

    # 3. lossless_dilated_bp8: 4 rungs
    for q in LOSSLESS_BG_QUALITIES:
        specs.append({
            "family": "lossless_dilated_bp8",
            "quality": q,
            "rung": f"lossless_dilated_bp8_q{q}",
            "band_px": 8,
        })

    return specs


ALL_LADDER_SPECS = build_ladder_specs()


def encode_decode_image_rungs(
    item: dict[str, Any],
    cache_decoded: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray], float]:
    """Encode and decode all ladder rungs for one image (CPU worker).

    Returns:
        tuple: (item metadata, rung metadata records, list of decoded images, elapsed seconds)
    """
    t0 = time.time()
    stem = str(item["stem"])
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    pred_mask_path = PRED_MASK_DIR / f"{stem}.png"

    img = load_image(img_path)
    pred_mask = load_image(pred_mask_path) > 127

    h, w = img.shape
    num_pixels = float(h * w)

    rung_records: list[dict[str, Any]] = []
    decoded_images: list[np.ndarray] = []

    # Pre-dilate mask once for lossless_dilated family
    dilated_mask_bp8 = dilate_mask(pred_mask, 8)

    for spec in ALL_LADDER_SPECS:
        fam = spec["family"]
        q = spec["quality"]
        rung_id = spec["rung"]

        if fam == "dwt_only":
            comp_bytes, _ = dwt_encode(img, quality=q)
            recon = dwt_decode(comp_bytes)
        elif fam == "graded_d2_4_bp8":
            params = {
                "delta_core": 2,
                "delta_band": 4,
                "band_px": 8,
                "bg_quality": q,
                "bg_method": "dwt",
                "fill": "inpaint",
            }
            comp_bytes, _, _ = graded_encode(img, pred_mask, params)
            recon = graded_decode(comp_bytes)
        elif fam == "lossless_dilated_bp8":
            params = {
                "delta_core": 0,
                "delta_band": 0,
                "band_px": 0,
                "bg_quality": q,
                "bg_method": "dwt",
                "fill": "inpaint",
            }
            comp_bytes, _, _ = graded_encode(img, dilated_mask_bp8, params)
            recon = graded_decode(comp_bytes)
        else:
            raise ValueError(f"Unknown family {fam}")

        total_bytes = len(comp_bytes)
        bpp = (total_bytes * 8.0) / num_pixels

        if cache_decoded:
            save_path = DECODED_CACHE_DIR / f"{stem}_{rung_id}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(save_path, recon)

        rung_records.append({
            "stem": stem,
            "source": item["source"],
            "label": item["label"],
            "fold": item["fold"],
            "family": fam,
            "quality": q,
            "rung": rung_id,
            "total_bytes": total_bytes,
            "bpp": bpp,
        })
        decoded_images.append(recon)

    elapsed = time.time() - t0
    return item, rung_records, decoded_images, elapsed


def load_models_for_fold(
    fold: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load steering and judge checkpoints for a specific fold in eval mode."""
    steering_model = load_steering_model(fold, device=device)
    judge_model = load_judge_model(fold, device=device)
    return steering_model, judge_model


def run_ladder_pipeline(
    manifest_df: pd.DataFrame,
    workers: int = 4,
    limit: Optional[int] = None,
    resume: bool = True,
    cache_decoded: bool = False,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Execute ladder pipeline across manifest images."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CRITICAL ERROR: CUDA is NOT available! RTX 2050 is mandatory.")
        device = torch.device("cuda")

    print(f"Running ladder compression pipeline on GPU: '{torch.cuda.get_device_name(device)}'")

    # Stratified limit if requested
    df = manifest_df.copy()
    if limit is not None and limit < len(df):
        strat_key = df["source"].astype(str) + "_" + df["label"].astype(str)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=42)
        idx, _ = next(sss.split(df, strat_key))
        df = df.iloc[idx].copy().sort_values("stem").reset_index(drop=True)
        print(f"Selected {len(df)} stratified images for ladder processing.")

    # Check already completed stems if resume is active
    completed_stems: set[str] = set()
    if resume and LADDERS_CSV.is_file():
        try:
            existing_df = pd.read_csv(LADDERS_CSV)
            if not existing_df.empty and "stem" in existing_df.columns:
                counts = existing_df.groupby("stem")["rung"].count()
                # Image is complete if all 21 rungs are present
                completed_stems = set(counts[counts >= len(ALL_LADDER_SPECS)].index)
                print(f"Resuming: found {len(completed_stems)} fully processed images in {LADDERS_CSV.name}.")
        except Exception as e:
            print(f"Warning reading existing {LADDERS_CSV}: {e}. Starting fresh.")

    remaining_df = df[~df["stem"].isin(completed_stems)].copy().reset_index(drop=True)
    print(f"Total images to process: {len(remaining_df)} (out of {len(df)})")

    # If new file, initialize CSV with header
    if not LADDERS_CSV.is_file() or len(completed_stems) == 0:
        LADDERS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(LADDERS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

    if len(remaining_df) == 0:
        print("All requested images are already processed!")
        return pd.read_csv(LADDERS_CSV)

    # Pre-load models by fold to avoid reloading per image
    models_by_fold: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    for k in range(5):
        if (remaining_df["fold"] == k).any():
            models_by_fold[k] = load_models_for_fold(k, device=device)

    # Process images with ProcessPoolExecutor for CPU codec compression
    items = remaining_df.to_dict(orient="records")
    time_per_image_list: list[float] = []

    # Benchmark first image to report timing immediately
    if len(items) > 0:
        first_item = items[0]
        print(f"\nBenchmarking codec execution on first image: {first_item['stem']}...")
        _, rungs_0, recons_0, t_codec = encode_decode_image_rungs(first_item, cache_decoded=cache_decoded)
        
        # Benchmark GPU inference for first image
        fold_0 = int(first_item["fold"])
        st_model_0, jd_model_0 = models_by_fold[fold_0]
        orig_img_0 = load_image(PROCESSED_IMAGES_DIR / f"{first_item['stem']}.png")

        t_gpu_start = time.time()
        p_orig_st = float(predict_steering(st_model_0, orig_img_0, device=device)[0])
        p_orig_jd = float(predict_proba_judge(jd_model_0, orig_img_0, device=device)[0])
        p_recons_st = predict_steering(st_model_0, recons_0, device=device, batch_size=32)
        p_recons_jd = predict_proba_judge(jd_model_0, recons_0, device=device, batch_size=32)
        t_gpu = time.time() - t_gpu_start

        t_total_single = t_codec + t_gpu
        time_per_image_list.append(t_total_single)
        print(
            f"[Timing Report] Image {first_item['stem']}: "
            f"CPU 21-rung encode+decode = {t_codec:.2f}s, "
            f"GPU batch inference = {t_gpu:.2f}s | "
            f"Total time per image = {t_total_single:.2f}s"
        )
        print(f"Projected time for all 800 images: {(800 * t_total_single) / 60.0:.2f} minutes (single worker).")

        # Write first image results
        with open(LADDERS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            for idx, r_meta in enumerate(rungs_0):
                row = dict(r_meta)
                row["prob_orig_steering"] = p_orig_st
                row["prob_orig_judge"] = p_orig_jd
                row["prob_steering"] = float(p_recons_st[idx])
                row["prob_judge"] = float(p_recons_jd[idx])
                writer.writerow(row)

        remaining_items = items[1:]
    else:
        remaining_items = []

    # Process rest of items
    if len(remaining_items) > 0:
        pbar = tqdm(total=len(remaining_items), desc="Processing ladder images")
        
        # Parallel CPU encoding/decoding
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit in chunks of 20 to keep memory under control
                chunk_size = 20
                for start_idx in range(0, len(remaining_items), chunk_size):
                    chunk = remaining_items[start_idx : start_idx + chunk_size]
                    future_to_item = {
                        executor.submit(encode_decode_image_rungs, it, cache_decoded): it
                        for it in chunk
                    }

                    for future in as_completed(future_to_item):
                        item, rungs, recons, t_c = future.result()
                        f_k = int(item["fold"])
                        st_m, jd_m = models_by_fold[f_k]

                        orig_img = load_image(PROCESSED_IMAGES_DIR / f"{item['stem']}.png")
                        t_g0 = time.time()
                        p_o_st = float(predict_steering(st_m, orig_img, device=device)[0])
                        p_o_jd = float(predict_proba_judge(jd_m, orig_img, device=device)[0])
                        p_r_st = predict_steering(st_m, recons, device=device, batch_size=32)
                        p_r_jd = predict_proba_judge(jd_m, recons, device=device, batch_size=32)
                        t_g = time.time() - t_g0

                        time_per_image_list.append(t_c + t_g)

                        with open(LADDERS_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                            for idx, r_meta in enumerate(rungs):
                                row = dict(r_meta)
                                row["prob_orig_steering"] = p_o_st
                                row["prob_orig_judge"] = p_o_jd
                                row["prob_steering"] = float(p_r_st[idx])
                                row["prob_judge"] = float(p_r_jd[idx])
                                writer.writerow(row)

                        pbar.update(1)
        else:
            for item in remaining_items:
                _, rungs, recons, t_c = encode_decode_image_rungs(item, cache_decoded=cache_decoded)
                f_k = int(item["fold"])
                st_m, jd_m = models_by_fold[f_k]

                orig_img = load_image(PROCESSED_IMAGES_DIR / f"{item['stem']}.png")
                t_g0 = time.time()
                p_o_st = float(predict_steering(st_m, orig_img, device=device)[0])
                p_o_jd = float(predict_proba_judge(jd_m, orig_img, device=device)[0])
                p_r_st = predict_steering(st_m, recons, device=device, batch_size=32)
                p_r_jd = predict_proba_judge(jd_m, recons, device=device, batch_size=32)
                t_g = time.time() - t_g0

                time_per_image_list.append(t_c + t_g)

                with open(LADDERS_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                    for idx, r_meta in enumerate(rungs):
                        row = dict(r_meta)
                        row["prob_orig_steering"] = p_o_st
                        row["prob_orig_judge"] = p_o_jd
                        row["prob_steering"] = float(p_r_st[idx])
                        row["prob_judge"] = float(p_r_jd[idx])
                        writer.writerow(row)

                pbar.update(1)

        pbar.close()

    if time_per_image_list:
        mean_t = np.mean(time_per_image_list)
        print(f"\n[Completed] Average time per image across batch: {mean_t:.2f} seconds.")

    out_df = pd.read_csv(LADDERS_CSV)
    print(f"Total ladder records saved: {len(out_df)} (covering {out_df['stem'].nunique()} unique images).")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 21-rung rate distortion ladders across medical dataset.")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for CPU codec compression (default: 4).")
    parser.add_argument("--limit", type=int, default=None, help="Process a stratified subset of N images.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume existing CSV, restart from scratch.")
    parser.add_argument("--cache-decoded", action="store_true", help="Cache decoded images to disk.")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.is_file(), f"Manifest missing at {manifest_path}"
    manifest_df = pd.read_csv(manifest_path)

    run_ladder_pipeline(
        manifest_df=manifest_df,
        workers=args.workers,
        limit=args.limit,
        resume=not args.no_resume,
        cache_decoded=args.cache_decoded,
    )


if __name__ == "__main__":
    main()
