"""Benchmark hybrid ROI codec using U-Net predicted masks and boundary dilation variants."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RESULTS_DIR,
    WORK_DIR,
    compute_original_bytes,
)
from medcomp.io_utils import load_image
from medcomp.metrics import is_lossless, masked_psnr
from medcomp.roi_codec import roi_decode, roi_encode

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
MASK_DIR = WORK_DIR / "pred" / "mask"

ROI_PRED_RESULTS_CSV = RESULTS_DIR / "roi_codec_predicted.csv"
ROI_PRED_SUMMARY_CSV = RESULTS_DIR / "roi_codec_predicted_summary.csv"

VARIANTS = [0, 4, 8]


def dilate_mask(mask: np.ndarray, d_pixels: int) -> np.ndarray:
    """Dilate binary mask by radius d_pixels."""
    if d_pixels <= 0:
        return mask
    k_size = 2 * d_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return (cv2.dilate(mask.astype(np.uint8), kernel) > 0)


def process_image_pred_roi(item: Dict[str, Any], png_bytes: float, existing_keys: Set[Tuple[str, int]] = set()) -> List[Dict[str, Any]]:
    """Evaluate predicted ROI hybrid codec for all dilation variants on a single image."""
    stem = str(item["stem"])
    source = str(item["source"])
    label = int(item["label"])
    fold = int(item["fold"])
    has_mask = bool(item["has_mask"])

    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    pred_mask_path = MASK_DIR / f"{stem}.png"

    img = load_image(img_path)
    base_pred_mask = (load_image(pred_mask_path) > 127)

    true_mask = None
    true_lung_pixels = 0
    if has_mask:
        true_mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
        true_mask = (load_image(true_mask_path) > 127)
        true_lung_pixels = int(np.sum(true_mask))

    rows: List[Dict[str, Any]] = []

    for d in VARIANTS:
        if (stem, d) in existing_keys:
            continue

        pred_d = dilate_mask(base_pred_mask, d)

        # Encode with Tuned DWT, Q=50, inpaint
        data, _, parts = roi_encode(img, pred_d, method="dwt", quality=50, fill="inpaint")
        recon = roi_decode(data)

        # Check exactness on the predicted ROI
        roi_lossless = is_lossless(img, recon, mask=pred_d)

        total_bytes = len(data)
        ratio_vs_png = float(png_bytes / total_bytes) if not np.isnan(png_bytes) else np.nan

        if has_mask and true_mask is not None:
            outside = true_mask & (~pred_d)
            outside_count = int(np.sum(outside))
            outside_pct = (outside_count / max(1, true_lung_pixels)) * 100.0
            p_lung = masked_psnr(img, recon, true_mask, region="roi")
            psnr_lung_str = "inf" if math.isinf(p_lung) else round(p_lung, 4)
        else:
            outside_count = ""
            outside_pct = ""
            psnr_lung_str = ""

        rows.append({
            "stem": stem,
            "source": source,
            "label": label,
            "fold": fold,
            "variant": f"dilation_{d}px",
            "d_pixels": d,
            "has_mask": has_mask,
            "total_bytes": total_bytes,
            "header_bytes": parts["header"],
            "mask_bytes": parts["mask"],
            "roi_bytes": parts["roi"],
            "background_bytes": parts["background"],
            "ratio_vs_png": round(ratio_vs_png, 4) if not np.isnan(ratio_vs_png) else "",
            "true_lung_pixels_outside_roi": outside_count,
            "pct_true_lung_outside_roi": round(outside_pct, 4) if isinstance(outside_pct, float) else "",
            "psnr_lung_true": psnr_lung_str,
            "roi_lossless": roi_lossless,
        })

    return rows


def compute_pred_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics grouped by dilation variant and source."""
    numeric_df = df.copy()
    for col in [
        "total_bytes", "mask_bytes", "roi_bytes", "background_bytes",
        "ratio_vs_png", "true_lung_pixels_outside_roi", "pct_true_lung_outside_roi",
        "psnr_lung_true",
    ]:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col].replace("inf", np.nan), errors="coerce")

    numeric_df["roi_lossless_bool"] = numeric_df["roi_lossless"].apply(
        lambda x: True if str(x).strip().lower() in ("true", "1") else False
    )

    summary_rows = []
    groups = [("all", numeric_df)] + [(s, numeric_df[numeric_df["source"] == s]) for s in sorted(numeric_df["source"].unique())]

    for source_label, sub_df in groups:
        for d in VARIANTS:
            v_name = f"dilation_{d}px"
            v_sub = sub_df[sub_df["variant"] == v_name]
            if v_sub.empty:
                continue

            masked_v_sub = v_sub[v_sub["has_mask"] == True]

            summary_rows.append({
                "source": source_label,
                "variant": v_name,
                "d_pixels": d,
                "count": len(v_sub),
                "mean_total_bytes": round(float(v_sub["total_bytes"].mean()), 1),
                "mean_mask_bytes": round(float(v_sub["mask_bytes"].mean()), 1),
                "mean_roi_bytes": round(float(v_sub["roi_bytes"].mean()), 1),
                "mean_background_bytes": round(float(v_sub["background_bytes"].mean()), 1),
                "mean_ratio_vs_png": round(float(v_sub["ratio_vs_png"].dropna().mean()), 4) if not v_sub["ratio_vs_png"].dropna().empty else np.nan,
                "roi_lossless_fraction": round(float(v_sub["roi_lossless_bool"].mean()), 4),
                "mean_lung_pixels_outside": round(float(masked_v_sub["true_lung_pixels_outside_roi"].dropna().mean()), 1) if not masked_v_sub.empty else np.nan,
                "mean_pct_lung_outside": round(float(masked_v_sub["pct_true_lung_outside_roi"].dropna().mean()), 4) if not masked_v_sub.empty else np.nan,
                "mean_psnr_lung_true": round(float(masked_v_sub["psnr_lung_true"].dropna().mean()), 2) if not masked_v_sub["psnr_lung_true"].dropna().empty else np.nan,
            })

    return pd.DataFrame(summary_rows)


def run_roi_predicted(limit: Optional[int] = None, workers: int = 4, force: bool = False) -> pd.DataFrame:
    """Benchmark predicted ROI codec across all 800 images."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    # Load PNG baseline compressed_bytes mapping
    png_bytes_map: dict[str, float] = {}
    if BASELINES_CSV.is_file():
        b_df = pd.read_csv(BASELINES_CSV)
        png_sub = b_df[b_df["codec"] == "PNG"]
        for _, r in png_sub.iterrows():
            png_bytes_map[str(r["stem"])] = float(r["compressed_bytes"])

    # Stratified subset if limit specified
    if limit is not None and limit < len(df):
        print(f"Selecting stratified subset of {limit} images (source + label, seed=42)...")
        df["strat_group"] = df["source"].astype(str) + "_" + df["label"].astype(str)
        sampled_indices = []
        rng = np.random.default_rng(42)
        for _, group in df.groupby("strat_group"):
            n_group = max(1, int(round(limit * len(group) / len(df))))
            chosen = rng.choice(group.index.values, size=min(n_group, len(group)), replace=False)
            sampled_indices.extend(chosen)
        if len(sampled_indices) > limit:
            sampled_indices = sampled_indices[:limit]
        elif len(sampled_indices) < limit:
            remaining = [idx for idx in df.index if idx not in sampled_indices]
            sampled_indices.extend(rng.choice(remaining, size=limit - len(sampled_indices), replace=False))
        df = df.loc[sampled_indices].copy()

    items = df.to_dict(orient="records")

    existing_keys: Set[Tuple[str, int]] = set()
    existing_rows: List[Dict[str, Any]] = []
    if not force and ROI_PRED_RESULTS_CSV.is_file():
        try:
            prev_df = pd.read_csv(ROI_PRED_RESULTS_CSV)
            for _, r in prev_df.iterrows():
                existing_keys.add((str(r["stem"]), int(r["d_pixels"])))
            existing_rows = prev_df.to_dict(orient="records")
            print(f"Resuming: found {len(existing_keys)} existing evaluations in {ROI_PRED_RESULTS_CSV}")
        except Exception as e:
            print(f"Could not read existing {ROI_PRED_RESULTS_CSV}: {e}. Starting fresh.")
    elif force:
        print("Force recomputation requested: starting fresh.")

    new_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    if workers <= 1:
        for item in tqdm(items, desc="Evaluating Predicted ROI"):
            p_bytes = png_bytes_map.get(str(item["stem"]), np.nan)
            res = process_image_pred_roi(item, p_bytes, existing_keys=existing_keys)
            new_results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_image_pred_roi, item, png_bytes_map.get(str(item["stem"]), np.nan), existing_keys)
                for item in items
            ]
            for fut in tqdm(as_completed(futures), total=len(items), desc="Evaluating Predicted ROI"):
                res = fut.result()
                new_results.extend(res)

    total_time = time.perf_counter() - start_time
    print(f"Predicted ROI evaluation completed in {total_time:.2f}s ({total_time / max(1, len(items)):.4f}s per image)")

    all_rows = existing_rows + new_results
    unique_rows: Dict[Tuple[str, int], Dict[str, Any]] = {
        (str(r["stem"]), int(r["d_pixels"])): r for r in all_rows
    }

    df_out = pd.DataFrame(list(unique_rows.values()))
    df_out.sort_values(by=["stem", "d_pixels"], inplace=True)
    df_out.to_csv(ROI_PRED_RESULTS_CSV, index=False)
    print(f"Saved {len(df_out)} rows to {ROI_PRED_RESULTS_CSV}")

    # Summary
    summary_df = compute_pred_roi_summary(df_out)
    summary_df.to_csv(ROI_PRED_SUMMARY_CSV, index=False)
    print(f"Saved predicted ROI summary to {ROI_PRED_SUMMARY_CSV}")

    exact_rate = df_out["roi_lossless"].apply(lambda x: str(x).lower() == "true").mean()
    print(f"Exact ROI lossless rate: {exact_rate * 100:.2f}%")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run predicted ROI codec benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--force", action="store_true", help="Force recomputation without resume")
    args = parser.parse_args()

    run_roi_predicted(limit=args.limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
