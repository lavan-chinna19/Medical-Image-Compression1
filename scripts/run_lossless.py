"""Benchmark custom lossless ROI and mask codecs across all masked dataset images."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
)
from medcomp.io_utils import load_image
from medcomp.lossless_codec import (
    lossless_decode,
    lossless_encode,
    lossless_encode_with_stats,
)
from medcomp.mask_codec import mask_encode

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
LOSSLESS_RESULTS_CSV = RESULTS_DIR / "lossless_results.csv"
LOSSLESS_SUMMARY_CSV = RESULTS_DIR / "lossless_summary.csv"


def process_single_image(item: Dict[str, Any], png_bpp: float) -> Dict[str, Any]:
    """Process a single masked image with lossless ROI and mask encoding."""
    stem = item["stem"]
    source = item["source"]
    label = int(item["label"])
    fold = int(item["fold"])

    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

    img = load_image(img_path)
    mask = (load_image(mask_path) > 127)

    h, w = img.shape
    num_pixels = img.size
    roi_pixels = int(np.sum(mask))
    roi_fraction = float(roi_pixels / num_pixels)

    # 1. Lossless ROI encoding (context=False) with timing
    t0 = time.perf_counter()
    roi_bytes_data, roi_stats = lossless_encode_with_stats(img, mask=mask, context=False)
    enc_time = time.perf_counter() - t0
    roi_bytes = len(roi_bytes_data)

    # 2. Lossless ROI decoding with timing and exactness check
    t1 = time.perf_counter()
    recon_roi = lossless_decode(roi_bytes_data, mask=mask)
    dec_time = time.perf_counter() - t1
    roi_exact = bool(np.array_equal(recon_roi[mask], img[mask]))

    roi_bpp = float((roi_bytes * 8) / max(1, roi_pixels))
    roi_entropy = float(roi_stats["residual_entropy"])
    roi_avg_len = float(roi_stats["avg_code_length"])

    # 3. Context-based ROI encoding (context=True)
    ctx_roi_data = lossless_encode(img, mask=mask, context=True)
    ctx_roi_bytes = len(ctx_roi_data)

    # 4. Mask encoding (both RLE and BBox variants)
    mask_rle_data = mask_encode(mask, bbox_mode=False)
    mask_rle_bytes = len(mask_rle_data)

    mask_bbox_data = mask_encode(mask, bbox_mode=True)
    mask_bbox_bytes = len(mask_bbox_data)

    mask_bytes = min(mask_rle_bytes, mask_bbox_bytes)

    # 5. Full-image lossless encoding (mask=None)
    full_data = lossless_encode(img, mask=None, context=False)
    full_bytes = len(full_data)
    full_bpp = float((full_bytes * 8) / num_pixels)

    return {
        "stem": stem,
        "source": source,
        "label": label,
        "fold": fold,
        "roi_pixels": roi_pixels,
        "roi_fraction": round(roi_fraction, 6),
        "roi_bytes": roi_bytes,
        "roi_bits_per_roi_pixel": round(roi_bpp, 4),
        "roi_entropy_of_residuals": round(roi_entropy, 4),
        "roi_avg_code_length": round(roi_avg_len, 4),
        "mask_bytes": mask_bytes,
        "mask_rle_bytes": mask_rle_bytes,
        "mask_bbox_bytes": mask_bbox_bytes,
        "full_image_lossless_bytes": full_bytes,
        "full_image_bpp": round(full_bpp, 4),
        "png_bpp": round(png_bpp, 4) if not np.isnan(png_bpp) else np.nan,
        "context_roi_bytes": ctx_roi_bytes,
        "encode_time_s": round(enc_time, 6),
        "decode_time_s": round(dec_time, 6),
        "roi_exact": roi_exact,
    }


def compute_lossless_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall and per-source summary table for lossless evaluation."""
    summary_rows = []

    groups = [("all", df)] + [(s, df[df["source"] == s]) for s in sorted(df["source"].unique())]

    for group_name, sub in groups:
        if sub.empty:
            continue
        summary_rows.append({
            "source": group_name,
            "images_count": len(sub),
            "mean_roi_pixels": round(float(sub["roi_pixels"].mean()), 1),
            "mean_roi_fraction": round(float(sub["roi_fraction"].mean()), 4),
            "mean_roi_bytes": round(float(sub["roi_bytes"].mean()), 1),
            "mean_roi_bpp": round(float(sub["roi_bits_per_roi_pixel"].mean()), 4),
            "mean_context_roi_bytes": round(float(sub["context_roi_bytes"].mean()), 1),
            "mean_context_roi_bpp": round(float((sub["context_roi_bytes"] * 8 / sub["roi_pixels"]).mean()), 4),
            "mean_roi_residual_entropy": round(float(sub["roi_entropy_of_residuals"].mean()), 4),
            "mean_roi_avg_code_length": round(float(sub["roi_avg_code_length"].mean()), 4),
            "mean_mask_bytes": round(float(sub["mask_bytes"].mean()), 1),
            "mean_mask_rle_bytes": round(float(sub["mask_rle_bytes"].mean()), 1),
            "mean_mask_bbox_bytes": round(float(sub["mask_bbox_bytes"].mean()), 1),
            "mean_full_lossless_bpp": round(float(sub["full_image_bpp"].mean()), 4),
            "mean_png_bpp": round(float(sub["png_bpp"].dropna().mean()), 4) if not sub["png_bpp"].dropna().empty else np.nan,
            "mean_encode_time_s": round(float(sub["encode_time_s"].mean()), 4),
            "mean_decode_time_s": round(float(sub["decode_time_s"].mean()), 4),
            "roi_exact_fraction": round(float(sub["roi_exact"].mean()), 4),
        })

    return pd.DataFrame(summary_rows)


def run_lossless(limit: Optional[int] = None, workers: int = 4, force: bool = False) -> pd.DataFrame:
    """Run lossless ROI and mask benchmark across masked images."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    # Filter to masked images only
    df = df[df["has_mask"] == True].copy()
    print(f"Total available masked images: {len(df)}")

    # Load PNG baseline bpp mapping
    png_bpp_map: dict[str, float] = {}
    if BASELINES_CSV.is_file():
        b_df = pd.read_csv(BASELINES_CSV)
        png_sub = b_df[b_df["codec"] == "PNG"]
        for _, r in png_sub.iterrows():
            png_bpp_map[str(r["stem"])] = float(r["bpp"])

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
        print(f"Sampled {len(df)} images: Montgomery={sum(df['source']=='Montgomery')}, Shenzhen={sum(df['source']=='Shenzhen')}")

    items = df.to_dict(orient="records")

    # Resume check
    existing_stems: Set[str] = set()
    existing_rows: List[Dict[str, Any]] = []
    if not force and LOSSLESS_RESULTS_CSV.is_file():
        try:
            prev_df = pd.read_csv(LOSSLESS_RESULTS_CSV)
            existing_stems = set(prev_df["stem"].astype(str))
            existing_rows = prev_df.to_dict(orient="records")
            print(f"Resuming: found {len(existing_stems)} existing results in {LOSSLESS_RESULTS_CSV}")
        except Exception as e:
            print(f"Could not read existing {LOSSLESS_RESULTS_CSV}: {e}. Starting fresh.")
    elif force:
        print("Force recomputation requested: starting fresh.")

    pending_items = [it for it in items if str(it["stem"]) not in existing_stems]
    print(f"Pending images to evaluate: {len(pending_items)}")

    new_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    if workers <= 1 or len(pending_items) <= 1:
        for item in tqdm(pending_items, desc="Evaluating Lossless"):
            png_val = png_bpp_map.get(str(item["stem"]), np.nan)
            res = process_single_image(item, png_val)
            new_results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(process_single_image, item, png_bpp_map.get(str(item["stem"]), np.nan)): item
                for item in pending_items
            }
            for fut in tqdm(as_completed(future_to_item), total=len(pending_items), desc="Evaluating Lossless"):
                res = fut.result()
                new_results.append(res)

    total_time = time.perf_counter() - start_time
    if pending_items:
        print(f"Lossless execution completed in {total_time:.2f}s ({total_time / len(pending_items):.4f}s per image)")

    all_rows = existing_rows + new_results
    unique_rows: Dict[str, Dict[str, Any]] = {str(r["stem"]): r for r in all_rows}

    df_out = pd.DataFrame(list(unique_rows.values()))
    df_out.sort_values(by=["stem"], inplace=True)
    df_out.to_csv(LOSSLESS_RESULTS_CSV, index=False)
    print(f"Saved {len(df_out)} rows to {LOSSLESS_RESULTS_CSV}")

    # Compute and save summary
    summary_df = compute_lossless_summary(df_out)
    summary_df.to_csv(LOSSLESS_SUMMARY_CSV, index=False)
    print(f"Saved lossless summary to {LOSSLESS_SUMMARY_CSV}")

    exact_fraction = df_out["roi_exact"].mean()
    print(f"Exact ROI reconstruction rate: {exact_fraction * 100:.2f}%")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run lossless ROI and mask benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--force", action="store_true", help="Force recomputation without resume")
    args = parser.parse_args()

    run_lossless(limit=args.limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
