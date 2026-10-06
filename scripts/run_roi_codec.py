"""Benchmark hybrid ROI codec (lossless ROI, lossy background) across masked dataset images."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
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
    compute_original_bytes,
)
from medcomp.io_utils import load_image
from medcomp.metrics import (
    bits_per_pixel,
    compression_ratio,
    is_lossless,
    masked_psnr,
    masked_ssim,
    psnr,
    ssim,
)
from medcomp.roi_codec import roi_decode, roi_encode

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
ROI_RESULTS_CSV = RESULTS_DIR / "roi_codec_results.csv"
ROI_SUMMARY_CSV = RESULTS_DIR / "roi_codec_summary.csv"

ROI_METHODS = ["dct", "dwt"]
ROI_QUALITIES = [10, 20, 30, 50, 70, 90]


def process_image_roi(item: Dict[str, Any], png_bytes: float, fill: str = "inpaint", existing_keys: Set[Tuple[str, str, int]] = set()) -> List[Dict[str, Any]]:
    """Evaluate all (method, quality) settings for a single masked image."""
    stem = item["stem"]
    source = item["source"]
    label = int(item["label"])
    fold = int(item["fold"])

    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

    img = load_image(img_path)
    mask = (load_image(mask_path) > 127)

    orig_bytes = compute_original_bytes(img)
    num_pixels = img.size
    roi_fraction = float(np.sum(mask) / num_pixels)

    rows: List[Dict[str, Any]] = []

    for method in ROI_METHODS:
        for q in ROI_QUALITIES:
            if (stem, method, q) in existing_keys:
                continue

            t0 = time.perf_counter()
            data, recon, parts = roi_encode(img, mask, method=method, quality=q, fill=fill)
            enc_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            recon_dec = roi_decode(data)
            dec_time = time.perf_counter() - t1

            total_bytes = len(data)
            cr = compression_ratio(orig_bytes, total_bytes)
            bpp = bits_per_pixel(total_bytes, num_pixels)
            ratio_vs_png = float(png_bytes / total_bytes) if not np.isnan(png_bytes) else np.nan

            p_full = psnr(img, recon_dec)
            s_full = ssim(img, recon_dec)
            p_bg = masked_psnr(img, recon_dec, mask, region="background")
            s_bg = masked_ssim(img, recon_dec, mask, region="background")
            p_roi = masked_psnr(img, recon_dec, mask, region="roi")
            s_roi = masked_ssim(img, recon_dec, mask, region="roi")
            roi_loss = is_lossless(img, recon_dec, mask=mask)

            rows.append({
                "stem": stem,
                "source": source,
                "label": label,
                "fold": fold,
                "method": method,
                "quality": q,
                "fill": fill,
                "roi_fraction": round(roi_fraction, 6),
                "orig_bytes": orig_bytes,
                "total_bytes": total_bytes,
                "header_bytes": parts["header"],
                "mask_bytes": parts["mask"],
                "roi_bytes": parts["roi"],
                "background_bytes": parts["background"],
                "compression_ratio": round(cr, 6),
                "bpp": round(bpp, 6),
                "png_bytes": int(png_bytes) if not np.isnan(png_bytes) else "",
                "ratio_vs_png": round(ratio_vs_png, 4) if not np.isnan(ratio_vs_png) else "",
                "psnr_full": round(p_full, 4) if not math.isinf(p_full) else "inf",
                "ssim_full": round(s_full, 6),
                "psnr_background": round(p_bg, 4) if not math.isinf(p_bg) else "inf",
                "ssim_background": round(s_bg, 6),
                "psnr_roi": "inf" if math.isinf(p_roi) else round(p_roi, 4),
                "ssim_roi": round(s_roi, 6),
                "roi_lossless": roi_loss,
                "encode_time_s": round(enc_time, 6),
                "decode_time_s": round(dec_time, 6),
            })

    return rows


def compute_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics grouped by method and quality, overall and per source."""
    numeric_df = df.copy()
    for col in [
        "roi_fraction", "total_bytes", "header_bytes", "mask_bytes", "roi_bytes",
        "background_bytes", "compression_ratio", "bpp", "ratio_vs_png",
        "psnr_full", "ssim_full", "psnr_background", "ssim_background",
        "ssim_roi", "encode_time_s", "decode_time_s",
    ]:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col].replace("inf", np.nan), errors="coerce")

    numeric_df["roi_lossless_bool"] = numeric_df["roi_lossless"].apply(
        lambda x: True if str(x).strip().lower() in ("true", "1") else False
    )

    summary_rows = []
    groups = [("all", numeric_df)] + [(s, numeric_df[numeric_df["source"] == s]) for s in sorted(numeric_df["source"].unique())]

    for source_label, sub_df in groups:
        for method in ROI_METHODS:
            m_sub = sub_df[sub_df["method"] == method]
            for q in ROI_QUALITIES:
                q_sub = m_sub[m_sub["quality"] == q]
                if q_sub.empty:
                    continue
                summary_rows.append({
                    "source": source_label,
                    "method": method,
                    "quality": q,
                    "mean_bpp": round(float(q_sub["bpp"].mean()), 4),
                    "mean_cr": round(float(q_sub["compression_ratio"].mean()), 3),
                    "mean_ratio_vs_png": round(float(q_sub["ratio_vs_png"].dropna().mean()), 4) if not q_sub["ratio_vs_png"].dropna().empty else np.nan,
                    "mean_psnr_full": round(float(q_sub["psnr_full"].dropna().mean()), 2),
                    "mean_ssim_full": round(float(q_sub["ssim_full"].mean()), 4),
                    "mean_psnr_background": round(float(q_sub["psnr_background"].dropna().mean()), 2),
                    "mean_ssim_background": round(float(q_sub["ssim_background"].mean()), 4),
                    "mean_mask_bytes": round(float(q_sub["mask_bytes"].mean()), 1),
                    "mean_roi_bytes": round(float(q_sub["roi_bytes"].mean()), 1),
                    "mean_background_bytes": round(float(q_sub["background_bytes"].mean()), 1),
                    "mean_total_bytes": round(float(q_sub["total_bytes"].mean()), 1),
                    "roi_lossless_fraction": round(float(q_sub["roi_lossless_bool"].mean()), 4),
                    "mean_encode_time_s": round(float(q_sub["encode_time_s"].mean()), 4),
                    "mean_decode_time_s": round(float(q_sub["decode_time_s"].mean()), 4),
                })

    return pd.DataFrame(summary_rows)


def run_roi_codec(limit: Optional[int] = None, workers: int = 4, force: bool = False, fill: str = "inpaint") -> pd.DataFrame:
    """Run hybrid ROI codec benchmark across masked images."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    # Filter to masked images only
    df = df[df["has_mask"] == True].copy()
    print(f"Total available masked images: {len(df)}")

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
        print(f"Sampled {len(df)} images: Montgomery={sum(df['source']=='Montgomery')}, Shenzhen={sum(df['source']=='Shenzhen')}")

    items = df.to_dict(orient="records")

    existing_keys: Set[Tuple[str, str, int]] = set()
    existing_rows: List[Dict[str, Any]] = []
    if not force and ROI_RESULTS_CSV.is_file():
        try:
            prev_df = pd.read_csv(ROI_RESULTS_CSV)
            for _, r in prev_df.iterrows():
                existing_keys.add((str(r["stem"]), str(r["method"]), int(r["quality"])))
            existing_rows = prev_df.to_dict(orient="records")
            print(f"Resuming: found {len(existing_keys)} existing evaluations in {ROI_RESULTS_CSV}")
        except Exception as e:
            print(f"Could not read existing {ROI_RESULTS_CSV}: {e}. Starting fresh.")
    elif force:
        print("Force recomputation requested: starting fresh.")

    new_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    if workers <= 1:
        for item in tqdm(items, desc="Evaluating ROI Codec"):
            p_bytes = png_bytes_map.get(str(item["stem"]), np.nan)
            res = process_image_roi(item, p_bytes, fill=fill, existing_keys=existing_keys)
            new_results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_image_roi, item, png_bytes_map.get(str(item["stem"]), np.nan), fill, existing_keys)
                for item in items
            ]
            for fut in tqdm(as_completed(futures), total=len(items), desc="Evaluating ROI Codec"):
                res = fut.result()
                new_results.extend(res)

    total_time = time.perf_counter() - start_time
    print(f"ROI Codec evaluation completed in {total_time:.2f}s ({total_time / max(1, len(items)):.4f}s per image)")

    all_rows = existing_rows + new_results
    unique_rows: Dict[Tuple[str, str, int], Dict[str, Any]] = {
        (str(r["stem"]), str(r["method"]), int(r["quality"])): r for r in all_rows
    }

    df_out = pd.DataFrame(list(unique_rows.values()))
    df_out.sort_values(by=["stem", "method", "quality"], inplace=True)
    df_out.to_csv(ROI_RESULTS_CSV, index=False)
    print(f"Saved {len(df_out)} rows to {ROI_RESULTS_CSV}")

    # Summary
    summary_df = compute_roi_summary(df_out)
    summary_df.to_csv(ROI_SUMMARY_CSV, index=False)
    print(f"Saved summary to {ROI_SUMMARY_CSV}")

    exact_rate = df_out["roi_lossless"].apply(lambda x: str(x).lower() == "true").mean()
    print(f"Exact ROI lossless rate: {exact_rate * 100:.2f}%")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run hybrid ROI codec benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--force", action="store_true", help="Force recomputation without resume")
    parser.add_argument("--fill", type=str, default="inpaint", choices=["none", "mean", "inpaint"])
    args = parser.parse_args()

    run_roi_codec(limit=args.limit, workers=args.workers, force=args.force, fill=args.fill)


if __name__ == "__main__":
    main()
