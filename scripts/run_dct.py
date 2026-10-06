"""Benchmark custom DCT codec across dataset images with resume capability."""

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
from medcomp.dct_codec import dct_decode, dct_encode_with_stats
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

DCT_QUALITIES = [10, 20, 30, 50, 70, 90]
DCT_RESULTS_CSV = RESULTS_DIR / "dct_results.csv"


def process_image_dct(item_info: Dict[str, Any], existing_keys: Set[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """Evaluate all DCT quality settings for a single image."""
    stem = item_info["stem"]
    source = item_info["source"]
    label = int(item_info["label"])
    fold = int(item_info["fold"])
    has_mask = bool(item_info["has_mask"])

    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    img = load_image(img_path)

    mask = None
    if has_mask:
        mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
        mask = (load_image(mask_path) > 127)

    orig_bytes = compute_original_bytes(img)
    num_pixels = img.size

    results = []
    for q in DCT_QUALITIES:
        if (stem, q) in existing_keys:
            continue

        # Time encoding
        t0 = time.perf_counter()
        comp_bytes, recon, stats = dct_encode_with_stats(img, quality=q)
        enc_time = time.perf_counter() - t0

        # Time decoding
        t1 = time.perf_counter()
        _ = dct_decode(comp_bytes)
        dec_time = time.perf_counter() - t1

        c_len = len(comp_bytes)
        cr = compression_ratio(orig_bytes, c_len)
        bpp = bits_per_pixel(c_len, num_pixels)
        p_full = psnr(img, recon)
        s_full = ssim(img, recon)

        if has_mask and mask is not None:
            p_roi = masked_psnr(img, recon, mask, region="roi")
            p_bg = masked_psnr(img, recon, mask, region="background")
            s_roi = masked_ssim(img, recon, mask, region="roi")
            s_bg = masked_ssim(img, recon, mask, region="background")
            roi_loss = is_lossless(img, recon, mask=mask)
        else:
            p_roi = float("nan")
            p_bg = float("nan")
            s_roi = float("nan")
            s_bg = float("nan")
            roi_loss = float("nan")

        results.append({
            "stem": stem,
            "source": source,
            "label": label,
            "fold": fold,
            "has_mask": has_mask,
            "codec": "DCT",
            "setting": q,
            "orig_bytes": orig_bytes,
            "compressed_bytes": c_len,
            "compression_ratio": round(cr, 6),
            "bpp": round(bpp, 6),
            "psnr_full": round(p_full, 4) if not math.isinf(p_full) else "inf",
            "ssim_full": round(s_full, 6),
            "psnr_roi": round(p_roi, 4) if (not math.isnan(p_roi) and not math.isinf(p_roi)) else ("inf" if math.isinf(p_roi) else ""),
            "psnr_background": round(p_bg, 4) if (not math.isnan(p_bg) and not math.isinf(p_bg)) else ("inf" if math.isinf(p_bg) else ""),
            "ssim_roi": round(s_roi, 6) if not math.isnan(s_roi) else "",
            "ssim_background": round(s_bg, 6) if not math.isnan(s_bg) else "",
            "roi_lossless": roi_loss if not math.isnan(roi_loss) else "",
            "dc_entropy": stats["dc_entropy"],
            "dc_avg_code_length": stats["dc_avg_code_length"],
            "ac_entropy": stats["ac_entropy"],
            "ac_avg_code_length": stats["ac_avg_code_length"],
            "encode_time_s": round(enc_time, 6),
            "decode_time_s": round(dec_time, 6),
        })

    return results


def run_dct(limit: Optional[int] = None, workers: int = 4, force: bool = False) -> pd.DataFrame:
    """Run custom DCT benchmark across images with resume capability."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

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
    existing_keys: Set[Tuple[str, int]] = set()
    existing_rows: List[Dict[str, Any]] = []
    if not force and DCT_RESULTS_CSV.is_file():
        try:
            prev_df = pd.read_csv(DCT_RESULTS_CSV)
            if "ssim_roi" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    existing_keys.add((str(r["stem"]), int(r["setting"])))
                existing_rows = prev_df.to_dict(orient="records")
                print(f"Resuming: found {len(existing_keys)} existing evaluations in {DCT_RESULTS_CSV}")
            else:
                print(f"Existing {DCT_RESULTS_CSV} missing ssim_roi column; recomputing from scratch.")
        except Exception as e:
            print(f"Could not read existing {DCT_RESULTS_CSV}: {e}. Starting fresh.")
    elif force:
        print("Force recomputation requested: starting fresh.")

    new_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    if workers <= 1:
        for item in tqdm(items, desc="Evaluating DCT"):
            res = process_image_dct(item, existing_keys)
            new_results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_image_dct, item, existing_keys) for item in items]
            for fut in tqdm(as_completed(futures), total=len(items), desc="Evaluating DCT"):
                res = fut.result()
                new_results.extend(res)

    total_time = time.perf_counter() - start_time
    print(f"DCT execution completed in {total_time:.2f}s ({total_time / max(1, len(items)):.4f}s per image)")

    # Combine and save
    all_rows = existing_rows + new_results
    unique_rows = {}
    for r in all_rows:
        key = (str(r["stem"]), int(r["setting"]))
        unique_rows[key] = r

    df_out = pd.DataFrame(list(unique_rows.values()))
    df_out.sort_values(by=["stem", "setting"], inplace=True)
    df_out.to_csv(DCT_RESULTS_CSV, index=False)
    print(f"Saved {len(df_out)} DCT evaluation rows to {DCT_RESULTS_CSV}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run custom DCT codec benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to evaluate")
    parser.add_argument("--force", action="store_true", help="Force recomputation without resume")
    args = parser.parse_args()

    run_dct(limit=args.limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
