"""Run standard compression baselines (JPEG, JPEG 2000, PNG) on dataset images."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.baselines import (
    check_jpeg2000_support,
    jpeg2000_codec,
    jpeg_codec,
    png_codec,
)
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
    mse,
    psnr,
    ssim,
)

# Standard evaluation settings
JPEG_QUALITIES = [10, 20, 30, 50, 70, 90]
JPEG2000_RATIOS = [5, 10, 20, 40, 80]
PNG_SETTINGS = ["lossless"]

BASELINES_CSV = RESULTS_DIR / "baselines.csv"


def encode_decode_timed(codec: str, img: np.ndarray, setting: Any) -> Tuple[bytes, np.ndarray, float, float]:
    """Encode and decode with separate high-resolution timing."""
    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()

    # Time encode
    t0 = time.perf_counter()
    if codec == "JPEG":
        im.save(buf, format="JPEG", quality=int(setting))
    elif codec == "JPEG2000":
        im.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(setting)], irreversible=True)
    elif codec == "PNG":
        im.save(buf, format="PNG", compress_level=9)
    else:
        raise ValueError(f"Unknown codec: {codec}")
    enc_time = time.perf_counter() - t0

    comp_bytes = buf.getvalue()
    buf.seek(0)

    # Time decode
    t1 = time.perf_counter()
    with Image.open(buf) as rec_im:
        reconstructed = np.array(rec_im, dtype=np.uint8)
    dec_time = time.perf_counter() - t1

    return comp_bytes, reconstructed, enc_time, dec_time


def process_image_baselines(item_info: Dict[str, Any], existing_keys: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    """Evaluate all baseline configurations for a single image."""
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

    configs = []
    for q in JPEG_QUALITIES:
        configs.append(("JPEG", str(q)))
    for r in JPEG2000_RATIOS:
        configs.append(("JPEG2000", str(r)))
    for p in PNG_SETTINGS:
        configs.append(("PNG", str(p)))

    results = []
    for codec, setting in configs:
        if (stem, codec, setting) in existing_keys:
            continue

        comp_bytes, recon, enc_time, dec_time = encode_decode_timed(codec, img, setting)

        c_len = len(comp_bytes)
        cr = compression_ratio(orig_bytes, c_len)
        bpp = bits_per_pixel(c_len, num_pixels)
        p_full = psnr(img, recon)
        s_full = ssim(img, recon)

        if has_mask and mask is not None:
            p_roi = masked_psnr(img, recon, mask, region="roi")
            p_bg = masked_psnr(img, recon, mask, region="background")
            roi_loss = is_lossless(img, recon, mask=mask)
        else:
            p_roi = float("nan")
            p_bg = float("nan")
            roi_loss = float("nan")

        results.append({
            "stem": stem,
            "source": source,
            "label": label,
            "fold": fold,
            "has_mask": has_mask,
            "codec": codec,
            "setting": setting,
            "orig_bytes": orig_bytes,
            "compressed_bytes": c_len,
            "compression_ratio": round(cr, 6),
            "bpp": round(bpp, 6),
            "psnr_full": round(p_full, 4) if not math.isinf(p_full) else "inf",
            "ssim_full": round(s_full, 6),
            "psnr_roi": round(p_roi, 4) if (not math.isnan(p_roi) and not math.isinf(p_roi)) else ("inf" if math.isinf(p_roi) else ""),
            "psnr_background": round(p_bg, 4) if (not math.isnan(p_bg) and not math.isinf(p_bg)) else ("inf" if math.isinf(p_bg) else ""),
            "roi_lossless": roi_loss if not math.isnan(roi_loss) else "",
            "encode_time_s": round(enc_time, 6),
            "decode_time_s": round(dec_time, 6),
        })

    return results


def run_baselines(limit: Optional[int] = None, workers: int = 4) -> pd.DataFrame:
    """Run baseline benchmark across images with resume capability."""
    if not check_jpeg2000_support():
        raise RuntimeError("Pillow build lacks JPEG 2000 (OpenJPEG) support.")

    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    # Stratified subset if limit specified
    if limit is not None and limit < len(df):
        print(f"Selecting stratified subset of {limit} images (source + label, seed=42)...")
        df["strat_group"] = df["source"].astype(str) + "_" + df["label"].astype(str)
        # Proportionate stratified sampling
        sampled_indices = []
        rng = np.random.default_rng(42)
        for _, group in df.groupby("strat_group"):
            n_group = max(1, int(round(limit * len(group) / len(df))))
            chosen = rng.choice(group.index.values, size=min(n_group, len(group)), replace=False)
            sampled_indices.extend(chosen)
        # Adjust to exact limit
        if len(sampled_indices) > limit:
            sampled_indices = sampled_indices[:limit]
        elif len(sampled_indices) < limit:
            remaining = [idx for idx in df.index if idx not in sampled_indices]
            sampled_indices.extend(rng.choice(remaining, size=limit - len(sampled_indices), replace=False))
        df = df.loc[sampled_indices].copy()
        print(f"Sampled {len(df)} images: Montgomery={sum(df['source']=='Montgomery')}, Shenzhen={sum(df['source']=='Shenzhen')}")

    items = df.to_dict(orient="records")

    # Resume check
    existing_keys: Set[Tuple[str, str, str]] = set()
    existing_rows: List[Dict[str, Any]] = []
    if BASELINES_CSV.is_file():
        try:
            prev_df = pd.read_csv(BASELINES_CSV)
            for _, r in prev_df.iterrows():
                existing_keys.add((str(r["stem"]), str(r["codec"]), str(r["setting"])))
            existing_rows = prev_df.to_dict(orient="records")
            print(f"Resuming: found {len(existing_keys)} existing evaluations in {BASELINES_CSV}")
        except Exception as e:
            print(f"Could not read existing {BASELINES_CSV}: {e}. Starting fresh.")

    new_results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    if workers <= 1:
        for item in tqdm(items, desc="Evaluating baselines"):
            res = process_image_baselines(item, existing_keys)
            new_results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_image_baselines, item, existing_keys) for item in items]
            for fut in tqdm(as_completed(futures), total=len(items), desc="Evaluating baselines"):
                res = fut.result()
                new_results.extend(res)

    total_time = time.perf_counter() - start_time
    print(f"Execution completed in {total_time:.2f}s ({total_time / max(1, len(items)):.4f}s per image)")

    # Combine and save
    all_rows = existing_rows + new_results
    # Deduplicate in case of resume overlaps
    unique_rows = {}
    for r in all_rows:
        key = (str(r["stem"]), str(r["codec"]), str(r["setting"]))
        unique_rows[key] = r

    df_out = pd.DataFrame(list(unique_rows.values()))
    # Sort for clean presentation
    df_out.sort_values(by=["stem", "codec", "setting"], inplace=True)
    df_out.to_csv(BASELINES_CSV, index=False)
    print(f"Saved {len(df_out)} baseline rows to {BASELINES_CSV}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run compression baselines benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to evaluate")
    args = parser.parse_args()

    run_baselines(limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
