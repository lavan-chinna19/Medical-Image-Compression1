"""Task-based benchmark: encode/decode every image across all codecs and classify on GPU."""

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
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.baselines import jpeg_codec, jpeg2000_codec
from medcomp.classifier import load_model, predict_proba
from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    RESULTS_DIR,
    WORK_DIR,
    compute_original_bytes,
)
from medcomp.dct_codec import dct_encode
from medcomp.dwt_codec import dwt_encode
from medcomp.io_utils import load_image, save_image
from medcomp.roi_codec import roi_encode

MASK_DIR = WORK_DIR / "pred" / "mask"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_per_image.csv"

# Compression settings per codec
JPEG_QUALITIES = [10, 20, 30, 50, 70, 90]
JPEG2000_RATIOS = [5, 10, 20, 40, 80]
DCT_QUALITIES = [10, 20, 30, 50, 70, 90]
DWT_QUALITIES = [10, 20, 30, 50, 70, 90]
ROI_DWT_QUALITIES = [10, 20, 30, 50, 70, 90]

BENCHMARK_COLUMNS = [
    "stem",
    "source",
    "label",
    "fold",
    "codec",
    "setting",
    "total_bytes",
    "bpp",
    "prob_orig",
    "prob_dec",
    "pred_orig",
    "pred_dec",
    "flip",
    "abs_prob_diff",
]


def dilate_mask(mask: np.ndarray, d_pixels: int = 4) -> np.ndarray:
    """Dilate binary mask by radius d_pixels."""
    if d_pixels <= 0:
        return mask
    k_size = 2 * d_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def encode_decode_single_image(
    item: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, Any, int, np.ndarray]]]:
    """Encode and decode all compression variants for a single image.

    Executed in a worker process on CPU.

    Returns:
        tuple: (item metadata, list of (codec, setting, total_bytes, recon_img))
    """
    stem = str(item["stem"])
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = MASK_DIR / f"{stem}.png"

    img = load_image(img_path)
    mask = load_image(mask_path)
    dilated_m = dilate_mask(mask > 127, d_pixels=4)

    h, w = img.shape
    raw_bytes = int(h * w)

    variants: list[tuple[str, Any, int, np.ndarray]] = []

    # 1. Original (uncompressed reference)
    variants.append(("original", 0, raw_bytes, img))

    # 2. JPEG
    for q in JPEG_QUALITIES:
        c_bytes, recon = jpeg_codec(img, quality=q)
        variants.append(("jpeg", q, len(c_bytes), recon))

    # 3. JPEG2000
    for r in JPEG2000_RATIOS:
        c_bytes, recon = jpeg2000_codec(img, ratio=r)
        variants.append(("jpeg2000", r, len(c_bytes), recon))

    # 4. Custom DCT
    for q in DCT_QUALITIES:
        c_bytes, recon = dct_encode(img, quality=q)
        variants.append(("dct", q, len(c_bytes), recon))

    # 5. Tuned DWT
    for q in DWT_QUALITIES:
        c_bytes, recon = dwt_encode(img, quality=q)
        variants.append(("dwt", q, len(c_bytes), recon))

    # 6. ROI-DWT (hybrid lossless ROI + lossy background)
    for q in ROI_DWT_QUALITIES:
        c_bytes, recon, _ = roi_encode(img, dilated_m, method="dwt", quality=q, fill="inpaint")
        variants.append(("roi_dwt", q, len(c_bytes), recon))

    return item, variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-based compression benchmark.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU worker processes (default: 4).")
    parser.add_argument("--limit", type=int, default=None, help="Stratified sample limit of images to process.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size on GPU (default: 32).")
    parser.add_argument("--cache-decoded", action="store_true", help="Save decoded images to disk.")
    args = parser.parse_args()

    # Hardware check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for classifier inference on RTX 2050!")
    device = torch.device("cuda")
    print(f"Inference device: {torch.cuda.get_device_name(device)} with mixed precision.")

    manifest_path = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    oof_path = RESULTS_DIR / "classifier_oof.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Classifier OOF predictions not found: {oof_path}")
    oof_df = pd.read_csv(oof_path)
    oof_probs = dict(zip(oof_df["stem"], oof_df["prob_tb"]))

    # Preload all 5 fold models onto GPU
    print("Loading 5 fold models onto GPU...")
    fold_models = [load_model(k, device=device) for k in range(5)]
    for m in fold_models:
        m.eval()

    # Apply stratified limit if specified
    if args.limit is not None and args.limit < len(manifest_df):
        strat_key = manifest_df["source"] + "_" + manifest_df["label"].astype(str)
        selected_df, _ = train_test_split(
            manifest_df,
            train_size=args.limit,
            stratify=strat_key,
            random_state=42,
        )
        selected_df = selected_df.reset_index(drop=True)
        print(f"Selected {len(selected_df)} images stratified by source and label (limit={args.limit}).")
    else:
        selected_df = manifest_df.copy().reset_index(drop=True)

    # Resume support
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_stems: set[str] = set()
    write_header = not BENCHMARK_CSV.is_file()

    if BENCHMARK_CSV.is_file():
        existing_df = pd.read_csv(BENCHMARK_CSV)
        if not existing_df.empty and "stem" in existing_df.columns:
            counts = existing_df["stem"].value_counts()
            # An image has 30 variants (1 orig + 6 jpeg + 5 j2k + 6 dct + 6 dwt + 6 roi)
            completed_stems = set(counts[counts >= 30].index)
            print(f"Resuming: found {len(completed_stems)} already completed images in {BENCHMARK_CSV.name}")

    pending_items = [
        row.to_dict()
        for _, row in selected_df.iterrows()
        if str(row["stem"]) not in completed_stems
    ]
    print(f"Total images to process in this run: {len(pending_items)}")

    if not pending_items:
        print("All requested images are already processed!")
        return

    csv_file = open(BENCHMARK_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=BENCHMARK_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    if args.cache_decoded:
        decoded_cache_dir = WORK_DIR / "decoded"
        decoded_cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    n_processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for item, variants in tqdm(
            executor.map(encode_decode_single_image, pending_items),
            total=len(pending_items),
            desc="Benchmarking Codecs",
        ):
            stem = str(item["stem"])
            source = str(item["source"])
            label = int(item["label"])
            fold = int(item["fold"])

            # Model for this image is strictly the model of the fold where it was held out
            model = fold_models[fold]

            # Collect uint8 images for batch inference
            recon_imgs = [v[3] for v in variants]
            h, w = recon_imgs[0].shape
            pixel_count = h * w

            # Batch predict on GPU
            probs = predict_proba(
                model,
                recon_imgs,
                device=device,
                batch_size=args.batch_size,
            )

            # Variant 0 is the original uncompressed image
            prob_orig = float(probs[0])
            pred_orig = int(prob_orig >= 0.5)

            # Assert prob_orig matches classifier_oof.csv within 1e-3
            expected_oof_prob = oof_probs[stem]
            assert abs(prob_orig - expected_oof_prob) < 1e-3, (
                f"Mismatch for {stem}: prob_orig={prob_orig:.5f} vs OOF={expected_oof_prob:.5f}"
            )

            # Process and write all rows for this image
            for v_idx, (codec, setting, total_bytes, recon_img) in enumerate(variants):
                prob_dec = float(probs[v_idx])
                pred_dec = int(prob_dec >= 0.5)
                flip = int(pred_dec != pred_orig)
                abs_prob_diff = float(abs(prob_dec - prob_orig))
                bpp = float((total_bytes * 8.0) / pixel_count)

                row_dict = {
                    "stem": stem,
                    "source": source,
                    "label": label,
                    "fold": fold,
                    "codec": codec,
                    "setting": setting,
                    "total_bytes": total_bytes,
                    "bpp": bpp,
                    "prob_orig": prob_orig,
                    "prob_dec": prob_dec,
                    "pred_orig": pred_orig,
                    "pred_dec": pred_dec,
                    "flip": flip,
                    "abs_prob_diff": abs_prob_diff,
                }
                writer.writerow(row_dict)

                if args.cache_decoded:
                    c_dir = WORK_DIR / "decoded" / f"{codec}_{setting}"
                    c_dir.mkdir(parents=True, exist_ok=True)
                    save_image(c_dir / f"{stem}.png", recon_img)

            csv_file.flush()
            n_processed += 1

    csv_file.close()
    elapsed = time.time() - t_start
    time_per_img = elapsed / max(n_processed, 1)
    projected_total = (time_per_img * len(manifest_df)) / 60.0

    print(f"\nCompleted {n_processed} images in {elapsed:.2f} seconds ({time_per_img:.3f} s/image).")
    print(f"Projected total time for all {len(manifest_df)} images: {projected_total:.2f} minutes.")


if __name__ == "__main__":
    main()
