"""Region ablation experiment: assess classifier dependency on lung fields vs background.

Builds three composites for each image and DWT quality in [10, 30, 50, 70]:
- lungs_lossy: original outside dilated mask, lossy L inside.
- background_lossy: lossy L outside dilated mask, original inside.
- all_lossy: lossy L everywhere.

Classifies all variants on GPU with the patient's held-out fold model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import load_model, predict_proba
from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    RESULTS_DIR,
    WORK_DIR,
)
from medcomp.dwt_codec import dwt_encode
from medcomp.io_utils import load_image

MASK_DIR = WORK_DIR / "pred" / "mask"
OUT_CSV = RESULTS_DIR / "region_ablation_per_image.csv"

QUALITIES = [10, 30, 50, 70]
VARIANTS = ["lungs_lossy", "background_lossy", "all_lossy"]

COLUMNS = [
    "stem",
    "source",
    "label",
    "fold",
    "variant",
    "quality",
    "fraction_of_pixels_replaced",
    "prob_orig",
    "prob_dec",
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


def generate_image_composites(
    item: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, int, float, np.ndarray]], np.ndarray]:
    """Generate all ablation composite images for a single patient X-ray.

    Executed in worker process on CPU.

    Returns:
        tuple: (item metadata, list of (variant, quality, fraction_replaced, composite_img), orig_img)
    """
    stem = str(item["stem"])
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = MASK_DIR / f"{stem}.png"

    img = load_image(img_path)
    mask = load_image(mask_path)
    dilated_m = dilate_mask(mask > 127, d_pixels=4)

    m_frac = float(dilated_m.mean())
    bg_frac = float(1.0 - m_frac)

    composites: list[tuple[str, int, float, np.ndarray]] = []

    for q in QUALITIES:
        # Encode with tuned DWT to get lossy image L
        _, lossy_L = dwt_encode(img, quality=q)

        # 1. lungs_lossy: original outside mask, lossy L inside mask
        lungs_lossy = img.copy()
        lungs_lossy[dilated_m] = lossy_L[dilated_m]
        composites.append(("lungs_lossy", q, m_frac, lungs_lossy))

        # 2. background_lossy: lossy L outside mask, original inside mask
        bg_lossy = lossy_L.copy()
        bg_lossy[dilated_m] = img[dilated_m]
        composites.append(("background_lossy", q, bg_frac, bg_lossy))

        # 3. all_lossy: lossy L everywhere
        composites.append(("all_lossy", q, 1.0, lossy_L))

    return item, composites, img


def main() -> None:
    parser = argparse.ArgumentParser(description="Run region ablation for TB classifier.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU worker processes (default: 4).")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU inference batch size (default: 32).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for classifier evaluation on RTX 2050!")
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_stems: set[str] = set()
    write_header = not OUT_CSV.is_file()

    if OUT_CSV.is_file():
        existing_df = pd.read_csv(OUT_CSV)
        if not existing_df.empty and "stem" in existing_df.columns:
            counts = existing_df["stem"].value_counts()
            # 4 qualities * 3 variants = 12 rows per image
            completed_stems = set(counts[counts >= 12].index)
            print(f"Resuming: found {len(completed_stems)} already completed images in {OUT_CSV.name}")

    pending_items = [
        row.to_dict()
        for _, row in manifest_df.iterrows()
        if str(row["stem"]) not in completed_stems
    ]
    print(f"Total images to process: {len(pending_items)}")

    if not pending_items:
        print("All images are already processed!")
        return

    csv_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    t_start = time.time()
    n_processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for item, composites, orig_img in tqdm(
            executor.map(generate_image_composites, pending_items),
            total=len(pending_items),
            desc="Region Ablation",
        ):
            stem = str(item["stem"])
            source = str(item["source"])
            label = int(item["label"])
            fold = int(item["fold"])

            model = fold_models[fold]

            # List of images for batch inference: [orig_img, comp_0, comp_1, ...]
            # Total = 1 + 12 = 13 images
            images_to_predict = [orig_img] + [c[3] for c in composites]

            probs = predict_proba(
                model,
                images_to_predict,
                device=device,
                batch_size=args.batch_size,
            )

            prob_orig = float(probs[0])
            pred_orig = int(prob_orig >= 0.5)

            # Assert prob_orig matches classifier_oof.csv within 1e-3
            expected_oof = oof_probs[stem]
            assert abs(prob_orig - expected_oof) < 1e-3, (
                f"Mismatch for {stem}: prob_orig={prob_orig:.5f} vs OOF={expected_oof:.5f}"
            )

            # Record rows for the 12 composites
            for idx, (variant, quality, frac_replaced, _) in enumerate(composites):
                prob_dec = float(probs[idx + 1])
                pred_dec = int(prob_dec >= 0.5)
                flip = int(pred_dec != pred_orig)
                abs_prob_diff = float(abs(prob_dec - prob_orig))

                row_dict = {
                    "stem": stem,
                    "source": source,
                    "label": label,
                    "fold": fold,
                    "variant": variant,
                    "quality": quality,
                    "fraction_of_pixels_replaced": frac_replaced,
                    "prob_orig": prob_orig,
                    "prob_dec": prob_dec,
                    "flip": flip,
                    "abs_prob_diff": abs_prob_diff,
                }
                writer.writerow(row_dict)

            csv_file.flush()
            n_processed += 1

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\nRegion ablation completed for {n_processed} images in {elapsed:.2f} seconds ({elapsed/max(n_processed,1):.3f} s/image).")
    print(f"Saved results to {OUT_CSV}")


if __name__ == "__main__":
    main()
