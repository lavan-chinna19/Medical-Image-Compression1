"""Out-of-fold inference with 8-pass Monte Carlo dropout uncertainty estimation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
import torch
from torch.amp import autocast
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, WORK_DIR
from medcomp.io_utils import load_image, save_image
from medcomp.seg_data import inverse_transform, pad_to_square
from medcomp.unet import UNet, enable_mc_dropout

UNET_DIR = WORK_DIR / "unet"
PRED_DIR = WORK_DIR / "pred"
PROB_DIR = PRED_DIR / "prob"
UNC_DIR = PRED_DIR / "unc"
MASK_DIR = PRED_DIR / "mask"


def load_fold_model(fold_idx: int, device: torch.device) -> UNet:
    """Load trained U-Net checkpoint for a specific fold."""
    ckpt_path = UNET_DIR / f"fold{fold_idx}.pt"
    assert ckpt_path.is_file(), f"Checkpoint not found for fold {fold_idx}: {ckpt_path}"
    model = UNet(in_channels=1, out_channels=1, base_channels=32)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def postprocess_binary_mask(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold probability map, keep two largest connected components, and fill holes.

    Args:
        prob_map: 2D float array in [0, 1].
        threshold: Binarization threshold.

    Returns:
        np.ndarray: 2D uint8 mask (0 or 255).
    """
    binary = (prob_map > threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Sort foreground components (ignore background label 0)
        fg_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        keep_labels = fg_labels[:min(2, len(fg_labels))]
        cleaned = np.isin(labels, keep_labels)
    else:
        cleaned = binary.astype(bool)

    filled = binary_fill_holes(cleaned)
    return (filled.astype(np.uint8)) * 255


@torch.no_grad()
def predict_mc_dropout(
    models: list[UNet],
    img_256: torch.Tensor,
    n_passes: int = 8,
    device: torch.device = torch.device("cuda:0"),
) -> tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo dropout passes across models and compute mean probability and uncertainty.

    Args:
        models: List of U-Net models (1 model for out-of-fold, 5 models for unmasked ensemble).
        img_256: Input tensor of shape (1, 1, 256, 256) on device.
        n_passes: Number of MC dropout forward passes per model.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Mean probability map (256, 256).
            - Standard deviation uncertainty map (256, 256).
    """
    all_prob_passes: list[np.ndarray] = []

    for model in models:
        model.eval()
        enable_mc_dropout(model)

        for _ in range(n_passes):
            with autocast("cuda", dtype=torch.float16):
                logits = model(img_256)
                probs = torch.sigmoid(logits)
            p_arr = probs.squeeze().cpu().numpy().astype(np.float32)
            all_prob_passes.append(p_arr)

    stack = np.stack(all_prob_passes, axis=0)
    mean_prob = np.mean(stack, axis=0)
    unc_map = np.std(stack, axis=0)

    return mean_prob, unc_map


def run_prediction() -> None:
    """Generate out-of-fold and ensemble predictions for all 800 dataset images."""
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is NOT available for prediction.")

    device = torch.device("cuda:0")
    print(f"Prediction hardware: GPU '{torch.cuda.get_device_name(0)}'")

    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    PROB_DIR.mkdir(parents=True, exist_ok=True)
    UNC_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Preload all 5 models
    fold_models = [load_fold_model(k, device) for k in range(5)]
    print("Loaded all 5 fold models successfully.")

    n_processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting U-Net"):
        stem = str(row["stem"])
        fold = int(row["fold"])
        has_mask = bool(row["has_mask"])

        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        img = load_image(img_path)
        orig_h, orig_w = img.shape

        # Preprocess
        padded_img, _ = pad_to_square(img)
        res_img = cv2.resize(padded_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img_t = torch.from_numpy(res_img).unsqueeze(0).unsqueeze(0).to(device)

        if has_mask:
            # Out-of-fold prediction: only the model where this image was held out (fold k)
            models_to_use = [fold_models[fold]]
        else:
            # Unmasked image: average of all 5 fold models
            models_to_use = fold_models

        prob_256, unc_256 = predict_mc_dropout(models_to_use, img_t, n_passes=8, device=device)

        # Map back to original processed dimensions
        prob_orig = inverse_transform(prob_256, orig_h, orig_w)
        unc_orig = inverse_transform(unc_256, orig_h, orig_w)

        # Binary postprocessed mask
        mask_orig = postprocess_binary_mask(prob_orig, threshold=0.5)

        # Save 8-bit images
        prob_8bit = np.clip(np.round(prob_orig * 255.0), 0, 255).astype(np.uint8)
        unc_8bit = np.clip(np.round(unc_orig * 255.0), 0, 255).astype(np.uint8)

        save_image(PROB_DIR / f"{stem}.png", prob_8bit)
        save_image(UNC_DIR / f"{stem}.png", unc_8bit)
        save_image(MASK_DIR / f"{stem}.png", mask_orig)

        n_processed += 1

    print(f"\nGenerated predictions for all {n_processed} images in {PRED_DIR}")


if __name__ == "__main__":
    run_prediction()
