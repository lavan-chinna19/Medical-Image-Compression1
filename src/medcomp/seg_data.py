"""PyTorch Dataset and spatial preprocessing for lung segmentation.

Features:
- Symmetric square zero-padding preserving aspect ratio, followed by 256x256 resizing.
- Training augmentations:
  - Random affine: rotation in [-7, 7] deg, scale in [0.9, 1.1], translation in [-5%, 5%].
  - Photometric jitter: brightness and contrast scaling.
  - Strictly NO horizontal flips (chest anatomy is lateralized: cardiac notch on left).
- Inverse transformation taking 256x256 probability maps back to original processed dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DATA_DIR, PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR
from .io_utils import load_image


def pad_to_square(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    """Pad a 2D image to a symmetric square with zeros.

    Args:
        img: 2D numpy array of shape (H, W).

    Returns:
        tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
            - Padded square image of shape (S, S) where S = max(H, W).
            - Offsets tuple: (pad_top, pad_bottom, pad_left, pad_right, orig_h, orig_w).
    """
    orig_h, orig_w = img.shape[:2]
    s = max(orig_h, orig_w)
    pad_h = s - orig_h
    pad_w = s - orig_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 2:
        padded = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    offsets = (pad_top, pad_bottom, pad_left, pad_right, orig_h, orig_w)
    return padded, offsets


def inverse_transform(
    prob_map: np.ndarray | torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Transform a 256x256 probability map back to the original processed dimensions.

    Resizes the probability map to (S, S) using bilinear interpolation, then crops out
    the symmetric padding to restore (orig_h, orig_w).

    Args:
        prob_map: 2D float array or tensor of shape (256, 256) with values in [0, 1].
        orig_h: Original height in pixels.
        orig_w: Original width in pixels.

    Returns:
        np.ndarray: Probability map of shape (orig_h, orig_w) with float32 values in [0, 1].
    """
    if isinstance(prob_map, torch.Tensor):
        arr = prob_map.detach().cpu().numpy().squeeze()
    else:
        arr = np.asarray(prob_map, dtype=np.float32).squeeze()

    s = max(orig_h, orig_w)
    # 1. Bilinear resize back to square (S, S)
    square_prob = cv2.resize(arr, (s, s), interpolation=cv2.INTER_LINEAR)

    # 2. Crop out symmetric zero-padding
    pad_h = s - orig_h
    pad_w = s - orig_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    cropped = square_prob[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
    return np.clip(cropped.astype(np.float32), 0.0, 1.0)


def apply_training_augmentation(
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply affine and photometric augmentations to 256x256 image and mask.

    - Rotation: [-7, 7] deg
    - Scale: [0.9, 1.1]
    - Translation: [-5%, +5%]
    - Brightness jitter: * [0.85, 1.15]
    - Contrast jitter: (x - 0.5) * [0.85, 1.15] + 0.5
    - NO horizontal flip.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(-7.0, 7.0)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.05 * w, 0.05 * w)
    ty = rng.uniform(-0.05 * h, 0.05 * h)

    center = (w / 2.0, h / 2.0)
    m_rot = cv2.getRotationMatrix2D(center, angle, scale)
    m_rot[0, 2] += tx
    m_rot[1, 2] += ty

    # Warp image (bilinear) and mask (nearest)
    aug_img = cv2.warpAffine(img, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aug_mask = cv2.warpAffine(mask, m_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Photometric jitter on float [0, 1]
    b_factor = rng.uniform(0.85, 1.15)
    c_factor = rng.uniform(0.85, 1.15)

    aug_img = aug_img * b_factor
    aug_img = (aug_img - 0.5) * c_factor + 0.5
    aug_img = np.clip(aug_img, 0.0, 1.0)

    return aug_img, aug_mask


class LungSegDataset(Dataset):
    """PyTorch Dataset over data/manifest.csv and work/processed for U-Net training/inference."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        is_train: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            manifest_df: Pandas DataFrame containing dataset rows (must include 'stem', 'has_mask').
            is_train: If True, enables stochastic training augmentations.
            seed: Random seed for augmentations.
        """
        self.df = manifest_df.reset_index(drop=True)
        self.is_train = is_train
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        stem = str(row["stem"])
        has_mask = bool(row["has_mask"])

        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        img = load_image(img_path)

        if has_mask:
            mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
            mask = (load_image(mask_path) > 127).astype(np.float32)
        else:
            mask = np.zeros(img.shape, dtype=np.float32)

        # Pad to symmetric square
        padded_img, _ = pad_to_square(img)
        padded_mask, _ = pad_to_square(mask)

        # Resize to 256x256
        res_img = cv2.resize(padded_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        res_mask = cv2.resize(padded_mask, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        if self.is_train:
            res_img, res_mask = apply_training_augmentation(res_img, res_mask, self.rng)

        # Shape: (1, 256, 256)
        img_t = torch.from_numpy(res_img).unsqueeze(0).float()
        mask_t = torch.from_numpy(res_mask).unsqueeze(0).float()

        return img_t, mask_t, stem
