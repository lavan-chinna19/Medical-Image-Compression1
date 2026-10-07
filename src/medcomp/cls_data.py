"""PyTorch Dataset and preprocessing for tuberculosis (TB) chest X-ray classification.

Features:
- Aspect-ratio preserving square zero-padding, resized to 320x320.
- Replicated from 1 channel (grayscale) to 3 channels (RGB).
- ImageNet mean and std normalization:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
- Training augmentations:
    - Small rotation: [-7, 7] deg
    - Scale: [0.9, 1.1]
    - Translation: [-5%, 5%]
    - Brightness and contrast jitter
    - Strictly NO horizontal flips (anatomy is side-specific)
- Supports both disk-based images and in-memory uint8 arrays for downstream codec evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DATA_DIR, PROCESSED_IMAGES_DIR
from .io_utils import load_image
from .seg_data import pad_to_square

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_classification_augmentation(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply spatial and photometric augmentations to 320x320 float image in [0, 1].

    - Rotation: [-7, 7] deg
    - Scale: [0.9, 1.1]
    - Translation: [-5%, 5%]
    - Brightness jitter: * [0.85, 1.15]
    - Contrast jitter: (x - 0.5) * [0.85, 1.15] + 0.5
    - NO horizontal flips.
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

    aug_img = cv2.warpAffine(
        img,
        m_rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    b_factor = rng.uniform(0.85, 1.15)
    c_factor = rng.uniform(0.85, 1.15)

    aug_img = aug_img * b_factor
    aug_img = (aug_img - 0.5) * c_factor + 0.5
    return np.clip(aug_img, 0.0, 1.0)


def preprocess_for_classifier(
    img: np.ndarray,
    is_train: bool = False,
    rng: Optional[np.random.Generator] = None,
    target_size: int = 320,
) -> torch.Tensor:
    """Preprocess a 2D uint8 grayscale image into a normalized 3-channel PyTorch tensor.

    Steps:
    1. Pad to square with zeros preserving aspect ratio.
    2. Resize to (target_size, target_size) using cv2.INTER_AREA.
    3. Normalize pixel intensities to [0, 1].
    4. Apply training augmentation if is_train=True.
    5. Replicate to 3 channels.
    6. Apply ImageNet mean and std normalization.

    Args:
        img: 2D uint8 numpy array.
        is_train: If True, applies training augmentations.
        rng: Optional NumPy random generator for deterministic augmentation.
        target_size: Target square image dimension (default: 320).

    Returns:
        torch.Tensor: Tensor of shape (3, target_size, target_size), dtype float32.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={getattr(img, 'shape', None)}")

    # 1. Pad to square
    padded_img, _ = pad_to_square(img)

    # 2. Resize to 320x320
    res_img = cv2.resize(padded_img, (target_size, target_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # 3. Training augmentation
    if is_train:
        generator = rng if rng is not None else np.random.default_rng()
        res_img = apply_classification_augmentation(res_img, generator)

    # 4. Replicate to 3 channels: (320, 320, 3)
    rgb_img = np.repeat(res_img[:, :, np.newaxis], 3, axis=2)

    # 5. ImageNet normalization: (x - mean) / std
    norm_img = (rgb_img - IMAGENET_MEAN) / IMAGENET_STD

    # 6. Transpose to (3, 320, 320)
    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
    return tensor


class TBClassificationDataset(Dataset):
    """Dataset for training and evaluating TB classifier across all 800 images."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        is_train: bool = False,
        in_memory_images: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            manifest_df: DataFrame with 'stem' and 'label' columns.
            is_train: If True, enables stochastic training augmentations.
            in_memory_images: Optional dictionary mapping stem to uint8 2D numpy array.
            seed: Seed for augmentations.
        """
        self.df = manifest_df.reset_index(drop=True)
        self.is_train = is_train
        self.in_memory_images = in_memory_images or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        stem = str(row["stem"])
        label_val = float(row["label"])

        if stem in self.in_memory_images:
            img = self.in_memory_images[stem]
        else:
            img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
            img = load_image(img_path)

        tensor = preprocess_for_classifier(img, is_train=self.is_train, rng=self.rng, target_size=320)
        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        return tensor, label_tensor, stem


def predict_proba(*args: Any, **kwargs: Any) -> Any:
    """Compute predicted TB probabilities for image(s) using classifier model.

    Delegates to medcomp.classifier.predict_proba.
    """
    from .classifier import predict_proba as _predict_proba

    return _predict_proba(*args, **kwargs)
