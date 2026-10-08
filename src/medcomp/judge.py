"""Independent Judge classifier using torchvision EfficientNet-B0.

Trained independently from the steering classifier (ResNet18) to provide
an unbiased, external judge of clinical degradation under compression.

Architecture:
- torchvision EfficientNet-B0 pretrained on ImageNet (EfficientNet_B0_Weights.DEFAULT).
- Final classifier head replaced with:
    nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 1),
    )
- Output: 1 unnormalized logit (for BCEWithLogitsLoss).
- Mixed precision inference and sigmoid activation for probability output.
- Input resolution: 320x320.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .cls_data import preprocess_for_classifier
from .config import WORK_DIR

JUDGE_DIR = WORK_DIR / "judge"


def get_judge(pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 binary classifier for TB detection as an independent judge.

    Args:
        pretrained: Whether to initialize with ImageNet weights.

    Returns:
        nn.Module: EfficientNet-B0 with dropout (p=0.2) and a 1-logit linear head.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1),
    )
    return model


def load_judge_model(
    fold: Union[int, str, Path],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load trained judge checkpoint for a given fold.

    Args:
        fold: Fold index (0-4), identifier string, or direct checkpoint Path.
        device: Torch device (defaults to CUDA if available; raises error if CUDA required).

    Returns:
        nn.Module: Judge model loaded with checkpoint weights in eval mode.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! CUDA is mandatory for RTX 2050.")
        device = torch.device("cuda")

    if isinstance(fold, (int, np.integer)):
        ckpt_path = JUDGE_DIR / f"fold{fold}.pt"
    elif isinstance(fold, str) and (JUDGE_DIR / f"{fold}.pt").is_file():
        ckpt_path = JUDGE_DIR / f"{fold}.pt"
    elif isinstance(fold, str) and fold.isdigit():
        ckpt_path = JUDGE_DIR / f"fold{fold}.pt"
    else:
        ckpt_path = Path(fold)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Judge checkpoint not found: {ckpt_path}")

    model = get_judge(pretrained=False)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_proba_judge(
    model: nn.Module,
    images: Union[np.ndarray, Sequence[np.ndarray], torch.Tensor],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute predicted TB probabilities using the judge model.

    Accepts:
    - Single 2D uint8 numpy array (H, W).
    - Sequence/list of 2D uint8 numpy arrays.
    - 3D numpy array (N, H, W).
    - Preprocessed torch.Tensor of shape (3, H, W) or (N, 3, H, W).

    Args:
        model: Trained judge model.
        images: Image input(s).
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        np.ndarray: 1D array of float32 probabilities in [0, 1].
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available!")
            device = torch.device("cuda")

    model.eval()
    is_cuda = device.type == "cuda"

    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            tensor_batch = images.unsqueeze(0).float()
        elif images.ndim == 4:
            tensor_batch = images.float()
        else:
            raise ValueError(f"Expected 3D or 4D Tensor, got shape {images.shape}")

        n_samples = tensor_batch.shape[0]
        probs_list: list[np.ndarray] = []
        for i in range(0, n_samples, batch_size):
            batch = tensor_batch[i : i + batch_size].to(device)
            with autocast("cuda", enabled=is_cuda):
                logits = model(batch)
            batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().float().numpy()
            probs_list.append(np.atleast_1d(batch_probs))
        return np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=np.float32)

    if isinstance(images, np.ndarray) and images.ndim == 2:
        img_list = [images]
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        img_list = [images[i] for i in range(images.shape[0])]
    elif isinstance(images, (list, tuple)):
        img_list = list(images)
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")

    if len(img_list) == 0:
        return np.array([], dtype=np.float32)

    probs_list = []
    for i in range(0, len(img_list), batch_size):
        chunk = img_list[i : i + batch_size]
        tensors = [preprocess_for_classifier(img, is_train=False, target_size=320) for img in chunk]
        batch = torch.stack(tensors, dim=0).to(device)
        with autocast("cuda", enabled=is_cuda):
            logits = model(batch)
        batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().float().numpy()
        probs_list.append(np.atleast_1d(batch_probs))

    return np.concatenate(probs_list, axis=0)
