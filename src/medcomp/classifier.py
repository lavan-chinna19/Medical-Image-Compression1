"""TB binary classification model using torchvision ResNet18 and inference routines.

Architecture:
- torchvision ResNet18 backbone pretrained on ImageNet (ResNet18_Weights.DEFAULT).
- Final fc layer replaced with:
    nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(512, 1),
    )
- Output: 1 unnormalized logit (for BCEWithLogitsLoss).
- Mixed precision inference and sigmoid activation for probability output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision.models import ResNet18_Weights, resnet18

from .cls_data import preprocess_for_classifier
from .config import WORK_DIR

CLASSIFIER_DIR = WORK_DIR / "classifier"


def get_classifier(pretrained: bool = True) -> nn.Module:
    """Build ResNet18 binary classifier for TB detection.

    Args:
        pretrained: Whether to initialize with ImageNet weights.

    Returns:
        nn.Module: ResNet18 with dropout (p=0.2) and a 1-logit linear head.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1),
    )
    return model


def load_model(
    fold: Union[int, str, Path],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load trained classifier checkpoint for a given fold.

    Args:
        fold: Fold index (0-4), identifier string (e.g. 'cross_shenzhen'), or direct checkpoint Path.
        device: Torch device (defaults to CUDA if available, else CPU).

    Returns:
        nn.Module: Model loaded with checkpoint weights in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(fold, (int, np.integer)):
        ckpt_path = CLASSIFIER_DIR / f"fold{fold}.pt"
    elif isinstance(fold, str) and (CLASSIFIER_DIR / f"{fold}.pt").is_file():
        ckpt_path = CLASSIFIER_DIR / f"{fold}.pt"
    elif isinstance(fold, str) and fold.isdigit():
        ckpt_path = CLASSIFIER_DIR / f"fold{fold}.pt"
    else:
        ckpt_path = Path(fold)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")

    model = get_classifier(pretrained=False)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    images: Union[np.ndarray, Sequence[np.ndarray], torch.Tensor],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute predicted TB probabilities for one or multiple images.

    Accepts:
    - Single 2D uint8 numpy array (H, W).
    - Sequence/list of 2D uint8 numpy arrays.
    - 3D numpy array (N, H, W).
    - Preprocessed torch.Tensor of shape (3, H, W) or (N, 3, H, W).

    Args:
        model: Trained classifier model.
        images: Image input(s).
        device: Torch device (defaults to model device or CUDA if available).
        batch_size: Inference batch size.

    Returns:
        np.ndarray: 1D array of float32 probabilities in [0, 1].
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    is_cuda = device.type == "cuda"

    # Preprocess inputs into a torch.Tensor
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

    # Numpy array or list of numpy arrays
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
        tensors = [preprocess_for_classifier(img, is_train=False) for img in chunk]
        batch = torch.stack(tensors, dim=0).to(device)
        with autocast("cuda", enabled=is_cuda):
            logits = model(batch)
        batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().float().numpy()
        probs_list.append(np.atleast_1d(batch_probs))

    return np.concatenate(probs_list, axis=0)


def compute_binary_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_prob: Union[Sequence[float], np.ndarray],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard binary classification metrics at a given decision threshold.

    Metrics:
    - Accuracy: (TP + TN) / total
    - Precision (PPV): TP / (TP + FP)
    - Recall (Sensitivity, TPR): TP / (TP + FN)
    - Specificity (TNR): TN / (TN + FP)
    - F1 score: 2 * Precision * Recall / (Precision + Recall)
    - ROC AUC: Area under the ROC curve
    - Confusion matrix: [[TN, FP], [FN, TP]]

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        threshold: Decision threshold for positive prediction (default: 0.5).

    Returns:
        Dict containing scalar metrics and confusion matrix elements.
    """
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_prob ({len(yp)})")

    y_pred = (yp >= threshold).astype(np.int64)

    tn = int(np.sum((yt == 0) & (y_pred == 0)))
    fp = int(np.sum((yt == 0) & (y_pred == 1)))
    fn = int(np.sum((yt == 1) & (y_pred == 0)))
    tp = int(np.sum((yt == 1) & (y_pred == 1)))
    total = len(yt)

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    unique_classes = np.unique(yt)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(yt, yp))
    else:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
    }


def bootstrap_ci_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_prob: Union[Sequence[float], np.ndarray],
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute point estimates and percentile bootstrap confidence intervals.

    Performs non-parametric sampling with replacement over sample indices.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold for classification.
        n_bootstraps: Number of bootstrap iterations (default: 1000).
        confidence_level: CI level (default: 0.95 for 95% CI).
        seed: Random seed for deterministic bootstrapping.

    Returns:
        Dict mapping metric name -> {'point': float, 'ci_lower': float, 'ci_upper': float}.
    """
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    n_samples = len(yt)

    if n_samples == 0:
        raise ValueError("Cannot bootstrap on empty inputs")

    point_metrics = compute_binary_metrics(yt, yp, threshold=threshold)
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]

    rng = np.random.default_rng(seed)
    bootstrap_vals: dict[str, list[float]] = {k: [] for k in metric_keys}

    # Pre-generate bootstrap resample indices
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = yt[idx]
        yp_sample = yp[idx]
        y_pred_sample = (yp_sample >= threshold).astype(np.int64)

        # Accuracy
        acc = float(np.mean(yt_sample == y_pred_sample))
        bootstrap_vals["accuracy"].append(acc)

        # TP, FP, TN, FN
        tn_s = np.sum((yt_sample == 0) & (y_pred_sample == 0))
        fp_s = np.sum((yt_sample == 0) & (y_pred_sample == 1))
        fn_s = np.sum((yt_sample == 1) & (y_pred_sample == 0))
        tp_s = np.sum((yt_sample == 1) & (y_pred_sample == 1))

        prec = float(tp_s / (tp_s + fp_s)) if (tp_s + fp_s) > 0 else 0.0
        rec = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
        spec = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
        f1_s = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        bootstrap_vals["precision"].append(prec)
        bootstrap_vals["recall"].append(rec)
        bootstrap_vals["specificity"].append(spec)
        bootstrap_vals["f1"].append(f1_s)

        # AUC: requires both classes present
        if len(np.unique(yt_sample)) > 1:
            try:
                auc_s = float(roc_auc_score(yt_sample, yp_sample))
                bootstrap_vals["auc"].append(auc_s)
            except ValueError:
                pass

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    results: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        pt = float(point_metrics[k])
        vals = np.array(bootstrap_vals[k], dtype=np.float64)
        if len(vals) > 0 and not np.isnan(pt):
            ci_low = float(np.percentile(vals, lower_pct))
            ci_high = float(np.percentile(vals, upper_pct))
        else:
            ci_low = float("nan")
            ci_high = float("nan")
        results[k] = {
            "point": pt,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        }

    return results
