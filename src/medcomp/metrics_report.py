"""Metrics reporting and evaluation utilities for the medcomp pipeline.

Pure logic only: completely independent of Streamlit UI frameworks,
fully testable in standard unit tests without requiring a GPU or UI dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .config import RESULTS_DIR


def load_oof(path: Union[str, Path]) -> pd.DataFrame:
    """Load out-of-fold classification prediction CSV file into a standardized DataFrame.

    Tolerant of column name variations between classifier_oof.csv, judge_oof.csv,
    and alternative evaluation formats.

    Canonical columns produced:
        stem (str), source (str), label (int), fold (int), prob_tb (float), pred (int)

    Args:
        path: Filepath to the OOF CSV file.

    Returns:
        pd.DataFrame with standardized columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns cannot be mapped.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"OOF CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Column name mappings
    col_map: dict[str, str] = {}
    lower_to_orig = {col.lower().strip(): col for col in df.columns}

    # 1. Stem / Identifier
    for candidate in ["stem", "id", "image_id", "filename", "name"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "stem"
            break

    # 2. Source / Dataset
    for candidate in ["source", "dataset", "site", "cohort"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "source"
            break

    # 3. Label / Ground truth
    for candidate in ["label", "y_true", "target", "gt", "ground_truth", "class"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "label"
            break

    # 4. Fold
    for candidate in ["fold", "kfold", "split", "fold_idx"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "fold"
            break

    # 5. Probability
    for candidate in ["prob_tb", "prob", "probability", "y_prob", "pred_prob", "score"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "prob_tb"
            break

    # 6. Binary Prediction
    for candidate in ["pred", "prediction", "y_pred"]:
        if candidate in lower_to_orig:
            col_map[lower_to_orig[candidate]] = "pred"
            break

    renamed = df.rename(columns=col_map)

    required = ["label", "prob_tb"]
    missing = [c for c in required if c not in renamed.columns]
    if missing:
        raise ValueError(f"OOF CSV {csv_path.name} missing essential columns: {missing}")

    if "stem" not in renamed.columns:
        renamed["stem"] = [f"image_{i:04d}" for i in range(len(renamed))]

    if "source" not in renamed.columns:
        renamed["source"] = "Unknown"

    if "fold" not in renamed.columns:
        renamed["fold"] = 0

    if "pred" not in renamed.columns:
        renamed["pred"] = (renamed["prob_tb"] >= 0.5).astype(int)

    renamed["stem"] = renamed["stem"].astype(str)
    renamed["source"] = renamed["source"].astype(str)
    renamed["label"] = renamed["label"].astype(int)
    renamed["fold"] = renamed["fold"].astype(int)
    renamed["prob_tb"] = renamed["prob_tb"].astype(float)
    renamed["pred"] = renamed["pred"].astype(int)

    return renamed[["stem", "source", "label", "fold", "prob_tb", "pred"]].copy()


def classification_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    prob: Union[Sequence[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> dict[str, Union[float, int]]:
    """Compute binary classification metrics and confusion matrix counts.

    Handles edge cases gracefully:
    - Zero divisions return float('nan').
    - Single-class y_true returns float('nan') for AUC and undefined rates.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        prob: Predicted positive class probabilities in [0, 1].
        threshold: Decision threshold for positive prediction (default: 0.5).

    Returns:
        dict with keys: accuracy, precision, recall, specificity, f1, auc, tn, fp, fn, tp.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(prob, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(prob)={len(yp)}")

    total = len(yt)
    if total == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = (yp >= threshold).astype(np.int64)

    tn = int(np.sum((yt == 0) & (y_pred == 0)))
    fp = int(np.sum((yt == 0) & (y_pred == 1)))
    fn = int(np.sum((yt == 1) & (y_pred == 0)))
    tp = int(np.sum((yt == 1) & (y_pred == 1)))

    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0.0:
        f1 = float("nan")
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    # ROC AUC: requires both positive and negative classes
    unique_classes = np.unique(yt)
    if len(unique_classes) > 1:
        try:
            auc = float(roc_auc_score(yt, yp))
        except ValueError:
            auc = float("nan")
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
    }


def roc_points(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    prob: Union[Sequence[float], np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Compute ROC curve operating points (False Positive Rate, True Positive Rate, Threshold).

    Args:
        y_true: Ground truth binary labels.
        prob: Predicted probabilities.

    Returns:
        pd.DataFrame with columns ['fpr', 'tpr', 'threshold'].
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(prob, dtype=np.float64)

    if len(np.unique(yt)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])

    fpr, tpr, thresholds = roc_curve(yt, yp)
    return pd.DataFrame({
        "fpr": fpr.astype(float),
        "tpr": tpr.astype(float),
        "threshold": thresholds.astype(float),
    })


def threshold_sweep(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    prob: Union[Sequence[float], np.ndarray, pd.Series],
    thresholds: Union[Sequence[float], np.ndarray],
) -> pd.DataFrame:
    """Evaluate classification metrics across a sweep of decision thresholds.

    Guarantees: As threshold decreases, recall (sensitivity) never decreases.

    Args:
        y_true: Ground truth binary labels.
        prob: Predicted probabilities.
        thresholds: Iterable of threshold values to test.

    Returns:
        pd.DataFrame with one row per threshold and all classification metrics.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(prob, dtype=np.float64)

    rows = []
    for thresh in thresholds:
        m = classification_metrics(yt, yp, threshold=float(thresh))
        m["threshold"] = float(thresh)
        rows.append(m)

    df = pd.DataFrame(rows)
    # Order columns logically
    cols = ["threshold", "accuracy", "precision", "recall", "specificity", "f1", "auc", "tn", "fp", "fn", "tp"]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols]


def bootstrap_ci(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    prob: Union[Sequence[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
    n: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Compute percentile bootstrap confidence intervals by resampling cases with replacement.

    Args:
        y_true: Ground truth binary labels.
        prob: Predicted positive class probabilities.
        threshold: Decision threshold.
        n: Number of bootstrap iterations (default: 1000).
        seed: Random seed for deterministic reproducibility.
        confidence_level: Desired interval width (default: 0.95 for 95% CI).

    Returns:
        dict mapping metric name to (ci_lower, ci_upper) tuple.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(prob, dtype=np.float64)
    n_samples = len(yt)

    metrics_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]

    if n_samples == 0:
        return {k: (float("nan"), float("nan")) for k in metrics_keys}

    rng = np.random.default_rng(seed)
    bootstrap_vals: dict[str, list[float]] = {k: [] for k in metrics_keys}

    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = yt[idx]
        yp_sample = yp[idx]

        m = classification_metrics(yt_sample, yp_sample, threshold=threshold)
        for k in metrics_keys:
            val = m[k]
            if not np.isnan(val):
                bootstrap_vals[k].append(float(val))

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_dict: dict[str, tuple[float, float]] = {}
    for k in metrics_keys:
        vals = np.array(bootstrap_vals[k], dtype=np.float64)
        if len(vals) > 0:
            low = float(np.percentile(vals, lower_pct))
            high = float(np.percentile(vals, upper_pct))
        else:
            low = float("nan")
            high = float("nan")
        ci_dict[k] = (low, high)

    return ci_dict


def load_results(
    results_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    """Load benchmark and evaluation CSV results files from results directory.

    Returns None for any missing file instead of crashing.

    Files loaded:
        - classifier_metrics: results/classifier_metrics.csv
        - judge_metrics: results/judge_metrics.csv
        - classifier_crossscanner: results/classifier_crossscanner.csv
        - benchmark_summary: results/benchmark_summary.csv
        - benchmark_summary_per_source: results/benchmark_summary_per_source.csv
        - benchmark_operating_points: results/benchmark_operating_points.csv
        - segmentation_summary: results/segmentation_summary.csv

    Args:
        results_dir: Directory containing CSV results files (defaults to RESULTS_DIR).

    Returns:
        dict mapping file key to pd.DataFrame (or None if file is missing).
    """
    res_path = Path(results_dir) if results_dir is not None else RESULTS_DIR

    target_files = {
        "classifier_metrics": "classifier_metrics.csv",
        "judge_metrics": "judge_metrics.csv",
        "classifier_crossscanner": "classifier_crossscanner.csv",
        "benchmark_summary": "benchmark_summary.csv",
        "benchmark_summary_per_source": "benchmark_summary_per_source.csv",
        "benchmark_operating_points": "benchmark_operating_points.csv",
        "segmentation_summary": "segmentation_summary.csv",
    }

    loaded: dict[str, Optional[pd.DataFrame]] = {}
    for key, filename in target_files.items():
        file_path = res_path / filename
        if file_path.is_file():
            try:
                loaded[key] = pd.read_csv(file_path)
            except Exception:
                loaded[key] = None
        else:
            loaded[key] = None

    return loaded
