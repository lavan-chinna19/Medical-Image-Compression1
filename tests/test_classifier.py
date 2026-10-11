"""Unit tests for TB classification dataset, ResNet18 model, and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from medcomp.classifier import (
    bootstrap_ci_metrics,
    compute_binary_metrics,
    get_classifier,
    predict_proba,
)
from medcomp.cls_data import TBClassificationDataset, preprocess_for_classifier
from medcomp.config import DATA_DIR


def test_dataset_output_shape_and_in_memory_image():
    """Verify TBClassificationDataset returns (3, 320, 320) tensors and supports in-memory images."""
    dummy_img = (np.random.rand(400, 500) * 255).astype(np.uint8)
    manifest = pd.DataFrame({
        "stem": ["test_sample_01"],
        "label": [1],
        "source": ["Montgomery"],
        "fold": [0],
    })

    ds = TBClassificationDataset(
        manifest_df=manifest,
        is_train=False,
        in_memory_images={"test_sample_01": dummy_img},
    )

    tensor, label, stem = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 320, 320)
    assert tensor.dtype == torch.float32
    assert label.item() == 1.0
    assert stem == "test_sample_01"


def test_preprocess_for_classifier_in_memory_uint8():
    """Verify preprocess_for_classifier handles raw uint8 2D arrays directly."""
    img = np.zeros((256, 128), dtype=np.uint8)
    img[50:100, 20:80] = 200
    tensor = preprocess_for_classifier(img, is_train=False, target_size=320)
    assert tensor.shape == (3, 320, 320)
    assert tensor.dtype == torch.float32


def test_held_out_fold_leakage_prevention():
    """Verify that for every fold k, held-out images never appear in train or validation sets."""
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Manifest file not found.")

    df = pd.read_csv(manifest_path)
    from sklearn.model_selection import train_test_split

    for k in range(5):
        test_df = df[df["fold"] == k]
        train_pool = df[df["fold"] != k]

        # 10% stratified validation split carved from train_pool
        train_df, val_df = train_test_split(
            train_pool,
            test_size=0.10,
            stratify=train_pool[["source", "label"]],
            random_state=42 + k,
        )

        test_stems = set(test_df["stem"])
        train_stems = set(train_df["stem"])
        val_stems = set(val_df["stem"])

        # Strictly disjoint checks
        assert len(test_stems.intersection(train_stems)) == 0, f"Fold {k}: held-out test stems in train set!"
        assert len(test_stems.intersection(val_stems)) == 0, f"Fold {k}: held-out test stems in val set!"
        assert len(train_stems.intersection(val_stems)) == 0, f"Fold {k}: train and val stems overlap!"


def test_predictions_range_in_0_1():
    """Verify that predict_proba returns probabilities strictly bounded in [0, 1]."""
    model = get_classifier(pretrained=False)
    model.eval()

    # Pass synthetic in-memory images
    imgs = [
        np.full((320, 320), fill_value=50, dtype=np.uint8),
        np.full((400, 300), fill_value=180, dtype=np.uint8),
        np.zeros((100, 100), dtype=np.uint8),
    ]

    probs = predict_proba(model, imgs, device=torch.device("cpu"), batch_size=2)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 3
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_perfect_prediction_toy_case():
    """Verify perfect predictions give AUC 1.0 and expected confusion matrix."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.85, 0.90, 0.95, 0.99], dtype=float)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tn"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 4
    expected_cm = np.array([[4, 0], [0, 4]], dtype=np.int64)
    np.testing.assert_array_equal(metrics["confusion_matrix"], expected_cm)


def test_bootstrap_ci_contains_point_estimate():
    """Verify that the 95% bootstrap CI contains the point estimate for standard predictions."""
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    # Add noise to true labels for realistic probabilistic predictions
    y_prob = np.clip(y_true * 0.6 + np.random.uniform(0.1, 0.3, size=n), 0.01, 0.99)

    ci_results = bootstrap_ci_metrics(y_true, y_prob, threshold=0.5, n_bootstraps=500, seed=42)

    for metric_name, res in ci_results.items():
        point = res["point"]
        low = res["ci_lower"]
        high = res["ci_upper"]
        assert not np.isnan(point), f"{metric_name} point estimate is NaN"
        assert not np.isnan(low), f"{metric_name} CI lower is NaN"
        assert not np.isnan(high), f"{metric_name} CI upper is NaN"
        assert low <= point <= high, (
            f"Metric {metric_name}: point estimate {point:.4f} is outside [{low:.4f}, {high:.4f}]"
        )
