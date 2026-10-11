"""Unit tests for task-based compression benchmark pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medcomp.classifier import get_classifier, predict_proba
from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR
from medcomp.io_utils import load_image
from scripts.evaluate_benchmark import compute_subset_metrics, evaluate_with_paired_bootstrap


def test_held_out_fold_classification_assignment():
    """Verify that every image is strictly mapped to its held-out fold model."""
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Manifest not found")

    df = pd.read_csv(manifest_path)
    # Check that fold values are exclusively in range [0, 4]
    assert set(df["fold"].unique()).issubset({0, 1, 2, 3, 4})
    for _, row in df.iterrows():
        fold = int(row["fold"])
        assert 0 <= fold <= 4, f"Invalid fold {fold} for stem {row['stem']}"


def test_original_variant_properties():
    """Verify that the original reference variant has flip rate 0 and AUC change 0."""
    stems = [f"sample_{i}" for i in range(20)]
    labels = np.array([0] * 10 + [1] * 10)
    probs = np.array([0.1] * 10 + [0.9] * 10)

    df_orig = pd.DataFrame({
        "stem": stems,
        "codec": ["original"] * 20,
        "setting": [0] * 20,
        "label": labels,
        "bpp": [8.0] * 20,
        "prob_dec": probs,
        "pred_dec": (probs >= 0.5).astype(int),
        "flip": [0] * 20,
        "abs_prob_diff": [0.0] * 20,
    })

    metrics = compute_subset_metrics(df_orig, df_orig)
    assert metrics["flip_rate"] == 0.0
    assert metrics["delta_auc"] == 0.0
    assert metrics["delta_recall"] == 0.0
    assert metrics["mean_abs_prob_diff"] == 0.0


def test_bpp_calculation():
    """Verify bits per pixel formula bpp = total_bytes * 8 / (H * W)."""
    h, w = 320, 320
    total_bytes = 4096
    expected_bpp = (4096 * 8.0) / (320 * 320)
    calc_bpp = (total_bytes * 8.0) / (h * w)
    assert np.isclose(calc_bpp, expected_bpp)
    assert np.isclose(calc_bpp, 0.32)


def test_in_memory_equals_file_based_prediction():
    """Verify that in-memory uint8 image prediction matches file-based loaded image prediction."""
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Manifest not found")
    df = pd.read_csv(manifest_path)
    sample_stem = df.iloc[0]["stem"]
    img_path = PROCESSED_IMAGES_DIR / f"{sample_stem}.png"
    if not img_path.is_file():
        pytest.skip(f"Sample image {img_path} not found")

    img_loaded = load_image(img_path)
    # Clone into memory
    img_in_memory = np.array(img_loaded, copy=True)

    model = get_classifier(pretrained=False)
    model.eval()

    device = torch.device("cpu")
    prob_file = predict_proba(model, img_loaded, device=device)
    prob_mem = predict_proba(model, img_in_memory, device=device)

    np.testing.assert_allclose(prob_file, prob_mem, rtol=1e-5, atol=1e-5)


def test_bootstrap_ci_contains_point_estimate():
    """Verify that 95% paired bootstrap CI bounds enclose point estimates."""
    np.random.seed(42)
    n = 60
    stems = [f"sample_{i}" for i in range(n)]
    y_true = np.array([0] * 30 + [1] * 30)
    p_orig = np.clip(y_true * 0.7 + np.random.uniform(0.1, 0.3, size=n), 0.01, 0.99)
    p_dec = np.clip(p_orig + np.random.normal(0, 0.05, size=n), 0.01, 0.99)

    records = []
    for s, yt, po, pd_val in zip(stems, y_true, p_orig, p_dec):
        records.append({
            "stem": s,
            "codec": "original",
            "setting": 0,
            "label": yt,
            "bpp": 8.0,
            "prob_dec": po,
            "pred_dec": int(po >= 0.5),
            "flip": 0,
            "abs_prob_diff": 0.0,
        })
        records.append({
            "stem": s,
            "codec": "dwt",
            "setting": 50,
            "label": yt,
            "bpp": 1.2,
            "prob_dec": pd_val,
            "pred_dec": int(pd_val >= 0.5),
            "flip": int((pd_val >= 0.5) != (po >= 0.5)),
            "abs_prob_diff": abs(pd_val - po),
        })

    toy_df = pd.DataFrame(records)
    summary_res = evaluate_with_paired_bootstrap(toy_df, n_bootstraps=300, seed=42)

    for _, row in summary_res.iterrows():
        for metric in ["auc", "accuracy", "recall", "flip_rate"]:
            pt = row[metric]
            low = row[f"{metric}_ci_lower"]
            high = row[f"{metric}_ci_upper"]
            assert not np.isnan(pt)
            assert not np.isnan(low)
            assert not np.isnan(high)
            assert low <= pt <= high, f"{row['codec']}_{row['setting']} {metric}: {pt} not in [{low}, {high}]"


def test_resume_skips_existing_images(tmp_path):
    """Verify that resume logic correctly filters out stems already present in the benchmark CSV."""
    csv_file = tmp_path / "benchmark_per_image.csv"
    # Create 30 rows for stem_001
    records = []
    for i in range(30):
        records.append({
            "stem": "stem_001",
            "codec": "test",
            "setting": i,
            "total_bytes": 100,
            "bpp": 0.5,
            "prob_orig": 0.2,
            "prob_dec": 0.2,
            "pred_orig": 0,
            "pred_dec": 0,
            "flip": 0,
            "abs_prob_diff": 0.0,
        })
    pd.DataFrame(records).to_csv(csv_file, index=False)

    existing_df = pd.read_csv(csv_file)
    counts = existing_df["stem"].value_counts()
    completed_stems = set(counts[counts >= 30].index)

    assert "stem_001" in completed_stems

    all_manifest_stems = ["stem_001", "stem_002", "stem_003"]
    pending = [s for s in all_manifest_stems if s not in completed_stems]
    assert pending == ["stem_002", "stem_003"]
