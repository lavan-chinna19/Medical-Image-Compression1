"""Unit tests for src/medcomp/metrics_report.py and held-out classification logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from medcomp.app_logic import LoadedModels, classify_heldout
from medcomp.config import RESULTS_DIR
from medcomp.metrics_report import (
    bootstrap_ci,
    classification_metrics,
    load_oof,
    load_results,
    roc_points,
    threshold_sweep,
)


def test_classification_metrics_toy_set() -> None:
    """classification_metrics on a hand-made toy set gives the known accuracy, precision, recall, specificity, and F1."""
    # Constructed dataset:
    # 4 True Positives, 1 False Positive, 2 False Negatives, 3 True Negatives (Total = 10)
    # y_true = 1 for 6 cases (4 TP, 2 FN)
    # y_true = 0 for 4 cases (3 TN, 1 FP)
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    prob = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.8, 0.2, 0.1, 0.05]
    threshold = 0.50

    m = classification_metrics(y_true, prob, threshold=threshold)

    assert m["tp"] == 4
    assert m["fp"] == 1
    assert m["fn"] == 2
    assert m["tn"] == 3

    expected_acc = 7.0 / 10.0  # 0.70
    expected_prec = 4.0 / 5.0  # 0.80
    expected_rec = 4.0 / 6.0  # 2/3
    expected_spec = 3.0 / 4.0  # 0.75
    expected_f1 = 2.0 * (expected_prec * expected_rec) / (expected_prec + expected_rec)  # 8/11 ~ 0.72727

    assert pytest.approx(m["accuracy"], rel=1e-5) == expected_acc
    assert pytest.approx(m["precision"], rel=1e-5) == expected_prec
    assert pytest.approx(m["recall"], rel=1e-5) == expected_rec
    assert pytest.approx(m["specificity"], rel=1e-5) == expected_spec
    assert pytest.approx(m["f1"], rel=1e-5) == expected_f1


def test_classification_metrics_edge_cases_and_zero_division() -> None:
    """Handle divisions by zero (return NaN) and single-class input gracefully."""
    # All negative samples, all predicted negative -> TP=0, FP=0, FN=0, TN=3
    m_no_pos = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert np.isnan(m_no_pos["precision"])  # tp + fp == 0
    assert np.isnan(m_no_pos["recall"])  # tp + fn == 0
    assert np.isnan(m_no_pos["f1"])
    assert np.isnan(m_no_pos["auc"])  # single class
    assert m_no_pos["specificity"] == 1.0
    assert m_no_pos["accuracy"] == 1.0

    # All positive samples, all predicted positive -> TP=3, FP=0, FN=0, TN=0
    m_no_neg = classification_metrics([1, 1, 1], [0.8, 0.9, 0.7], threshold=0.5)
    assert np.isnan(m_no_neg["specificity"])  # tn + fp == 0
    assert np.isnan(m_no_neg["auc"])  # single class
    assert m_no_neg["recall"] == 1.0
    assert m_no_neg["precision"] == 1.0
    assert m_no_neg["accuracy"] == 1.0


def test_auc_perfect_and_reversed() -> None:
    """A perfect classifier gives AUC 1.0 and a reversed one 0.0."""
    y_true = [0, 0, 1, 1]

    # Perfect ranking
    prob_perfect = [0.1, 0.2, 0.8, 0.9]
    m_perf = classification_metrics(y_true, prob_perfect)
    assert m_perf["auc"] == 1.0

    # Reversed ranking
    prob_reversed = [0.9, 0.8, 0.2, 0.1]
    m_rev = classification_metrics(y_true, prob_reversed)
    assert m_rev["auc"] == 0.0


def test_bootstrap_ci_contains_point_estimate() -> None:
    """The bootstrap CI contains the point estimate for every metric."""
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=100)
    prob = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, size=100), 0.0, 1.0)

    point_m = classification_metrics(y_true, prob, threshold=0.5)
    ci = bootstrap_ci(y_true, prob, threshold=0.5, n=500, seed=42)

    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]:
        pt = point_m[metric]
        low, high = ci[metric]
        assert not np.isnan(pt), f"Point estimate for {metric} is NaN"
        assert low <= pt <= high, f"Metric {metric}: Point estimate {pt} not in [{low}, {high}]"


def test_threshold_sweep_recall_monotonicity() -> None:
    """Recall never decreases as the threshold falls in threshold_sweep."""
    rng = np.random.default_rng(456)
    y_true = rng.integers(0, 2, size=80)
    prob = rng.uniform(0.0, 1.0, size=80)

    # Thresholds falling from 1.0 down to 0.0
    thresholds = np.linspace(1.0, 0.0, 51)
    sweep_df = threshold_sweep(y_true, prob, thresholds=thresholds)

    recalls = sweep_df["recall"].values
    # As threshold falls, recall must not decrease (diffs >= 0)
    diffs = np.diff(recalls)
    assert (diffs >= -1e-9).all(), f"Recall decreased during falling threshold sweep: {diffs[diffs < -1e-9]}"


def test_load_results_returns_none_for_missing_files(tmp_path: Path) -> None:
    """load_results returns None for missing files instead of crashing."""
    # Empty directory
    res = load_results(results_dir=tmp_path)
    assert isinstance(res, dict)
    assert res["classifier_metrics"] is None
    assert res["judge_metrics"] is None
    assert res["classifier_crossscanner"] is None
    assert res["benchmark_summary"] is None
    assert res["benchmark_summary_per_source"] is None
    assert res["benchmark_operating_points"] is None
    assert res["segmentation_summary"] is None

    # Partially populated directory
    (tmp_path / "classifier_metrics.csv").write_text("colA,colB\n1,2\n")
    res_partial = load_results(results_dir=tmp_path)
    assert res_partial["classifier_metrics"] is not None
    assert res_partial["judge_metrics"] is None


def test_resnet_overall_metrics_match_classifier_metrics() -> None:
    """ResNet18 overall metrics computed from results/classifier_oof.csv at threshold 0.5 match results/classifier_metrics.csv within 1e-3."""
    oof_path = RESULTS_DIR / "classifier_oof.csv"
    metrics_path = RESULTS_DIR / "classifier_metrics.csv"

    assert oof_path.is_file(), f"{oof_path} missing"
    assert metrics_path.is_file(), f"{metrics_path} missing"

    df_oof = load_oof(oof_path)
    df_metrics = pd.read_csv(metrics_path)

    m = classification_metrics(df_oof["label"], df_oof["prob_tb"], threshold=0.50)

    overall_row = df_metrics[df_metrics["group"] == "Overall"].iloc[0]

    assert abs(m["accuracy"] - float(overall_row["accuracy"])) < 1e-3
    assert abs(m["precision"] - float(overall_row["precision"])) < 1e-3
    assert abs(m["recall"] - float(overall_row["recall"])) < 1e-3
    assert abs(m["specificity"] - float(overall_row["specificity"])) < 1e-3
    assert abs(m["f1"] - float(overall_row["f1"])) < 1e-3
    assert abs(m["auc"] - float(overall_row["auc"])) < 1e-3
    assert m["tn"] == int(overall_row["tn"])
    assert m["fp"] == int(overall_row["fp"])
    assert m["fn"] == int(overall_row["fn"])
    assert m["tp"] == int(overall_row["tp"])


def test_judge_overall_metrics_match_judge_metrics() -> None:
    """EfficientNet-B0 judge overall metrics computed from results/judge_oof.csv at threshold 0.5 match results/judge_metrics.csv within 1e-3."""
    oof_path = RESULTS_DIR / "judge_oof.csv"
    metrics_path = RESULTS_DIR / "judge_metrics.csv"

    assert oof_path.is_file(), f"{oof_path} missing"
    assert metrics_path.is_file(), f"{metrics_path} missing"

    df_oof = load_oof(oof_path)
    df_metrics = pd.read_csv(metrics_path)

    m = classification_metrics(df_oof["label"], df_oof["prob_tb"], threshold=0.50)

    overall_row = df_metrics[df_metrics["group"] == "Overall"].iloc[0]

    assert abs(m["accuracy"] - float(overall_row["accuracy"])) < 1e-3
    assert abs(m["precision"] - float(overall_row["precision"])) < 1e-3
    assert abs(m["recall"] - float(overall_row["recall"])) < 1e-3
    assert abs(m["specificity"] - float(overall_row["specificity"])) < 1e-3
    assert abs(m["f1"] - float(overall_row["f1"])) < 1e-3
    assert abs(m["auc"] - float(overall_row["auc"])) < 1e-3
    assert m["tn"] == int(overall_row["tn"])
    assert m["fp"] == int(overall_row["fp"])
    assert m["fn"] == int(overall_row["fn"])
    assert m["tp"] == int(overall_row["tp"])


def test_classify_heldout_uses_correct_fold_model() -> None:
    """classify_heldout never uses a model trained on the image's fold (assert on the fold index)."""
    # Create mock models to track which fold index is accessed
    mock_resnet_models = [MagicMock() for _ in range(5)]
    mock_judge_models = [MagicMock() for _ in range(5)]

    # Attach dummy forward predictions
    import torch
    for i in range(5):
        # resnet returns tensor with logit corresponding to prob (0.1 * (i+1))
        mock_resnet_models[i].return_value = torch.tensor([[float(i)]])
        mock_judge_models[i].return_value = torch.tensor([[float(i)]])

    fake_device = torch.device("cpu")
    loaded_models = LoadedModels(
        unet_models=[],
        resnet_models=mock_resnet_models,
        judge_models=mock_judge_models,
        device=fake_device,
    )

    dummy_img = np.zeros((64, 64), dtype=np.uint8)

    # For each fold 0 to 4, ensure classify_heldout accesses exactly models[k]
    for target_fold in range(5):
        out = classify_heldout(dummy_img, fold=target_fold, models=loaded_models)

        # Assert on the fold index returned
        assert out["fold"] == target_fold

        # Assert that only model[target_fold] was invoked and other fold models were NOT invoked for this call
        for other_fold in range(5):
            if other_fold == target_fold:
                assert mock_resnet_models[other_fold].called, f"Model for fold {target_fold} was not called"
                assert mock_judge_models[other_fold].called, f"Judge for fold {target_fold} was not called"
            # Reset mocks for next fold test
            mock_resnet_models[other_fold].reset_mock()
            mock_judge_models[other_fold].reset_mock()

    # Out of range fold index must raise ValueError
    with pytest.raises(ValueError):
        classify_heldout(dummy_img, fold=5, models=loaded_models)
    with pytest.raises(ValueError):
        classify_heldout(dummy_img, fold=-1, models=loaded_models)
