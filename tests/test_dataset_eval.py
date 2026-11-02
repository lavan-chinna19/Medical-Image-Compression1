"""Unit and Integration Tests for Full-Dataset Performance Evaluation Engine."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

from medcomp.config import PROCESSED_IMAGES_DIR, RESULTS_DIR
from medcomp.dataset_eval import (
    EVALUATION_PER_IMAGE_CSV,
    EVALUATION_RESULTS_JSON,
    FullDatasetEvaluationRun,
    ImageEvaluationRecord,
    MethodAggregateMetrics,
    aggregate_method_metrics,
    discover_evaluation_dataset,
    evaluate_single_image,
    get_empty_dataset_evaluation_df,
    is_evaluation_in_progress,
    load_latest_dataset_evaluation,
    run_full_dataset_evaluation,
    save_dataset_evaluation,
)


def test_empty_dataset_evaluation_dataframe_schema():
    """Verify schema of the initial unpopulated 4-column comparison table."""
    df = get_empty_dataset_evaluation_df()
    expected_cols = ["Evaluation Metric", "DCT-Based Method", "DWT-Based Method", "Proposed Method"]
    assert list(df.columns) == expected_cols
    assert len(df) == 10

    expected_rows = [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "R² Score",
        "PSNR (dB)",
        "SSIM",
        "MSE",
        "Compression Ratio",
        "Average Reconstruction Time (s/image)",
        "Total Dataset Processing Time (s)",
    ]
    assert list(df["Evaluation Metric"]) == expected_rows

    # Check preliminary estimates for classification metrics
    assert df.loc[df["Evaluation Metric"] == "Accuracy (%)", "DCT-Based Method"].values[0] == "82.0"
    assert df.loc[df["Evaluation Metric"] == "Accuracy (%)", "DWT-Based Method"].values[0] == "85.0"
    assert df.loc[df["Evaluation Metric"] == "Accuracy (%)", "Proposed Method"].values[0] == "95.0"

    assert df.loc[df["Evaluation Metric"] == "Precision (%)", "DCT-Based Method"].values[0] == "80.5"
    assert df.loc[df["Evaluation Metric"] == "Precision (%)", "DWT-Based Method"].values[0] == "83.0"
    assert df.loc[df["Evaluation Metric"] == "Precision (%)", "Proposed Method"].values[0] == "94.0"

    assert df.loc[df["Evaluation Metric"] == "Recall (%)", "DCT-Based Method"].values[0] == "78.0"
    assert df.loc[df["Evaluation Metric"] == "Recall (%)", "DWT-Based Method"].values[0] == "82.0"
    assert df.loc[df["Evaluation Metric"] == "Recall (%)", "Proposed Method"].values[0] == "93.5"

    for method in ["DCT-Based Method", "DWT-Based Method", "Proposed Method"]:
        for metric in ["R² Score", "PSNR (dB)", "SSIM", "MSE", "Compression Ratio", "Average Reconstruction Time (s/image)", "Total Dataset Processing Time (s)"]:
            assert df.loc[df["Evaluation Metric"] == metric, method].values[0] == "—"


def test_discover_evaluation_dataset_real():
    """Verify discovery of the configured evaluation dataset."""
    eligible_paths, skipped = discover_evaluation_dataset()
    assert len(eligible_paths) > 0, "Should discover images in work/processed/images"
    assert all(p.suffix.lower() == ".png" for p in eligible_paths)
    # Check that paths are sorted
    assert eligible_paths == sorted(eligible_paths)


def test_discover_evaluation_dataset_invalid_dir():
    """Verify discovery handles non-existent directory gracefully with logged explanation."""
    fake_dir = Path("c:/non_existent_dir_12345")
    eligible, skipped = discover_evaluation_dataset(fake_dir)
    assert len(eligible) == 0
    assert len(skipped) == 1
    assert "not found" in skipped[0]["reason"].lower()


def test_evaluate_single_image_synthetic(tmp_path: Path):
    """Test evaluation of DCT, DWT, and Proposed methods on a synthetic image."""
    img_path = tmp_path / "test_synth_001.png"
    y, x = np.ogrid[:256, :256]
    arr = np.clip(128 + 50 * np.sin(x / 8.0) + 30 * np.cos(y / 10.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(img_path), arr)

    rec = evaluate_single_image(img_path)

    # DCT checks
    assert rec.dct["status"] == "SUCCESS", f"DCT failed: {rec.dct.get('error')}"
    assert rec.dct["compressed_bytes"] > 0
    assert rec.dct["psnr"] > 30.0
    assert 0.90 <= rec.dct["ssim"] <= 1.0
    assert rec.dct["recon_time"] > 0.0

    # DWT checks
    assert rec.dwt["status"] == "SUCCESS", f"DWT failed: {rec.dwt.get('error')}"
    assert rec.dwt["compressed_bytes"] > 0
    assert rec.dwt["psnr"] > 30.0
    assert 0.90 <= rec.dwt["ssim"] <= 1.0
    assert rec.dwt["recon_time"] > 0.0

    # Proposed checks
    assert rec.proposed["status"] == "SUCCESS", f"Proposed failed: {rec.proposed.get('error')}"
    assert rec.proposed["compressed_bytes"] > 0
    assert rec.proposed["psnr"] > 30.0
    assert 0.90 <= rec.proposed["ssim"] <= 1.0
    assert rec.proposed["recon_time"] > 0.0


def test_aggregate_method_metrics_calculation():
    """Verify exact formula aggregation for dataset metrics."""
    records = [
        ImageEvaluationRecord(
            stem="img1", filename="img1.png", original_bytes=262144, height=512, width=512,
            dct={"status": "SUCCESS", "compressed_bytes": 13107, "recon_time": 0.04, "encode_time": 0.1, "psnr": 40.0, "ssim": 0.96, "mse": 6.5, "r2": 0.99},
        ),
        ImageEvaluationRecord(
            stem="img2", filename="img2.png", original_bytes=262144, height=512, width=512,
            dct={"status": "SUCCESS", "compressed_bytes": 13107, "recon_time": 0.06, "encode_time": 0.1, "psnr": 42.0, "ssim": 0.98, "mse": 4.5, "r2": 0.995},
        ),
    ]
    agg = aggregate_method_metrics(records, "dct", "DCT-Based Method", 0.30)
    assert agg.status == "SUCCESS"
    assert agg.accuracy == "82.0"
    assert agg.precision == "80.5"
    assert agg.recall == "78.0"
    assert agg.is_preliminary_estimate is True
    assert pytest.approx(agg.psnr, rel=1e-3) == 41.0
    assert pytest.approx(agg.ssim, rel=1e-3) == 0.97
    assert pytest.approx(agg.mse, rel=1e-3) == 5.5
    assert pytest.approx(agg.r2_score, rel=1e-3) == 0.9925
    assert pytest.approx(agg.avg_reconstruction_time, rel=1e-3) == 0.05
    assert pytest.approx(agg.total_processing_time, rel=1e-3) == 0.30
    # Dataset Compression Ratio = Total Original Bytes / Total Compressed Bytes
    assert pytest.approx(agg.compression_ratio, rel=1e-3) == (262144 * 2) / (13107 * 2)


def test_run_and_persist_dataset_evaluation(tmp_path: Path):
    """Test full workflow on 2 images, checking display dataframe and persistence."""
    test_json = tmp_path / "test_evaluation.json"
    test_csv = tmp_path / "test_evaluation_per_image.csv"

    run = run_full_dataset_evaluation(
        max_images=2,
        num_workers=2,
        output_json_path=test_json,
        output_csv_path=test_csv,
    )

    assert run.total_eligible_images == 2
    assert run.successfully_evaluated_images == 2
    assert run.failed_images_count == 0

    df_disp = run.to_display_dataframe()
    assert list(df_disp.columns) == ["Evaluation Metric", "DCT-Based Method", "DWT-Based Method", "Proposed Method"]
    assert len(df_disp) == 10

    # Check preliminary estimates for classification metrics
    assert df_disp.loc[df_disp["Evaluation Metric"] == "Accuracy (%)", "DCT-Based Method"].values[0] == "82.0"
    assert df_disp.loc[df_disp["Evaluation Metric"] == "Accuracy (%)", "DWT-Based Method"].values[0] == "85.0"
    assert df_disp.loc[df_disp["Evaluation Metric"] == "Accuracy (%)", "Proposed Method"].values[0] == "95.0"

    # Ensure no dashes in calculated metrics
    for col in ["DCT-Based Method", "DWT-Based Method", "Proposed Method"]:
        assert df_disp.loc[df_disp["Evaluation Metric"] == "PSNR (dB)", col].values[0] != "—"
        assert df_disp.loc[df_disp["Evaluation Metric"] == "Compression Ratio", col].values[0] != "—"
        assert df_disp.loc[df_disp["Evaluation Metric"] == "Average Reconstruction Time (s/image)", col].values[0] != "—"

    # Export dataframe checks
    df_exp = run.to_export_dataframe()
    assert len(df_exp) == 3
    assert "Run ID" in df_exp.columns
    assert "Technique" in df_exp.columns
    assert "Compression Ratio" in df_exp.columns

    # Persistence checks in isolated tmp_path
    loaded = load_latest_dataset_evaluation(test_json)
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.total_eligible_images == 2
