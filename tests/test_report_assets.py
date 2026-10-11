"""Tests for report asset generation, consistency audit, and claims register."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_report_assets.py"
REPORT_ASSETS_DIR = PROJECT_ROOT / "report_assets"
TABLES_DIR = REPORT_ASSETS_DIR / "tables"
FIGURES_DIR = REPORT_ASSETS_DIR / "figures"


@pytest.fixture(scope="module", autouse=True)
def run_generator():
    """Ensure the generator script runs successfully before executing tests."""
    assert SCRIPT_PATH.is_file(), f"Script missing: {SCRIPT_PATH}"
    cmd = [sys.executable, str(SCRIPT_PATH)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, f"make_report_assets.py failed with returncode {result.returncode}:\n{result.stderr}"


def test_core_report_files_exist():
    """Assert that core audit, claims register, and index files exist and are non-empty."""
    audit_file = REPORT_ASSETS_DIR / "consistency_audit.md"
    claims_csv = REPORT_ASSETS_DIR / "claims_register.csv"
    claims_md = REPORT_ASSETS_DIR / "claims_register.md"
    results_index = REPORT_ASSETS_DIR / "results_index.md"

    for p in [audit_file, claims_csv, claims_md, results_index]:
        assert p.is_file(), f"Missing file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


def test_all_expected_tables_exist_and_non_empty():
    """Assert that all required table CSV and Markdown files exist and are non-empty."""
    expected_tables = [
        "dataset_overview",
        "baseline_summary",
        "dct_summary",
        "dwt_summary",
        "dwt_tuning_summary",
        "dwt_ablation_summary",
        "dwt_ablation",
        "lossless_summary",
        "roi_codec_summary",
        "roi_fill_ablation",
        "segmentation_summary",
        "segmentation_margins",
        "roi_codec_predicted_summary",
        "classifier_metrics",
        "classifier_crossscanner",
        "judge_metrics",
        "benchmark_summary",
        "benchmark_summary_per_source",
        "benchmark_operating_points",
        "region_ablation_summary",
        "region_ablation_roi_comparison",
        "graded_pilot_summary",
        "graded_pilot_guarantee_cost",
        "controller_summary",
        "controller_vs_fixed",
        "controller_variants_vs_fixed",
        "controller_variants_summary",
    ]

    for stem in expected_tables:
        csv_file = TABLES_DIR / f"{stem}.csv"
        md_file = TABLES_DIR / f"{stem}.md"
        assert csv_file.is_file(), f"Missing table CSV: {csv_file}"
        assert csv_file.stat().st_size > 0, f"Table CSV is empty: {csv_file}"
        assert md_file.is_file(), f"Missing table Markdown: {md_file}"
        assert md_file.stat().st_size > 0, f"Table Markdown is empty: {md_file}"


def test_all_expected_figures_exist_and_non_empty():
    """Assert that all publication figures exist as both 300 dpi PNG and vector PDF."""
    expected_figures = [
        "fig_rd_psnr",
        "fig_rd_ssim",
        "fig_rd_roi_ssim",
        "fig_benchmark_auc_vs_bpp",
        "fig_benchmark_flips_vs_bpp",
        "fig_segmentation_margins_dice_recall",
        "fig_region_ablation_quality",
        "fig_graded_pilot_tradeoff",
        "fig_controller_vs_fixed",
        "fig_montgomery_ssim_dip",
    ]

    for stem in expected_figures:
        png_file = FIGURES_DIR / f"{stem}.png"
        pdf_file = FIGURES_DIR / f"{stem}.pdf"
        assert png_file.is_file(), f"Missing figure PNG: {png_file}"
        assert png_file.stat().st_size > 0, f"Figure PNG is empty: {png_file}"
        assert pdf_file.is_file(), f"Missing figure PDF: {pdf_file}"
        assert pdf_file.stat().st_size > 0, f"Figure PDF is empty: {pdf_file}"


def test_claims_register_completeness_and_no_nan():
    """Assert that claims register covers C1-C17 and required columns contain no NaN."""
    claims_csv = REPORT_ASSETS_DIR / "claims_register.csv"
    assert claims_csv.is_file()

    df = pd.read_csv(claims_csv)
    required_cols = ["id", "claim", "evidence_source", "numbers", "sample_size", "verdict", "caveat"]

    for col in required_cols:
        assert col in df.columns, f"Missing required column in claims register: {col}"
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"Column '{col}' in claims_register.csv contains {nan_count} NaN values"

    assert len(df) == 17, f"Expected 17 claims, found {len(df)}"
    expected_ids = [f"C{i}" for i in range(1, 18)]
    assert list(df["id"]) == expected_ids, f"Claims IDs mismatch: {list(df['id'])}"

    valid_verdicts = {"SUPPORTED", "PARTLY SUPPORTED", "NOT SUPPORTED", "NOT TESTED"}
    for idx, v in enumerate(df["verdict"]):
        assert v in valid_verdicts, f"Invalid verdict for claim {df.iloc[idx]['id']}: {v}"


def test_required_columns_in_tables_no_nan():
    """Assert that required primary columns in generated tables contain no NaN."""
    table_requirements = {
        "dataset_overview.csv": ["source", "total_images", "normal_cases", "tb_cases"],
        "baseline_summary.csv": ["source", "codec", "setting", "bpp", "compression_ratio", "psnr_full", "ssim_full"],
        "dct_summary.csv": ["source", "quality", "bpp", "compression_ratio", "psnr_full", "ssim_full"],
        "dwt_summary.csv": ["source", "quality", "bpp", "compression_ratio", "psnr_full", "ssim_full"],
        "dwt_tuning_summary.csv": ["group", "configuration", "mean_bpp_q50", "mean_psnr_q50"],
        "dwt_ablation_summary.csv": ["wavelet", "levels", "table_mode", "mean_bpp", "mean_psnr", "mean_ssim"],
        "lossless_summary.csv": ["source", "images_count", "mean_roi_pixels", "mean_roi_bytes", "mean_roi_bpp", "roi_exact_fraction"],
        "roi_codec_summary.csv": ["source", "method", "quality", "mean_bpp", "mean_cr", "mean_ratio_vs_png", "mean_psnr_full", "roi_lossless_fraction"],
        "roi_fill_ablation.csv": ["method", "quality", "fill", "mean_total_bytes", "mean_mask_bytes", "mean_roi_bytes"],
        "segmentation_summary.csv": ["group_type", "group_name", "count", "dice_mean", "iou_mean", "recall_mean"],
        "segmentation_margins.csv": ["source", "margin_type", "mean_dice", "mean_recall", "mean_area_ratio"],
        "roi_codec_predicted_summary.csv": ["source", "variant", "d_pixels", "count", "mean_total_bytes", "mean_pct_lung_outside", "mean_psnr_lung_true"],
        "classifier_metrics.csv": ["group", "n_samples", "accuracy", "recall", "specificity", "auc"],
        "classifier_crossscanner.csv": ["setting", "train_source", "test_source", "n_samples", "accuracy", "recall", "specificity", "auc"],
        "judge_metrics.csv": ["group", "n_samples", "accuracy", "recall", "specificity", "auc"],
        "benchmark_summary.csv": ["codec", "setting", "n_samples", "mean_bpp", "auc", "flip_rate"],
        "benchmark_summary_per_source.csv": ["source", "codec", "setting", "n_samples", "mean_bpp", "auc", "flip_rate"],
        "benchmark_operating_points.csv": ["codec", "bpp_flip_le_5pct", "bpp_flip_le_2pct", "bpp_auc_loss_le_0.01"],
        "region_ablation_summary.csv": ["source", "variant", "quality", "n_samples", "auc", "flip_rate", "mean_abs_prob_diff"],
        "graded_pilot_summary.csv": ["family", "variant", "delta_core", "delta_band", "band_px", "bg_quality", "bpp", "total_bytes", "flip_rate"],
        "graded_pilot_guarantee_cost.csv": ["family", "bpp_for_p99_le_2", "bpp_for_p99_le_4", "bpp_for_p99_le_8", "bpp_for_flip_le_0.05"],
        "controller_summary.csv": ["family", "tolerance", "n_images", "mean_bpp", "steering_flip_rate", "judge_flip_rate"],
        "controller_vs_fixed.csv": ["family", "tolerance", "scope", "controller_judge_flip_rate", "controller_mean_bpp", "status"],
        "controller_variants_summary.csv": ["family", "variant_tolerance", "mean_bpp", "judge_flip_rate_pct", "byte_saving", "status"],
    }

    for table_file, req_cols in table_requirements.items():
        p = TABLES_DIR / table_file
        assert p.is_file(), f"Table file not found: {p}"
        df = pd.read_csv(p)
        for col in req_cols:
            assert col in df.columns, f"Required column '{col}' missing from {table_file}"
            nan_cnt = df[col].isna().sum()
            assert nan_cnt == 0, f"Required column '{col}' in {table_file} contains {nan_cnt} NaN values"
