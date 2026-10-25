"""Unit tests for Comparative Analysis Engine and 9 Evaluation Metrics."""

from __future__ import annotations

import numpy as np
import pytest

from medcomp.comparison import (
    TECHNIQUE_ORDER,
    compute_r2_score,
    evaluate_technique,
    get_initial_comparison_df,
    run_all_techniques,
)


def test_compute_r2_score_identical():
    """Identical images must yield R^2 = 1.0."""
    orig = np.random.randint(20, 230, (64, 64), dtype=np.uint8)
    r2 = compute_r2_score(orig, orig.copy())
    assert pytest.approx(r2, rel=1e-5) == 1.0


def test_compute_r2_score_constant_image():
    """Constant image handling should not divide by zero."""
    orig = np.full((64, 64), 128, dtype=np.uint8)
    recon_same = np.full((64, 64), 128, dtype=np.uint8)
    assert compute_r2_score(orig, recon_same) == 1.0

    recon_diff = np.full((64, 64), 130, dtype=np.uint8)
    assert compute_r2_score(orig, recon_diff) == 0.0


def test_compute_r2_score_shape_mismatch():
    """Shape mismatch must raise ValueError."""
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.zeros((32, 33), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_r2_score(a, b)


def test_initial_comparison_df():
    """Verify initial comparison table format and dashed placeholders."""
    df = get_initial_comparison_df()
    assert len(df) == 6
    assert list(df["S. No."]) == [1, 2, 3, 4, 5, 6]
    expected_techniques = ["DCT", "DWT", "JPEG", "JPEG2000", "Hybrid DCT–DWT", "Proposed Technique"]
    assert list(df["Compression Technique"]) == expected_techniques
    # Check dashed uncomputed results
    for col in ["Accuracy (%)", "Precision (%)", "Recall (%)", "R² Score", "PSNR (dB)", "SSIM", "MSE", "Compression Ratio", "Reconstruction Time (s)"]:
        assert all(val == "—" for val in df[col])


@pytest.mark.parametrize("tech_key", TECHNIQUE_ORDER)
def test_evaluate_technique_smoke(tech_key: str):
    """Smoke test each of the 6 techniques on a synthetic medical phantom image."""
    # Create smooth synthetic medical phantom
    y, x = np.ogrid[:128, :128]
    phantom = np.clip(128 + 60 * np.sin(x / 10.0) + 40 * np.cos(y / 12.0), 0, 255).astype(np.uint8)

    # Optional synthetic lung mask
    mask = ((x - 64) ** 2 + (y - 64) ** 2 < 40 ** 2).astype(bool)

    res = evaluate_technique(tech_key, phantom, mask_bool=mask)

    assert res.status == "SUCCESS", f"Technique {tech_key} failed: {res.error_message}"
    assert res.accuracy == "N/A"
    assert res.precision == "N/A"
    assert res.recall == "N/A"
    assert res.r2_score is not None and -10.0 <= res.r2_score <= 1.0
    assert res.psnr is not None and (res.psnr > 15.0 or np.isinf(res.psnr))
    assert res.ssim is not None and 0.0 <= res.ssim <= 1.0
    assert res.mse is not None and res.mse >= 0.0
    assert res.compression_ratio is not None and res.compression_ratio > 0.0
    assert res.reconstruction_time is not None and res.reconstruction_time > 0.0
    assert res.recon_img is not None and res.recon_img.shape == phantom.shape


def test_run_all_techniques():
    """Verify run_all_techniques executes all 6 methods sequentially."""
    img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
    results = run_all_techniques(img)
    assert len(results) == 6
    assert [r.technique_key for r in results] == TECHNIQUE_ORDER
    assert all(r.status == "SUCCESS" for r in results)
