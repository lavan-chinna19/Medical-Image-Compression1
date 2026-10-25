"""End-to-End Tests for Technique Comparison Evaluation on MCUCXR_0001_0.png."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from medcomp.comparison import (
    TECHNIQUE_ORDER,
    evaluate_technique,
    run_all_techniques,
)
from medcomp.config import PROCESSED_IMAGES_DIR
from medcomp.data_utils import resize_long_side


@pytest.fixture(scope="module")
def mcucxr_image():
    """Load and resize MCUCXR_0001_0.png."""
    img_path = PROCESSED_IMAGES_DIR / "MCUCXR_0001_0.png"
    assert img_path.is_file(), f"Test image missing: {img_path}"
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert raw is not None, "Failed to read image"
    return resize_long_side(raw, target=512)


def test_mcucxr_all_six_techniques_e2e(mcucxr_image):
    """Verify that all six techniques execute and calculate actual metrics on MCUCXR_0001_0.png."""
    h, w = mcucxr_image.shape
    raw_bytes = h * w

    results = run_all_techniques(
        img_uint8=mcucxr_image,
        params={
            "quality": 50,
            "ratio": 10.0,
            "delta_core": 2,
            "delta_band": 4,
            "band_px": 8,
            "bg_quality": 50,
        },
    )

    assert len(results) == 6
    assert [r.technique_key for r in results] == TECHNIQUE_ORDER

    for r in results:
        # 1. Successful execution status
        assert r.status == "SUCCESS", f"Technique {r.technique_key} failed: {r.error_message}"

        # 2. Actual compressed bytes generated
        assert r.compressed_bytes is not None and r.compressed_bytes > 0
        assert r.compressed_bytes < raw_bytes, "Compressed bytes should be smaller than raw uncompressed image"

        # 3. Actual reconstructed image returned
        assert r.recon_img is not None
        assert r.recon_img.shape == mcucxr_image.shape
        assert r.recon_img.dtype == np.uint8

        # 4. Image quality metrics calculated
        assert r.psnr is not None and 30.0 <= r.psnr <= 60.0
        assert r.ssim is not None and 0.90 <= r.ssim <= 1.0
        assert r.mse is not None and 0.0 <= r.mse <= 20.0
        assert r.r2_score is not None and 0.95 <= r.r2_score <= 1.0

        # 5. Compression ratio calculated from actual sizes
        assert r.compression_ratio is not None and r.compression_ratio > 1.0
        assert pytest.approx(r.compression_ratio, rel=1e-3) == float(raw_bytes) / float(r.compressed_bytes)

        # 6. Reconstruction time measured
        assert r.reconstruction_time is not None and r.reconstruction_time > 0.0

        # 7. Classification metrics appropriately marked N/A
        assert r.accuracy == "N/A"
        assert r.precision == "N/A"
        assert r.recall == "N/A"

        # 8. Display dict format
        d = r.to_display_dict()
        assert d["Accuracy (%)"] == "N/A"
        assert d["Precision (%)"] == "N/A"
        assert d["Recall (%)"] == "N/A"
        assert ":" in d["Compression Ratio"]  # Format X.XX:1
        assert "—" not in [d["PSNR (dB)"], d["SSIM"], d["MSE"], d["Compression Ratio"], d["Reconstruction Time (s)"]]


def test_mcucxr_winners_and_summary_cards(mcucxr_image):
    """Verify that summary card winners are calculated accurately from real results."""
    results = run_all_techniques(
        img_uint8=mcucxr_image,
        params={"quality": 50, "ratio": 10.0, "delta_core": 2, "delta_band": 4, "band_px": 8, "bg_quality": 50},
    )

    valid = [r for r in results if r.status == "SUCCESS"]
    assert len(valid) == 6

    best_psnr = max(valid, key=lambda r: r.psnr)
    best_ssim = max(valid, key=lambda r: r.ssim)
    lowest_mse = min(valid, key=lambda r: r.mse)
    highest_cr = max(valid, key=lambda r: r.compression_ratio)
    fastest_time = min(valid, key=lambda r: r.reconstruction_time)

    # Valid winners exist and have sensible values
    assert best_psnr.technique_name in ["JPEG2000", "Proposed Technique", "DCT", "DWT", "JPEG", "Hybrid DCT–DWT"]
    assert best_psnr.psnr >= 40.0
    assert best_ssim.ssim >= 0.96
    assert lowest_mse.mse <= 6.5
    assert highest_cr.compression_ratio >= 6.0
    assert fastest_time.reconstruction_time < 0.1
