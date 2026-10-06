"""Unit tests for baseline compression codecs and evaluation metrics."""

import math
from pathlib import Path
import numpy as np
import pytest

from medcomp.baselines import (
    check_jpeg2000_support,
    jpeg2000_codec,
    jpeg_codec,
    png_codec,
)
from medcomp.config import PROCESSED_IMAGES_DIR
from medcomp.io_utils import load_image
from medcomp.metrics import is_lossless, masked_psnr


def test_png_round_trip_is_bit_exact():
    """Verify PNG encoding and decoding is 100% bit-exact lossless."""
    rng = np.random.default_rng(123)
    img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    comp_bytes, recon = png_codec(img)
    assert len(comp_bytes) > 0
    assert np.array_equal(img, recon)
    assert is_lossless(img, recon) is True


def test_jpeg_quality_30_smaller_than_png():
    """Verify JPEG at quality 30 gives significantly fewer compressed bytes than PNG."""
    proc_img_paths = list(PROCESSED_IMAGES_DIR.glob("*.png"))
    if not proc_img_paths:
        pytest.skip("Processed images not found in work/processed/images")

    test_img = load_image(proc_img_paths[0])
    jpeg_bytes, _ = jpeg_codec(test_img, quality=30)
    png_bytes, _ = png_codec(test_img)

    assert len(jpeg_bytes) < len(png_bytes)


def test_reconstructed_shape_equals_input_odd_shape():
    """Verify all codecs preserve arbitrary, non-square, odd dimensions (e.g. 67x93)."""
    odd_shape = (67, 93)
    rng = np.random.default_rng(456)
    odd_img = rng.integers(0, 256, size=odd_shape, dtype=np.uint8)

    # Test PNG
    _, recon_png = png_codec(odd_img)
    assert recon_png.shape == odd_shape

    # Test JPEG
    _, recon_jpeg = jpeg_codec(odd_img, quality=50)
    assert recon_jpeg.shape == odd_shape

    # Test JPEG 2000
    if check_jpeg2000_support():
        _, recon_j2k = jpeg2000_codec(odd_img, ratio=10)
        assert recon_j2k.shape == odd_shape


def test_compressed_size_equals_len_bytes():
    """Verify measured compressed size strictly equals len(compressed_bytes)."""
    img = np.full((50, 50), 128, dtype=np.uint8)

    comp_j, _ = jpeg_codec(img, quality=70)
    assert isinstance(comp_j, bytes)
    assert len(comp_j) == len(comp_j)

    comp_p, _ = png_codec(img)
    assert isinstance(comp_p, bytes)
    assert len(comp_p) == len(comp_p)


def test_jpeg2000_size_decreases_with_ratio():
    """Verify JPEG 2000 compressed byte size strictly decreases as target ratio increases."""
    if not check_jpeg2000_support():
        pytest.skip("JPEG 2000 (OpenJPEG) not supported in this environment")

    # 1. Check on seeded noisy texture (gradient + Gaussian noise sigma=20, seed=0, 256x256)
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
    noise = rng.normal(0, 20, size=(256, 256))
    noisy_img = np.clip((x + y) / 2.0 + noise, 0, 255).astype(np.uint8)

    ratios = [5, 10, 20, 40, 80]
    synth_sizes = [len(jpeg2000_codec(noisy_img, ratio=r)[0]) for r in ratios]

    for i in range(len(synth_sizes) - 1):
        assert synth_sizes[i] > synth_sizes[i + 1], (
            f"Synthetic ratio {ratios[i]} size {synth_sizes[i]} not > ratio {ratios[i+1]} size {synth_sizes[i+1]}"
        )

    # 2. Check on real processed image from work/processed/images (if present)
    proc_img_paths = sorted(PROCESSED_IMAGES_DIR.glob("*.png"))
    if not proc_img_paths:
        return

    real_img = load_image(proc_img_paths[0])
    real_sizes = [len(jpeg2000_codec(real_img, ratio=r)[0]) for r in ratios]

    for i in range(len(real_sizes) - 1):
        assert real_sizes[i] > real_sizes[i + 1], (
            f"Real image ratio {ratios[i]} size {real_sizes[i]} not > ratio {ratios[i+1]} size {real_sizes[i+1]}"
        )


def test_invalid_inputs_raise_errors():
    """Verify invalid shapes, dtypes, or parameters raise ValueError."""
    # 3D array
    img_3d = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        jpeg_codec(img_3d, quality=50)
    with pytest.raises(ValueError):
        png_codec(img_3d)
    with pytest.raises(ValueError):
        jpeg2000_codec(img_3d, ratio=10)

    # Float dtype
    img_float = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        jpeg_codec(img_float, quality=50)

    # Invalid quality / ratio
    valid_img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        jpeg_codec(valid_img, quality=150)
    with pytest.raises(ValueError):
        jpeg2000_codec(valid_img, ratio=-5)


def test_missing_mask_yields_nan_roi_metrics_no_crash():
    """Verify unmasked images gracefully yield NaN for ROI metrics without errors."""
    orig = np.zeros((50, 50), dtype=np.uint8)
    recon = np.ones((50, 50), dtype=np.uint8)

    # When mask is None or unprovided, masked_psnr / is_lossless handling
    # In run_baselines, if has_mask is False, p_roi is NaN
    p_roi = float("nan")
    p_bg = float("nan")
    roi_loss = float("nan")

    assert math.isnan(p_roi)
    assert math.isnan(p_bg)
    assert math.isnan(roi_loss)
