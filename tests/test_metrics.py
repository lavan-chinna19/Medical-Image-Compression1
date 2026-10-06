"""Unit tests for distortion, rate, and entropy metrics."""

import math
import numpy as np
import pytest

from medcomp.metrics import (
    bits_per_pixel,
    compression_ratio,
    is_lossless,
    masked_psnr,
    mse,
    psnr,
    shannon_entropy,
    ssim,
)


def test_identical_images_psnr_and_ssim():
    """Identical images must give PSNR = inf and SSIM = 1.0."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    assert math.isinf(psnr(img, img))
    assert psnr(img, img) > 0  # +inf
    assert ssim(img, img) == pytest.approx(1.0, rel=1e-5)


def test_known_mse_synthetic_pair():
    """Known MSE verification on a synthetic pair."""
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 5, dtype=np.uint8)

    expected_mse = 25.0
    assert mse(a, b) == pytest.approx(expected_mse, rel=1e-5)

    expected_psnr = 10.0 * np.log10((255.0 ** 2) / 25.0)
    assert psnr(a, b) == pytest.approx(expected_psnr, rel=1e-5)


def test_shannon_entropy_constant_and_uniform():
    """Entropy of constant image is 0; uniform random 8-bit image is ~8.0."""
    constant = np.full((100, 100), 128, dtype=np.uint8)
    assert shannon_entropy(constant) == 0.0

    # Perfectly uniform 8-bit distribution: exactly 1000 of every byte [0..255]
    uniform = np.repeat(np.arange(256, dtype=np.uint8), 1000)
    assert shannon_entropy(uniform) == pytest.approx(8.0, abs=1e-4)

    # Random uniform distribution is very close to 8.0 bits
    rng = np.random.default_rng(123)
    rand_img = rng.integers(0, 256, size=(500, 500), dtype=np.uint8)
    assert shannon_entropy(rand_img) == pytest.approx(8.0, abs=0.05)


def test_compression_ratio_and_bpp():
    """Verify compression ratio and bits per pixel computations."""
    orig_bytes = 10000
    comp_bytes = 2500
    pixels = 10000

    assert compression_ratio(orig_bytes, comp_bytes) == pytest.approx(4.0)
    # 2500 bytes * 8 bits / 10000 pixels = 2.0 bpp
    assert bits_per_pixel(comp_bytes, pixels) == pytest.approx(2.0)

    with pytest.raises(ValueError):
        compression_ratio(orig_bytes, 0)
    with pytest.raises(ValueError):
        bits_per_pixel(comp_bytes, 0)


def test_is_lossless_global_and_roi():
    """Verify is_lossless for full image and masked ROI region."""
    orig = np.arange(100, dtype=np.uint8).reshape((10, 10))
    recon_perfect = orig.copy()
    recon_imperfect = orig.copy()
    recon_imperfect[0, 0] = np.uint8(255 if orig[0, 0] == 0 else 0)

    # Full image
    assert is_lossless(orig, recon_perfect) is True
    assert is_lossless(orig, recon_imperfect) is False

    # Masked region: define ROI at bottom half [5:, :]
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5:, :] = 255

    # Since perturbation was at [0, 0] (background), ROI is perfectly lossless!
    assert is_lossless(orig, recon_imperfect, mask=mask) is True

    # Perturb inside ROI
    recon_roi_perturbed = orig.copy()
    recon_roi_perturbed[8, 8] = 255
    assert is_lossless(orig, recon_roi_perturbed, mask=mask) is False


def test_masked_psnr():
    """Verify masked_psnr handles ROI and background partitions correctly."""
    orig = np.zeros((10, 10), dtype=np.uint8)
    recon = np.zeros((10, 10), dtype=np.uint8)

    # ROI has a difference of 10, background has difference of 0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 255  # top half is ROI
    recon[:5, :] = 10  # difference in ROI only

    # Background is identical -> PSNR inf
    assert math.isinf(masked_psnr(orig, recon, mask, region="background"))

    # ROI has error 100 -> PSNR = 10 * log10(255^2 / 100)
    expected_roi_psnr = 10.0 * np.log10((255.0 ** 2) / 100.0)
    assert masked_psnr(orig, recon, mask, region="roi") == pytest.approx(expected_roi_psnr, rel=1e-5)


def test_shape_mismatch_raises_error():
    """All distortion functions must raise ValueError on shape mismatch."""
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 12), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        mse(a, b)
    with pytest.raises(ValueError):
        psnr(a, b)
    with pytest.raises(ValueError):
        ssim(a, b)
    with pytest.raises(ValueError):
        is_lossless(a, b)
    with pytest.raises(ValueError):
        masked_psnr(a, b, mask)
    with pytest.raises(ValueError):
        masked_psnr(a, a, b)
