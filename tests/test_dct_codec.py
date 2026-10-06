"""Unit tests for custom DCT image codec and quantization tables."""

from pathlib import Path

import numpy as np
import pytest

from medcomp.config import PROCESSED_IMAGES_DIR
from medcomp.dct_codec import (
    dct_decode,
    dct_encode,
    get_quantization_table,
)
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import psnr


def test_dct_decode_equals_encoder_recon_bit_exact():
    """Decoder output must strictly match the encoder's internal dequantize+IDCT reconstruction."""
    rng = np.random.default_rng(999)
    img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)

    comp_bytes, recon_enc = dct_encode(img, quality=60)
    recon_dec = dct_decode(comp_bytes)

    assert recon_dec.shape == img.shape
    assert recon_dec.dtype == np.uint8
    assert np.array_equal(recon_enc, recon_dec)


def test_dct_non_multiple_of_8_dimensions():
    """Codec must handle arbitrary non-multiple-of-8 dimensions (67x93 and 93x67)."""
    rng = np.random.default_rng(777)
    for shape in [(67, 93), (93, 67)]:
        img = rng.integers(0, 256, size=shape, dtype=np.uint8)

        comp_bytes, recon_enc = dct_encode(img, quality=40)
        recon_dec = dct_decode(comp_bytes)

        assert recon_dec.shape == shape
        assert np.array_equal(recon_enc, recon_dec)


def test_dct_constant_and_zero_images():
    """Codec must successfully encode and decode constant and all-zero images."""
    # All-zero image
    zero_img = np.zeros((64, 64), dtype=np.uint8)
    comp_zero, recon_zero = dct_encode(zero_img, quality=50)
    dec_zero = dct_decode(comp_zero)
    assert np.array_equal(recon_zero, dec_zero)
    assert np.array_equal(dec_zero, zero_img)

    # Constant image (e.g. pixel value 142)
    const_img = np.full((64, 64), 142, dtype=np.uint8)
    comp_const, recon_const = dct_encode(const_img, quality=50)
    dec_const = dct_decode(comp_const)
    assert np.array_equal(recon_const, dec_const)


def test_dct_size_decreases_as_quality_decreases():
    """Compressed bitstream length must decrease monotonically as quality decreases."""
    rng = np.random.default_rng(123)
    # Synthetic image with structure and gradients
    x = np.linspace(0, 255, 128, dtype=np.uint8)
    img = np.clip(np.tile(x, (128, 1)) + rng.integers(0, 40, (128, 128)), 0, 255).astype(np.uint8)

    qualities = [90, 70, 50, 30, 20, 10]
    sizes = []
    for q in qualities:
        comp_bytes, _ = dct_encode(img, quality=q)
        sizes.append(len(comp_bytes))

    # Higher quality (earlier in list) must produce larger or equal byte size
    for i in range(len(sizes) - 1):
        assert sizes[i] >= sizes[i + 1], f"Size at Q={qualities[i]} ({sizes[i]}) < Q={qualities[i+1]} ({sizes[i+1]})"


def test_dct_psnr_quality_90_above_35db():
    """PSNR at quality=90 must exceed 35 dB on a real preprocessed medical image."""
    images = list_images(PROCESSED_IMAGES_DIR) if PROCESSED_IMAGES_DIR.exists() else []
    if not images:
        pytest.skip("Processed images not found. Run preprocess.py first.")

    sample_img = load_image(images[0])
    comp_bytes, recon = dct_encode(sample_img, quality=90)
    recon_dec = dct_decode(comp_bytes)

    assert np.array_equal(recon, recon_dec)
    measured_psnr = psnr(sample_img, recon_dec)
    assert measured_psnr > 35.0, f"Expected PSNR > 35.0 dB at Q=90, got {measured_psnr:.2f} dB"


def test_dct_real_images_montgomery_and_shenzhen():
    """Verify decode(encode(img)) equals encoder's recon exactly on real Montgomery and Shenzhen images."""
    if not PROCESSED_IMAGES_DIR.exists():
        pytest.skip("Processed images directory not found.")

    mcu_images = sorted(PROCESSED_IMAGES_DIR.glob("MCUCXR_*.png"))
    chn_images = sorted(PROCESSED_IMAGES_DIR.glob("CHNCXR_*.png"))

    if not mcu_images or not chn_images:
        pytest.skip("Processed Montgomery or Shenzhen images not found.")

    for img_path in [mcu_images[0], chn_images[0]]:
        img = load_image(img_path)
        for q in [20, 50, 90]:
            comp_bytes, recon_enc = dct_encode(img, quality=q)
            recon_dec = dct_decode(comp_bytes)

            assert recon_dec.shape == img.shape
            assert np.array_equal(recon_enc, recon_dec)


def test_dct_output_is_deterministic():
    """Multiple encode runs on the same input must yield identical bitstream bytes."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)

    bytes1, recon1 = dct_encode(img, quality=50)
    bytes2, recon2 = dct_encode(img, quality=50)

    assert bytes1 == bytes2
    assert np.array_equal(recon1, recon2)
