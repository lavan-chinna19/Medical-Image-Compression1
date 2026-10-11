"""Tests for custom lossless image codec (full image and masked ROI)."""

from pathlib import Path
import numpy as np
import pytest

from medcomp.config import PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR
from medcomp.io_utils import load_image
from medcomp.lossless_codec import lossless_decode, lossless_encode, lossless_encode_with_stats


def make_synthetic_image(shape: tuple[int, int] = (128, 128), seed: int = 42) -> np.ndarray:
    """Generate reproducible test image with gradient and noise."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y = np.linspace(20, 200, h)[:, None]
    x = np.linspace(10, 100, w)[None, :]
    base = y + x
    noise = rng.normal(0, 10, size=shape)
    return np.clip(np.round(base + noise), 0, 255).astype(np.uint8)


def test_full_image_roundtrip_random():
    """Lossless encode/decode bit-for-bit exact round trip on full random image."""
    img = make_synthetic_image((128, 128), seed=1)
    data = lossless_encode(img, mask=None, context=False)
    recon = lossless_decode(data, mask=None)
    assert np.array_equal(recon, img)


def test_full_image_roundtrip_context():
    """Lossless encode/decode bit-for-bit exact round trip with context=True."""
    img = make_synthetic_image((128, 128), seed=2)
    data = lossless_encode(img, mask=None, context=True)
    recon = lossless_decode(data, mask=None)
    assert np.array_equal(recon, img)


def test_constant_images():
    """Lossless codec handles flat constant images (0, 128, 255)."""
    for val in [0, 128, 255]:
        img = np.full((80, 80), val, dtype=np.uint8)
        data = lossless_encode(img)
        recon = lossless_decode(data)
        assert np.array_equal(recon, img)


def test_odd_shapes():
    """Lossless codec handles arbitrary non-multiple odd shapes (67x93, 93x67)."""
    for shape in [(67, 93), (93, 67)]:
        img = make_synthetic_image(shape, seed=3)
        data = lossless_encode(img)
        recon = lossless_decode(data)
        assert np.array_equal(recon, img)


def test_masked_roi_empty_mask():
    """Empty mask (all False) encodes cleanly and decodes to all zeros."""
    img = make_synthetic_image((64, 64), seed=4)
    mask = np.zeros(img.shape, dtype=bool)
    data = lossless_encode(img, mask=mask)
    recon = lossless_decode(data, mask=mask)
    assert recon.shape == img.shape
    assert np.all(recon == 0)


def test_masked_roi_full_mask():
    """Full mask (all True) decodes bit-for-bit identical to full image."""
    img = make_synthetic_image((64, 64), seed=5)
    mask = np.ones(img.shape, dtype=bool)
    data = lossless_encode(img, mask=mask)
    recon = lossless_decode(data, mask=mask)
    assert np.array_equal(recon, img)


def test_masked_roi_random_mask():
    """Random sparse mask preserves exact pixel values at ROI positions."""
    img = make_synthetic_image((96, 96), seed=6)
    rng = np.random.default_rng(7)
    mask = rng.random(img.shape) > 0.65  # ~35% ROI

    for ctx in [False, True]:
        data = lossless_encode(img, mask=mask, context=ctx)
        recon = lossless_decode(data, mask=mask)
        assert np.array_equal(recon[mask], img[mask])
        assert np.all(recon[~mask] == 0)


def test_real_processed_image_with_lung_mask():
    """Exact bit-for-bit round trip on real processed image with its clinical lung mask."""
    if not PROCESSED_IMAGES_DIR.is_dir() or not PROCESSED_MASKS_DIR.is_dir():
        pytest.skip("Processed images/masks directory not found.")

    candidates = sorted(PROCESSED_MASKS_DIR.glob("*.png"))
    if not candidates:
        pytest.skip("No masks found.")

    stem = candidates[0].stem
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
    if not img_path.is_file():
        pytest.skip(f"Image {img_path} not found.")

    img = load_image(img_path)
    mask = (load_image(mask_path) > 127)

    for ctx in [False, True]:
        data, stats = lossless_encode_with_stats(img, mask=mask, context=ctx)
        recon = lossless_decode(data, mask=mask)
        assert np.array_equal(recon[mask], img[mask])


def test_entropy_code_length_bound():
    """Average code length must stay within 1 bit of residual entropy (H <= L < H + 1)."""
    img = make_synthetic_image((128, 128), seed=8)
    _, stats = lossless_encode_with_stats(img, mask=None, context=False)
    ent = stats["residual_entropy"]
    avg_len = stats["avg_code_length"]

    assert avg_len >= ent - 1e-4, f"L ({avg_len}) must be >= H ({ent})"
    assert avg_len < ent + 1.0, f"L ({avg_len}) must be < H + 1 ({ent + 1.0})"


def test_deterministic_output():
    """Encoding twice with identical parameters yields identical bitstreams."""
    img = make_synthetic_image((64, 64), seed=9)
    d1 = lossless_encode(img, context=False)
    d2 = lossless_encode(img, context=False)
    assert d1 == d2
