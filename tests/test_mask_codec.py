"""Tests for custom binary mask codec (row-wise RLE and BBox RLE)."""

from pathlib import Path
import numpy as np
import pytest

from medcomp.config import PROCESSED_MASKS_DIR
from medcomp.io_utils import load_image
from medcomp.mask_codec import mask_decode, mask_encode


def test_mask_roundtrip_empty_and_full():
    """Verify all-zeros and all-ones masks round-trip exactly in both modes."""
    for mode in [False, True]:
        empty = np.zeros((64, 64), dtype=bool)
        d_empty = mask_encode(empty, bbox_mode=mode)
        r_empty = mask_decode(d_empty)
        assert np.array_equal(r_empty, empty)

        full = np.ones((64, 64), dtype=bool)
        d_full = mask_encode(full, bbox_mode=mode)
        r_full = mask_decode(d_full)
        assert np.array_equal(r_full, full)


def test_mask_roundtrip_single_pixel():
    """Mask with a single True pixel round trips exactly."""
    for mode in [False, True]:
        mask = np.zeros((80, 80), dtype=bool)
        mask[42, 37] = True
        data = mask_encode(mask, bbox_mode=mode)
        recon = mask_decode(data)
        assert np.array_equal(recon, mask)


def test_mask_roundtrip_stripes():
    """Horizontal and vertical striped patterns round trip bit-exact."""
    # Horizontal stripes
    h_stripes = np.zeros((64, 64), dtype=bool)
    h_stripes[::2, :] = True

    # Vertical stripes
    v_stripes = np.zeros((64, 64), dtype=bool)
    v_stripes[:, ::3] = True

    for pattern in [h_stripes, v_stripes]:
        for mode in [False, True]:
            data = mask_encode(pattern, bbox_mode=mode)
            recon = mask_decode(data)
            assert np.array_equal(recon, pattern)


def test_mask_roundtrip_odd_shapes():
    """Arbitrary non-square odd shapes round trip exactly."""
    rng = np.random.default_rng(101)
    for shape in [(67, 93), (93, 67)]:
        mask = rng.random(shape) > 0.5
        for mode in [False, True]:
            data = mask_encode(mask, bbox_mode=mode)
            recon = mask_decode(data)
            assert np.array_equal(recon, mask)


def test_mask_roundtrip_real_lung_mask():
    """Real clinical chest X-ray lung mask round trips bit-for-bit."""
    if not PROCESSED_MASKS_DIR.is_dir():
        pytest.skip("Processed masks directory not found.")

    candidates = sorted(PROCESSED_MASKS_DIR.glob("*.png"))
    if not candidates:
        pytest.skip("No masks found.")

    real_mask = (load_image(candidates[0]) > 127)

    for mode in [False, True]:
        data = mask_encode(real_mask, bbox_mode=mode)
        recon = mask_decode(data)
        assert np.array_equal(recon, real_mask)


def test_mask_deterministic_output():
    """Encoding the same mask twice yields identical byte sequences."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 15:45] = True

    d1 = mask_encode(mask, bbox_mode=False)
    d2 = mask_encode(mask, bbox_mode=False)
    assert d1 == d2

    d3 = mask_encode(mask, bbox_mode=True)
    d4 = mask_encode(mask, bbox_mode=True)
    assert d3 == d4
