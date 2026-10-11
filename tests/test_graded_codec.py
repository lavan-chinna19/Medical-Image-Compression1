"""Unit tests for graded zone compression codec."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, WORK_DIR
from medcomp.graded_codec import dilate_mask, graded_decode, graded_encode, make_zones
from medcomp.io_utils import load_image
from medcomp.roi_codec import roi_encode

PRED_MASK_DIR = WORK_DIR / "pred" / "mask"


def test_graded_round_trip_uses_only_bytes():
    """Verify graded_decode reconstructs image from bitstream bytes alone."""
    np.random.seed(42)
    h, w = 128, 128
    img = np.random.randint(20, 230, (h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[30:90, 30:90] = True

    params = {
        "delta_core": 0,
        "delta_band": 2,
        "band_px": 4,
        "bg_quality": 30,
        "bg_method": "dwt",
        "fill": "inpaint",
    }

    comp_bytes, recon_enc, parts = graded_encode(img, mask, params)
    assert len(comp_bytes) == sum(parts.values())

    recon_dec = graded_decode(comp_bytes)
    np.testing.assert_array_equal(recon_dec, recon_enc)


def test_zone_map_identical_in_encoder_and_decoder():
    """Verify zone map construction is identical and deterministic."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True

    zm1 = make_zones(mask, band_px=4)
    zm2 = make_zones(mask, band_px=4)
    np.testing.assert_array_equal(zm1, zm2)

    # Core is 1, band is 2, bg is 0
    assert np.all(zm1[mask] == 1)
    dilated = dilate_mask(mask, 4)
    band_pixels = dilated & (~mask)
    assert np.all(zm1[band_pixels] == 2)
    assert np.all(zm1[~dilated] == 0)


def test_lossless_dilated_is_exact_on_core():
    """Verify lossless_dilated variant (delta_core=0, no band) is bit-exact on its dilated mask."""
    np.random.seed(99)
    img = np.random.randint(0, 256, (120, 120), dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=bool)
    mask[30:80, 30:80] = True
    dilated_core = dilate_mask(mask, 4)

    params = {
        "delta_core": 0,
        "delta_band": 0,
        "band_px": 0,
        "bg_quality": 50,
        "bg_method": "dwt",
        "fill": "inpaint",
    }

    comp_bytes, recon, _ = graded_encode(img, dilated_core, params)
    dec = graded_decode(comp_bytes)

    np.testing.assert_array_equal(recon[dilated_core], img[dilated_core])
    np.testing.assert_array_equal(dec[dilated_core], img[dilated_core])


def test_lossless_dilated_matches_roi_codec_size_within_1_percent():
    """Verify lossless_dilated total size matches roi_codec within 1% on 5 real images."""
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Manifest not found")

    df = pd.read_csv(manifest_path)
    # Pick 5 images
    test_stems = df["stem"].iloc[:5].tolist()

    for stem in test_stems:
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = PRED_MASK_DIR / f"{stem}.png"
        if not img_path.is_file() or not mask_path.is_file():
            pytest.skip(f"Data for {stem} not available")

        img = load_image(img_path)
        base_mask = (load_image(mask_path) > 127)
        dilated_m = dilate_mask(base_mask, d_pixels=4)

        # 1. roi_codec (DWT, quality=50, inpaint)
        roi_bytes, _, _ = roi_encode(img, dilated_m, method="dwt", quality=50, fill="inpaint")
        size_roi = len(roi_bytes)

        # 2. graded_codec (delta_core=0, band_px=0 on dilated_m, DWT, quality=50, inpaint)
        params = {
            "delta_core": 0,
            "delta_band": 0,
            "band_px": 0,
            "bg_quality": 50,
            "bg_method": "dwt",
            "fill": "inpaint",
        }
        graded_bytes, _, _ = graded_encode(img, dilated_m, params)
        size_graded = len(graded_bytes)

        # Relative difference
        rel_diff = abs(size_graded - size_roi) / size_roi
        assert rel_diff < 0.01, f"Stem {stem}: size difference {rel_diff*100:.2f}% exceeds 1% (ROI={size_roi}, Graded={size_graded})"
