"""Unit tests for near-lossless coding with bounded error guarantees."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR
from medcomp.io_utils import load_image
from medcomp.lossless_codec import lossless_decode, lossless_encode
from medcomp.nearlossless import nl_decode, nl_encode


def test_nl_round_trip_matches_encoder_recon():
    """Verify that nl_decode exactly reproduces nl_encode reconstruction array."""
    np.random.seed(42)
    h, w = 120, 100
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    # 3 zones
    zone_map = np.zeros((h, w), dtype=np.uint8)
    zone_map[20:70, 20:80] = 1
    zone_map[10:80, 10:90][zone_map[10:80, 10:90] == 0] = 2

    deltas = {1: 1, 2: 3}
    comp_bytes, recon_enc = nl_encode(img, zone_map, deltas)
    recon_dec = nl_decode(comp_bytes, zone_map)

    np.testing.assert_array_equal(recon_dec, recon_enc)


@pytest.mark.parametrize("delta", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("shape", [(67, 93), (93, 67)])
def test_max_error_bounded_on_odd_shapes(delta: int, shape: tuple[int, int]):
    """Verify max absolute error is strictly <= delta for various deltas and odd shapes."""
    np.random.seed(delta * 10 + shape[0])
    h, w = shape
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    zone_map = np.zeros((h, w), dtype=np.uint8)
    zone_map[15:55, 15:75] = 1

    deltas = {1: delta, 2: delta}
    comp_bytes, recon_enc = nl_encode(img, zone_map, deltas)
    recon_dec = nl_decode(comp_bytes, zone_map)

    coded_mask = (zone_map == 1)
    abs_err = np.abs(img[coded_mask].astype(int) - recon_dec[coded_mask].astype(int))
    max_err = int(np.max(abs_err)) if len(abs_err) > 0 else 0

    assert max_err <= delta, f"Max error {max_err} exceeds bound delta={delta}"


def test_empty_zones_handling():
    """Verify proper handling when one or both zones are completely empty."""
    h, w = 60, 60
    img = np.zeros((h, w), dtype=np.uint8)

    # Completely empty zone_map
    empty_map = np.zeros((h, w), dtype=np.uint8)
    comp_bytes, recon = nl_encode(img, empty_map, {1: 2, 2: 4})
    dec = nl_decode(comp_bytes, empty_map)
    assert len(comp_bytes) > 0
    np.testing.assert_array_equal(dec, 0)

    # Only zone 1 present, zone 2 empty
    zone1_only = np.zeros((h, w), dtype=np.uint8)
    zone1_only[10:40, 10:40] = 1
    comp_bytes1, recon1 = nl_encode(img, zone1_only, {1: 2, 2: 4})
    dec1 = nl_decode(comp_bytes1, zone1_only)
    np.testing.assert_array_equal(dec1, recon1)


def test_delta_zero_gives_lossless_exact_pixels():
    """Verify that delta=0 gives exact lossless pixel values matching lossless_codec."""
    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Manifest not found")

    df = pd.read_csv(manifest_path)
    sample_stem = df.iloc[0]["stem"]
    img = load_image(PROCESSED_IMAGES_DIR / f"{sample_stem}.png")[:128, :128]

    # Create mask/zone
    mask = np.zeros(img.shape, dtype=bool)
    mask[20:100, 20:100] = True
    zone_map = mask.astype(np.uint8)

    # 1. lossless_codec
    data_ls = lossless_encode(img, mask=mask)
    recon_ls = lossless_decode(data_ls, mask=mask)

    # 2. nearlossless with delta=0
    data_nl, recon_nl = nl_encode(img, zone_map, deltas={1: 0, 2: 0})
    recon_nl_dec = nl_decode(data_nl, zone_map)

    # Check exact match with original image on mask
    np.testing.assert_array_equal(recon_nl[mask], img[mask])
    np.testing.assert_array_equal(recon_nl_dec[mask], img[mask])
    np.testing.assert_array_equal(recon_nl[mask], recon_ls[mask])


def test_sizes_shrink_as_delta_grows():
    """Verify that compressed size monotonically decreases as error tolerance delta increases."""
    np.random.seed(123)
    # Textured smooth image
    img = np.clip(np.sin(np.linspace(0, 10, 150))[:, None] * 100 + 128 + np.random.normal(0, 10, (150, 150)), 0, 255).astype(np.uint8)
    zone_map = np.ones((150, 150), dtype=np.uint8)

    sizes = []
    for d in [0, 1, 2, 4, 8]:
        data, _ = nl_encode(img, zone_map, deltas={1: d})
        sizes.append(len(data))

    for i in range(len(sizes) - 1):
        assert sizes[i] > sizes[i + 1], f"Size did not shrink from delta {[0, 1, 2, 4, 8][i]} ({sizes[i]}) to {[0, 1, 2, 4, 8][i+1]} ({sizes[i+1]})"
