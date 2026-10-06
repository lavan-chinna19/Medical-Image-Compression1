"""Unit tests for dataset parsing, mask operations, and resizing functions."""

import math
from pathlib import Path
import numpy as np
import pytest

from medcomp.config import DATA_ROOT
from medcomp.data_utils import (
    bounding_box,
    load_mask,
    mask_path_for,
    parse_reading,
    resize_long_side,
    resize_mask_nearest,
    roi_fraction,
    source_of,
)


def test_source_of():
    """Verify source classification based on prefix."""
    assert source_of("MCUCXR_0001_0") == "Montgomery"
    assert source_of("CHNCXR_0001_0") == "Shenzhen"
    with pytest.raises(ValueError):
        source_of("UNKNOWN_0001_0")


def test_parse_reading_montgomery(tmp_path: Path):
    """Verify standard Montgomery reading parsing."""
    p = tmp_path / "MCUCXR_0001_0.txt"
    p.write_text("Patient's Sex: F\nPatient's Age: 027Y\nnormal\n", encoding="utf-8")

    res = parse_reading(p)
    assert res["sex"] == "F"
    assert res["age_years"] == pytest.approx(27.0)
    assert res["findings_text"] == "normal"


def test_parse_reading_shenzhen(tmp_path: Path):
    """Verify standard Shenzhen reading parsing."""
    p = tmp_path / "CHNCXR_0001_0.txt"
    p.write_text("male 45yrs\nnormal\n", encoding="utf-8")

    res = parse_reading(p)
    assert res["sex"] == "M"
    assert res["age_years"] == pytest.approx(45.0)
    assert res["findings_text"] == "normal"


def test_parse_reading_irregular_units(tmp_path: Path):
    """Verify irregular formats: month, days, and typos like 'femal'."""
    p1 = tmp_path / "CHNCXR_0115_0.txt"
    p1.write_text("male 16month\nnormal\n", encoding="utf-8")
    res1 = parse_reading(p1)
    assert res1["sex"] == "M"
    assert res1["age_years"] == pytest.approx(16.0 / 12.0)

    p2 = tmp_path / "CHNCXR_0202_0.txt"
    p2.write_text("female 64days\nnormal\n", encoding="utf-8")
    res2 = parse_reading(p2)
    assert res2["sex"] == "F"
    assert res2["age_years"] == pytest.approx(64.0 / 365.25, rel=1e-3)

    p3 = tmp_path / "CHNCXR_0044_0.txt"
    p3.write_text("femal 32yrs\nnormal\n", encoding="utf-8")
    res3 = parse_reading(p3)
    assert res3["sex"] == "F"
    assert res3["age_years"] == pytest.approx(32.0)


def test_parse_reading_malformed_and_missing(tmp_path: Path):
    """Verify unparseable or missing files do not crash and return NaNs."""
    p_empty = tmp_path / "empty.txt"
    p_empty.write_text("", encoding="utf-8")
    res_empty = parse_reading(p_empty)
    assert res_empty["sex"] is None
    assert math.isnan(res_empty["age_years"])
    assert res_empty["findings_text"] == ""

    p_garbage = tmp_path / "garbage.txt"
    p_garbage.write_text("??? /// random non medical bytes 12345", encoding="utf-8")
    res_garbage = parse_reading(p_garbage)
    assert isinstance(res_garbage, dict)

    res_missing = parse_reading(tmp_path / "non_existent.txt")
    assert res_missing["sex"] is None
    assert math.isnan(res_missing["age_years"])


def test_mask_path_for_synthetic(tmp_path: Path):
    """Verify mask_path_for resolution for both Montgomery and Shenzhen naming rules."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    (masks_dir / "MCUCXR_0001_0.png").touch()
    (masks_dir / "CHNCXR_0001_0_mask.png").touch()

    # Montgomery rule
    assert mask_path_for("MCUCXR_0001_0", tmp_path) == (masks_dir / "MCUCXR_0001_0.png")
    # Shenzhen rule
    assert mask_path_for("CHNCXR_0001_0", tmp_path) == (masks_dir / "CHNCXR_0001_0_mask.png")
    # Non-existent
    assert mask_path_for("CHNCXR_9999_0", tmp_path) is None


def test_resize_long_side():
    """Verify resize_long_side keeps aspect ratio and scales long side to target."""
    # Landscape image: 400 x 800
    img_land = np.zeros((400, 800), dtype=np.uint8)
    res_land = resize_long_side(img_land, target=512)
    assert res_land.shape == (256, 512)

    # Portrait image: 1000 x 500
    img_port = np.zeros((1000, 500), dtype=np.uint8)
    res_port = resize_long_side(img_port, target=512)
    assert res_port.shape == (512, 256)

    # Square image: 300 x 300
    img_sq = np.zeros((300, 300), dtype=np.uint8)
    res_sq = resize_long_side(img_sq, target=512)
    assert res_sq.shape == (512, 512)


def test_resize_mask_nearest():
    """Verify nearest-neighbor mask resizing preserves strict binary values."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True

    resized = resize_mask_nearest(mask, (100, 120))
    assert resized.shape == (100, 120)
    assert resized.dtype == bool
    assert np.any(resized)
    assert not np.all(resized)


def test_roi_fraction_and_bounding_box():
    """Verify roi_fraction and bounding_box on known synthetic masks."""
    # 100 x 100 canvas with a 20 x 40 box = 800 pixels -> fraction 0.08
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:70] = True

    assert roi_fraction(mask) == pytest.approx(0.08)
    assert bounding_box(mask) == (20, 30, 39, 69)

    # Empty mask
    empty = np.zeros((50, 50), dtype=bool)
    assert roi_fraction(empty) == 0.0
    assert bounding_box(empty) == (0, 0, 0, 0)
