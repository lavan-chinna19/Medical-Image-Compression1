"""Integration tests for dataset manifest, cross-validation folds, and preprocessed outputs."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from medcomp.config import (
    DATA_DIR,
    DATA_ROOT,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
)

MANIFEST_PATH = DATA_DIR / "manifest.csv"


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset directory not found")
def test_manifest_structure_and_counts():
    """Verify manifest integrity, class distribution, and mask counts."""
    if not MANIFEST_PATH.is_file():
        pytest.skip("data/manifest.csv not generated yet")

    df = pd.read_csv(MANIFEST_PATH)

    # 1. Row count and uniqueness
    assert len(df) == 800, f"Expected 800 rows, got {len(df)}"
    assert df["stem"].nunique() == 800, "Duplicate stems detected in manifest"

    # 2. Source distribution
    mcu = df[df["source"] == "Montgomery"]
    chn = df[df["source"] == "Shenzhen"]
    assert len(mcu) == 138, f"Expected 138 Montgomery rows, got {len(mcu)}"
    assert len(chn) == 662, f"Expected 662 Shenzhen rows, got {len(chn)}"

    # 3. Mask count
    assert df["has_mask"].sum() == 704, f"Expected 704 masks, got {df['has_mask'].sum()}"
    assert mcu["has_mask"].sum() == 138
    assert chn["has_mask"].sum() == 566

    # 4. Class counts per source
    assert (mcu["label"] == 0).sum() == 80
    assert (mcu["label"] == 1).sum() == 58

    assert (chn["label"] == 0).sum() == 326
    assert (chn["label"] == 1).sum() == 336


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset directory not found")
def test_folds_distribution_and_splits():
    """Verify 5-fold stratification and cross-scanner splits."""
    if not MANIFEST_PATH.is_file():
        pytest.skip("data/manifest.csv not generated yet")

    df = pd.read_csv(MANIFEST_PATH)
    if "fold" not in df.columns:
        pytest.skip("Folds not yet created in manifest")

    # 1. No stem in more than one fold
    assert df["stem"].nunique() == len(df)

    # 2. Fold sizes differ by at most 1
    fold_counts = df["fold"].value_counts()
    assert len(fold_counts) == 5, f"Expected 5 folds, got {len(fold_counts)}"
    assert fold_counts.max() - fold_counts.min() <= 1

    # 3. Each fold has both classes and both sources
    for f_idx in range(5):
        f_df = df[df["fold"] == f_idx]
        assert set(f_df["source"].unique()) == {"Montgomery", "Shenzhen"}
        assert set(f_df["label"].unique()) == {0, 1}

    # 4. Cross-scanner split
    assert (df[df["source"] == "Shenzhen"]["split_xscanner"] == "train").all()
    assert (df[df["source"] == "Montgomery"]["split_xscanner"] == "test").all()


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Dataset directory not found")
def test_processed_images_and_masks_integrity():
    """Verify preprocessed dataset counts, matching shapes, and strict binary masks."""
    if not PROCESSED_IMAGES_DIR.exists() or not PROCESSED_MASKS_DIR.exists():
        pytest.skip("work/processed directory not generated yet")

    img_files = list(PROCESSED_IMAGES_DIR.glob("*.png"))
    mask_files = list(PROCESSED_MASKS_DIR.glob("*.png"))

    assert len(img_files) == 800, f"Expected 800 processed images, got {len(img_files)}"
    assert len(mask_files) == 704, f"Expected 704 processed masks, got {len(mask_files)}"

    # Check mask shapes match corresponding image shapes and contain only {0, 255}
    for mf in mask_files:
        img_p = PROCESSED_IMAGES_DIR / mf.name
        assert img_p.is_file(), f"Missing corresponding image for mask {mf.name}"

        m_arr = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        i_arr = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)

        assert m_arr.shape == i_arr.shape, (
            f"Shape mismatch for {mf.name}: mask {m_arr.shape} vs image {i_arr.shape}"
        )
        unique_vals = set(np.unique(m_arr))
        assert unique_vals.issubset({0, 255}), (
            f"Mask {mf.name} has non-binary values: {unique_vals}"
        )
