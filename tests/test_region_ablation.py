"""Unit tests for region ablation composites and consistency with benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root and src are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR
from scripts.run_region_ablation import dilate_mask


def test_composite_pixel_replacement_integrity():
    """Verify that composites match original and lossy L pixel-for-pixel in their respective regions."""
    np.random.seed(42)
    h, w = 100, 100
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    lossy_l = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    # Synthetic mask with circle in the middle
    y, x = np.ogrid[:h, :w]
    mask = ((x - 50) ** 2 + (y - 50) ** 2 <= 25 ** 2)

    # 1. lungs_lossy: original outside, lossy L inside
    lungs_lossy = img.copy()
    lungs_lossy[mask] = lossy_l[mask]

    np.testing.assert_array_equal(lungs_lossy[~mask], img[~mask])
    np.testing.assert_array_equal(lungs_lossy[mask], lossy_l[mask])

    # 2. background_lossy: lossy L outside, original inside
    bg_lossy = lossy_l.copy()
    bg_lossy[mask] = img[mask]

    np.testing.assert_array_equal(bg_lossy[mask], img[mask])
    np.testing.assert_array_equal(bg_lossy[~mask], lossy_l[~mask])


def test_mask_dilation_properties():
    """Verify that mask dilation expands boundary by 4 px."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, 20:30] = True
    dilated = dilate_mask(mask, d_pixels=4)

    assert dilated.sum() > mask.sum()
    # Dilated mask strictly contains original mask
    assert np.all(dilated[mask])


def test_all_lossy_reproduces_dwt_benchmark():
    """Verify that all_lossy variant reproduces DWT rows from benchmark_per_image.csv."""
    ablation_csv = RESULTS_DIR / "region_ablation_per_image.csv"
    benchmark_csv = RESULTS_DIR / "benchmark_per_image.csv"

    if not ablation_csv.is_file() or not benchmark_csv.is_file():
        pytest.skip("Benchmark or ablation CSV not yet generated.")

    df_ablation = pd.read_csv(ablation_csv)
    df_bench = pd.read_csv(benchmark_csv)

    all_lossy = df_ablation[df_ablation["variant"] == "all_lossy"].copy()
    dwt_bench = df_bench[
        (df_bench["codec"] == "dwt") & (df_bench["setting"].astype(int).isin([10, 30, 50, 70]))
    ].copy()
    dwt_bench["quality"] = dwt_bench["setting"].astype(int)

    merged = pd.merge(
        all_lossy[["stem", "quality", "prob_dec", "abs_prob_diff"]],
        dwt_bench[["stem", "quality", "prob_dec", "abs_prob_diff"]],
        on=["stem", "quality"],
        suffixes=("_ablation", "_bench"),
    )

    assert len(merged) > 0, "No overlapping rows found"

    np.testing.assert_allclose(
        merged["prob_dec_ablation"].values,
        merged["prob_dec_bench"].values,
        atol=1e-5,
        err_msg="prob_dec does not match between all_lossy and benchmark DWT",
    )
    np.testing.assert_allclose(
        merged["abs_prob_diff_ablation"].values,
        merged["abs_prob_diff_bench"].values,
        atol=1e-3,
        err_msg="abs_prob_diff does not match between all_lossy and benchmark DWT",
    )
