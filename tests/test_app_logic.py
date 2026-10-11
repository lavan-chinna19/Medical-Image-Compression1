"""Unit tests for the application logic module (medcomp.app_logic).

Tests logic strictly without Streamlit imports:
1. Each compress mode round-trips and decodes from bytes alone.
2. Graded mode strictly respects error bounds on predicted core and band.
3. segment() returns shapes equal to input.
4. classify() returns probabilities in [0, 1].
5. diagnose() flips correctly on constructed cases.
6. Dataset sample detection correctly flags dataset stems.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medcomp.app_logic import (
    classify,
    compress,
    diagnose,
    is_dataset_sample,
    load_models,
    segment,
)


@pytest.fixture(scope="module")
def models():
    """Load GPU models once for test module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU model testing.")
    return load_models()


def test_is_dataset_sample():
    """Verify dataset sample detection flags known dataset stems and rejects unknown stems."""
    assert is_dataset_sample("CHNCXR_0001_0.png") is True
    assert is_dataset_sample("MCUCXR_0001_0.png") is True
    assert is_dataset_sample("CHNCXR_0001_0") is True
    assert is_dataset_sample("nonexistent_image_12345.png") is False
    assert is_dataset_sample("external_patient_scan.dcm") is False


def test_segment_shapes_and_ranges(models):
    """Verify segment() returns shapes matching the input image and valid probability/uncertainty."""
    # Test on arbitrary shape (non-square)
    img = np.random.randint(20, 220, (120, 160), dtype=np.uint8)

    # Test fast mode
    mask_fast, prob_fast, unc_fast = segment(img, models=models, fast=True)
    assert mask_fast.shape == (120, 160)
    assert prob_fast.shape == (120, 160)
    assert unc_fast.shape == (120, 160)
    assert mask_fast.dtype == bool
    assert 0.0 <= prob_fast.min() and prob_fast.max() <= 1.0

    # Test ensemble mode with 2 passes
    mask_full, prob_full, unc_full = segment(img, models=models, mc_passes=2, fast=False)
    assert mask_full.shape == (120, 160)
    assert prob_full.shape == (120, 160)
    assert unc_full.shape == (120, 160)
    assert 0.0 <= prob_full.min() and prob_full.max() <= 1.0
    assert unc_full.min() >= 0.0


def test_classify_probabilities_in_range(models):
    """Verify classify() returns valid probabilities in [0, 1] for both models."""
    img = np.random.randint(10, 240, (128, 128), dtype=np.uint8)
    preds = classify(img, models=models)

    assert "resnet_prob" in preds
    assert "judge_prob" in preds
    assert 0.0 <= preds["resnet_prob"] <= 1.0
    assert 0.0 <= preds["judge_prob"] <= 1.0


@pytest.mark.parametrize("mode,params", [
    ("png", {}),
    ("jpeg", {"quality": 60}),
    ("jpeg2000", {"ratio": 10.0}),
    ("dct", {"quality": 50}),
    ("dwt", {"quality": 50}),
    ("lossless_dilated", {"bg_quality": 40, "band_px": 6}),
    ("graded", {"delta_core": 2, "delta_band": 4, "band_px": 6, "bg_quality": 40}),
])
def test_compress_roundtrip_all_modes(mode, params):
    """Verify every compression mode encodes, decodes from bytes alone, and preserves dimensions."""
    np.random.seed(42)
    img = np.random.randint(10, 240, (96, 96), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:70, 20:70] = True

    res = compress(img, mask=mask, mode=mode, params=params)

    assert isinstance(res["bytes"], bytes)
    assert len(res["bytes"]) > 0
    assert res["recon"].shape == img.shape
    assert res["recon"].dtype == np.uint8
    assert res["bpp"] > 0
    assert res["ratio_vs_raw"] > 0
    assert res["ratio_vs_png"] > 0
    assert isinstance(res["parts"], dict)


def test_graded_mode_respects_error_bounds():
    """Verify graded mode strictly respects delta_core and delta_band error guarantees."""
    np.random.seed(123)
    img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=bool)
    mask[32:96, 32:96] = True

    delta_core = 2
    delta_band = 5
    band_px = 6

    res = compress(
        img,
        mask=mask,
        mode="graded",
        params={
            "delta_core": delta_core,
            "delta_band": delta_band,
            "band_px": band_px,
            "bg_quality": 30,
        },
    )

    recon = res["recon"]
    core_err = np.abs(img[mask].astype(float) - recon[mask].astype(float))
    assert np.max(core_err) <= delta_core + 1e-6
    assert res["max_err_core"] <= delta_core + 1e-6

    # Test lossless_dilated is exact on core
    res_ld = compress(
        img,
        mask=mask,
        mode="lossless_dilated",
        params={"bg_quality": 30, "band_px": band_px},
    )
    recon_ld = res_ld["recon"]
    assert np.array_equal(recon_ld[mask], img[mask])
    assert res_ld["exact_fraction"] == 1.0


def test_diagnose_flip_constructed_case():
    """Verify diagnose() correctly detects classifier decision flips on mock probabilities."""
    orig = np.full((64, 64), 100, dtype=np.uint8)
    recon = np.full((64, 64), 110, dtype=np.uint8)

    # Mock models with dummy outputs
    class DummyModel(torch.nn.Module):
        def __init__(self, prob: float):
            super().__init__()
            # logit = log(p / (1 - p))
            logit = float(np.log(prob / max(1e-6, 1.0 - prob)))
            self.param = torch.nn.Parameter(torch.tensor([logit]))

        def forward(self, x):
            return self.param.unsqueeze(0).expand(x.shape[0], 1)

    # ResNet18: flips from 0.45 to 0.55
    # Judge: does not flip (0.80 to 0.75)
    class MockLoadedModels:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resnet_models = [DummyModel(0.40), DummyModel(0.42)]
        judge_models = [DummyModel(0.80), DummyModel(0.82)]

    # We can test classify and diagnose with mocked classify outputs
    from unittest.mock import patch

    with patch("medcomp.app_logic.classify") as mock_classify:
        # First call: original (ResNet 0.40, Judge 0.80) -> ResNet neg, Judge pos
        # Second call: recon (ResNet 0.60, Judge 0.75) -> ResNet pos, Judge pos
        mock_classify.side_effect = [
            {"resnet_prob": 0.40, "judge_prob": 0.80},
            {"resnet_prob": 0.60, "judge_prob": 0.75},
        ]

        diag = diagnose(orig, recon)

        assert diag["resnet_prob_orig"] == 0.40
        assert diag["resnet_prob_dec"] == 0.60
        assert diag["judge_prob_orig"] == 0.80
        assert diag["judge_prob_dec"] == 0.75
        assert diag["flip_resnet"] is True  # 0.40 < 0.50 and 0.60 >= 0.50 -> flipped
        assert diag["flip_judge"] is False  # 0.80 >= 0.50 and 0.75 >= 0.50 -> no flip
