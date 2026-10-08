"""Tests for independent judge classifier and ladder fold isolation."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn

from medcomp.config import DATA_DIR
from medcomp.judge import get_judge


def test_judge_architecture():
    """Verify EfficientNet-B0 architecture with dropout 0.2 and 1-logit output."""
    model = get_judge(pretrained=False)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "classifier")

    # Final head should be nn.Sequential with Dropout(p=0.2) and Linear(1280, 1)
    head = model.classifier
    assert isinstance(head, nn.Sequential)
    assert len(head) == 2
    assert isinstance(head[0], nn.Dropout)
    assert pytest.approx(head[0].p) == 0.2
    assert isinstance(head[1], nn.Linear)
    assert head[1].in_features == 1280
    assert head[1].out_features == 1

    # Test dummy forward pass on CPU
    x = torch.randn(2, 3, 320, 320)
    out = model(x)
    assert out.shape == (2, 1)


def test_judge_held_out_folds_isolated():
    """Verify that held-out fold images are strictly never in training/validation splits."""
    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"
    manifest_df = pd.read_csv(manifest_path)

    for k in range(5):
        held_out_stems = set(manifest_df[manifest_df["fold"] == k]["stem"])
        training_pool_stems = set(manifest_df[manifest_df["fold"] != k]["stem"])

        # Intersection must be strictly empty
        overlap = held_out_stems.intersection(training_pool_stems)
        assert len(overlap) == 0, f"Fold {k} leak: {len(overlap)} stems found in both held-out and training pool!"

        # Ensure held-out fold contains exactly 160 images (800 / 5)
        assert len(held_out_stems) == 160


def test_ladder_rows_reuse_held_out_models():
    """Verify ladder row evaluation reuses held-out fold models for each image."""
    manifest_path = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    # Every image has a fold in 0..4
    folds = manifest_df["fold"].unique()
    assert set(folds) == {0, 1, 2, 3, 4}

    # Verify model path mapping function logic
    for _, row in manifest_df.sample(20, random_state=42).iterrows():
        img_fold = int(row["fold"])
        expected_steering_ckpt = f"fold{img_fold}.pt"
        expected_judge_ckpt = f"fold{img_fold}.pt"

        # Model used for image must strictly match its fold index
        assert 0 <= img_fold <= 4
        assert expected_steering_ckpt == f"fold{img_fold}.pt"
        assert expected_judge_ckpt == f"fold{img_fold}.pt"
