"""Unit and integration tests for U-Net architecture, dataset, and metrics."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR
from medcomp.seg_data import LungSegDataset, inverse_transform, pad_to_square
from medcomp.unet import UNet, bce_dice_loss, dice_score, enable_mc_dropout, iou_score


def test_unet_forward_shape():
    """Verify forward pass output shape is (B, 1, 256, 256)."""
    model = UNet(in_channels=1, out_channels=1, base_channels=32)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_dice_and_iou_known_masks():
    """Verify dice_score and iou_score on known analytical cases."""
    # 1. Identical masks: Dice = 1.0, IoU = 1.0
    m = torch.ones(1, 1, 32, 32)
    assert abs(dice_score(m, m).item() - 1.0) < 1e-4
    assert abs(iou_score(m, m).item() - 1.0) < 1e-4

    # 2. Completely disjoint masks: Dice = 0.0, IoU = 0.0
    m1 = torch.zeros(1, 1, 32, 32)
    m1[:, :, :16, :] = 1.0
    m2 = torch.zeros(1, 1, 32, 32)
    m2[:, :, 16:, :] = 1.0
    assert abs(dice_score(m1, m2).item() - 0.0) < 1e-4
    assert abs(iou_score(m1, m2).item() - 0.0) < 1e-4

    # 3. 50% overlap: intersection=16*32, card=2*16*32+16*32=3*16*32
    # Dice = 2*1/3 = 2/3 = 0.6667, IoU = 1/2 = 0.50
    m3 = torch.zeros(1, 1, 32, 32)
    m3[:, :, :24, :] = 1.0
    # Overlap between m1 and m3 is 16*32
    d = dice_score(m1, m3).item()
    iou = iou_score(m1, m3).item()
    assert abs(d - (2.0 * 16.0 / (16.0 + 24.0))) < 1e-3
    assert abs(iou - (16.0 / 24.0)) < 1e-3


def test_mc_dropout_leaves_bn_in_eval_mode():
    """Verify enable_mc_dropout keeps Dropout layers in train mode while BatchNorm stays in eval mode."""
    model = UNet(in_channels=1, out_channels=1, base_channels=32)
    model.eval()

    # Before enable_mc_dropout: all modules are in eval mode
    for m in model.modules():
        assert not m.training

    enable_mc_dropout(model)

    # After enable_mc_dropout: Dropout/Dropout2d are training, BatchNorm is still eval
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            assert m.training, "Dropout should be in train mode"
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            assert not m.training, "BatchNorm should remain in eval mode"


def test_dataset_item_alignment_and_inverse_transform():
    """Verify dataset returns aligned tensors and inverse_transform restores exact shape."""
    manifest_csv = DATA_DIR / "manifest.csv"
    if not manifest_csv.is_file():
        pytest.skip("Manifest not found.")

    df = pd.read_csv(manifest_csv)
    masked_df = df[df["has_mask"] == True]
    if masked_df.empty:
        pytest.skip("No masked images.")

    ds = LungSegDataset(masked_df.head(2), is_train=False)
    img_t, mask_t, stem = ds[0]

    assert img_t.shape == (1, 256, 256)
    assert mask_t.shape == (1, 256, 256)
    assert 0.0 <= img_t.min() and img_t.max() <= 1.0
    assert set(np.unique(mask_t.numpy())).issubset({0.0, 1.0})

    # Test inverse transform with arbitrary non-square processed size
    orig_h, orig_w = 412, 512
    dummy_prob = torch.rand(256, 256)
    restored = inverse_transform(dummy_prob, orig_h, orig_w)
    assert restored.shape == (orig_h, orig_w)
    assert 0.0 <= restored.min() and restored.max() <= 1.0


def test_fold_leakage_assertion():
    """Verify images held out in fold k never appear in the training split for fold k."""
    manifest_csv = DATA_DIR / "manifest.csv"
    if not manifest_csv.is_file():
        pytest.skip("Manifest not found.")

    df = pd.read_csv(manifest_csv)
    masked_df = df[df["has_mask"] == True]

    for k in range(5):
        val_stems = set(masked_df[masked_df["fold"] == k]["stem"])
        train_stems = set(masked_df[masked_df["fold"] != k]["stem"])
        assert len(val_stems.intersection(train_stems)) == 0, f"Fold {k} leakage detected!"
