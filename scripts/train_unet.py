"""5-fold cross-validation training of U-Net lung segmentation with AMP and early stopping."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, RESULTS_DIR, WORK_DIR
from medcomp.seg_data import LungSegDataset
from medcomp.unet import UNet, bce_dice_loss, dice_score

UNET_DIR = WORK_DIR / "unet"


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders with automatic Windows fallback."""
    train_ds = LungSegDataset(train_df, is_train=True, seed=seed)
    val_ds = LungSegDataset(val_df, is_train=False, seed=seed)

    try:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        # Test 1 batch fetch to verify worker processes
        _ = next(iter(val_loader))
    except Exception as e:
        print(f"DataLoader with num_workers={num_workers} failed: {e}. Falling back to num_workers=0.")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """Train model for one epoch using automatic mixed precision (AMP)."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits)
        d_val = dice_score(probs, masks).item()

        total_loss += loss.item()
        total_dice += d_val
        n_batches += 1

    return float(total_loss / max(1, n_batches)), float(total_dice / max(1, n_batches))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model over evaluation dataset."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)

        probs = torch.sigmoid(logits)
        d_val = dice_score(probs, masks).item()

        total_loss += loss.item()
        total_dice += d_val
        n_batches += 1

    return float(total_loss / max(1, n_batches)), float(total_dice / max(1, n_batches))


def train_fold(
    fold_idx: int,
    manifest_df: pd.DataFrame,
    epochs: int = 35,
    batch_size: int = 8,
    num_workers: int = 2,
    patience: int = 10,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Train single fold cross-validation model."""
    device = torch.device(device_name)
    print(f"\n==================== Starting Training for Fold {fold_idx} ====================")

    # Fold split: train on all folds != fold_idx
    pool_df = manifest_df[manifest_df["fold"] != fold_idx].copy()
    test_df = manifest_df[manifest_df["fold"] == fold_idx].copy()

    # Stratified 10% validation hold-out from training pool
    pool_df["strat_group"] = pool_df["source"].astype(str) + "_" + pool_df["label"].astype(str)
    val_indices = []
    rng = np.random.default_rng(42 + fold_idx)
    for _, grp in pool_df.groupby("strat_group"):
        n_val = max(1, int(round(0.10 * len(grp))))
        chosen = rng.choice(grp.index.values, size=min(n_val, len(grp)), replace=False)
        val_indices.extend(chosen)

    val_df = pool_df.loc[val_indices].copy()
    train_df = pool_df.drop(index=val_indices).copy()

    print(f"Fold {fold_idx} dataset split: Train={len(train_df)}, EarlyStopVal={len(val_df)}, TestHoldout={len(test_df)}")

    # Model and optimizer
    set_seed(42 + fold_idx)
    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, batch_size=batch_size, num_workers=num_workers, seed=42 + fold_idx
    )

    best_val_dice = -1.0
    patience_counter = 0
    checkpoint_path = UNET_DIR / f"fold{fold_idx}.pt"
    curve_csv = RESULTS_DIR / f"unet_training_fold{fold_idx}.csv"

    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_dice = validate(model, val_loader, device)
        scheduler.step()
        ep_time = time.perf_counter() - t0

        cur_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_dice": round(train_dice, 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_dice, 4),
            "lr": round(cur_lr, 6),
            "time_s": round(ep_time, 2),
        })

        improved = val_dice > best_val_dice
        mark = " (*)" if improved else ""
        print(f"Epoch {ep:2d}/{epochs:2d} | Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}{mark} | Time: {ep_time:.1f}s")

        if improved:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "fold": fold_idx,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep} (patience={patience})")
                break

    # Save training curves
    pd.DataFrame(history).to_csv(curve_csv, index=False)
    print(f"Saved best checkpoint to {checkpoint_path} (Val Dice: {best_val_dice:.4f})")
    print(f"Saved training curve to {curve_csv}")

    # Free memory
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()

    return {"fold": fold_idx, "best_val_dice": best_val_dice, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train 5-fold U-Net cross-validation.")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch budget per fold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--fold", type=int, default=None, help="Train only specific fold (0..4)")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch on fold 0 for timing estimation")
    parser.add_argument("--force", action="store_true", help="Force retraining even if checkpoint exists")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is NOT available! Cannot proceed without GPU.")

    device_name = torch.cuda.get_device_name(0)
    print(f"Hardware Verified: Running on GPU '{device_name}' with PyTorch AMP")

    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    masked_df = df[df["has_mask"] == True].copy()

    UNET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("\n--- Running 1-Epoch Smoke Test on Fold 0 ---")
        t0 = time.perf_counter()
        res = train_fold(
            fold_idx=0,
            manifest_df=masked_df,
            epochs=1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patience=1,
            device_name="cuda:0",
        )
        total_smoke = time.perf_counter() - t0
        ep_time = res["history"][0]["time_s"]
        print(f"\n[Smoke Test Result] Time for 1 epoch: {ep_time:.2f}s")
        for budget in [20, 25, 30, 35]:
            proj_min = (budget * 5 * ep_time) / 60.0
            print(f"Projected total time for 5 folds @ {budget} epochs: {proj_min:.1f} minutes ({proj_min / 60.0:.2f} hours)")
        return

    target_folds = [args.fold] if args.fold is not None else list(range(5))

    for k in target_folds:
        ckpt_path = UNET_DIR / f"fold{k}.pt"
        if ckpt_path.is_file() and not args.force:
            print(f"Skipping fold {k}: checkpoint {ckpt_path} already exists.")
            continue

        train_fold(
            fold_idx=k,
            manifest_df=masked_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patience=args.patience,
            device_name="cuda:0",
        )


if __name__ == "__main__":
    main()
