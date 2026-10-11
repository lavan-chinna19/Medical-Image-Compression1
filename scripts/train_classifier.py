"""5-fold cross-validation training and cross-scanner experiments for TB classifier.

Uses ResNet18 with ImageNet weights, BCEWithLogitsLoss, AdamW, Cosine Annealing,
and automatic mixed precision (AMP) on CUDA.
Early stopping monitors validation AUC with patience 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import (
    CLASSIFIER_DIR,
    bootstrap_ci_metrics,
    compute_binary_metrics,
    get_classifier,
    predict_proba,
)
from medcomp.cls_data import TBClassificationDataset
from medcomp.config import DATA_DIR, RESULTS_DIR, WORK_DIR


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders with automatic Windows fallback."""
    train_ds = TBClassificationDataset(train_df, is_train=True, seed=seed)
    val_ds = TBClassificationDataset(val_df, is_train=False, seed=seed)

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
        # Verify worker creation
        _ = next(iter(val_loader))
    except Exception as e:
        print(f"DataLoader with num_workers={num_workers} encountered: {e}. Falling back to num_workers=0.")
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
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    """Train classifier for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_val_set(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate validation loss and validation AUC."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[float] = []
    all_targets: list[float] = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).squeeze(-1)
        all_preds.extend(probs.cpu().tolist())
        all_targets.extend(labels.squeeze(-1).cpu().tolist())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_preds, dtype=float)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = 0.5

    return avg_loss, auc


def train_fold(
    fold: int,
    manifest_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    device: torch.device,
    save_path: Path,
    history_path: Path,
) -> list[dict[str, Any]]:
    """Train single fold with 10% stratified validation split for early stopping."""
    print(f"\n{'='*30} Fold {fold} {'='*30}")
    # Held-out fold k is never used in training or validation
    train_pool = manifest_df[manifest_df["fold"] != fold].copy()

    # Stratified 10% validation split by (source, label)
    strat_key = train_pool["source"] + "_" + train_pool["label"].astype(str)
    train_df, val_df = train_test_split(
        train_pool,
        test_size=0.10,
        stratify=strat_key,
        random_state=42 + fold,
    )
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Held-out test samples: {len(manifest_df[manifest_df['fold'] == fold])}")

    train_loader, val_loader = create_dataloaders(
        train_df, val_df, batch_size=batch_size, num_workers=num_workers, seed=42 + fold
    )

    model = get_classifier(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_auc = -1.0
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc = evaluate_val_set(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": current_lr,
            "time_sec": elapsed,
        })

        print(
            f"Epoch {ep:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | lr: {current_lr:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path.name} (val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep} (patience={patience}, best AUC={best_val_auc:.4f})")
                break

    # Save training history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history to {history_path}")

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    return history


def run_cross_scanner_experiments(
    manifest_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    device: torch.device,
) -> None:
    """Train Shenzhen-only and Montgomery-only models and evaluate cross-domain generalization."""
    print("\n" + "=" * 70)
    print("RUNNING CROSS-SCANNER GENERALIZATION EXPERIMENTS")
    print("=" * 70)

    configs = [
        ("Shenzhen", "Montgomery", "cross_shenzhen"),
        ("Montgomery", "Shenzhen", "cross_montgomery"),
    ]

    cross_results: list[dict[str, Any]] = []

    for src_train, src_test, exp_id in configs:
        print(f"\n--- Experiment: Train on {src_train} -> Test on {src_test} ---")
        train_pool = manifest_df[manifest_df["source"] == src_train].copy()
        test_df = manifest_df[manifest_df["source"] == src_test].copy()

        # 10% stratified val split
        train_df, val_df = train_test_split(
            train_pool,
            test_size=0.10,
            stratify=train_pool["label"],
            random_state=42,
        )
        print(f"Train on {src_train}: {len(train_df)}, Val: {len(val_df)}, Test on {src_test}: {len(test_df)}")

        train_loader, val_loader = create_dataloaders(
            train_df, val_df, batch_size=batch_size, num_workers=num_workers, seed=42
        )

        ckpt_path = CLASSIFIER_DIR / f"{exp_id}.pt"
        history_path = RESULTS_DIR / f"classifier_training_{exp_id}.csv"

        if ckpt_path.is_file():
            print(f"Checkpoint {ckpt_path.name} already exists. Loading directly.")
        else:
            model = get_classifier(pretrained=True).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            scaler = GradScaler("cuda")

            best_val_auc = -1.0
            patience_counter = 0
            history = []

            for ep in range(1, epochs + 1):
                t0 = time.time()
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
                val_loss, val_auc = evaluate_val_set(model, val_loader, criterion, device)
                scheduler.step()
                elapsed = time.time() - t0

                history.append({
                    "epoch": ep,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time_sec": elapsed,
                })

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopped at epoch {ep}")
                        break

            pd.DataFrame(history).to_csv(history_path, index=False)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Load best model for evaluation on test set
        test_model = get_classifier(pretrained=False)
        test_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_model.to(device)
        test_model.eval()

        test_ds = TBClassificationDataset(test_df, is_train=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        preds: list[float] = []
        targets: list[int] = []
        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                imgs = imgs.to(device)
                with autocast("cuda"):
                    logits = test_model(imgs)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                preds.extend(np.atleast_1d(probs).tolist())
                targets.extend(lbls.squeeze(-1).int().cpu().tolist())

        del test_model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

        # Compute metrics with 95% bootstrap CI
        y_true = np.array(targets, dtype=int)
        y_prob = np.array(preds, dtype=float)
        ci_dict = bootstrap_ci_metrics(y_true, y_prob, threshold=0.5, n_bootstraps=1000, seed=42)

        row: dict[str, Any] = {
            "setting": f"Cross-domain: {src_train} -> {src_test}",
            "train_source": src_train,
            "test_source": src_test,
            "n_samples": len(y_true),
            "tn": int(np.sum((y_true == 0) & (y_prob < 0.5))),
            "fp": int(np.sum((y_true == 0) & (y_prob >= 0.5))),
            "fn": int(np.sum((y_true == 1) & (y_prob < 0.5))),
            "tp": int(np.sum((y_true == 1) & (y_prob >= 0.5))),
        }
        for metric, vals in ci_dict.items():
            row[metric] = vals["point"]
            row[f"{metric}_ci_lower"] = vals["ci_lower"]
            row[f"{metric}_ci_upper"] = vals["ci_upper"]

        cross_results.append(row)

    # Also load in-domain metrics from classifier_metrics.csv to display next to cross-scanner numbers
    in_domain_path = RESULTS_DIR / "classifier_metrics.csv"
    if in_domain_path.is_file():
        id_df = pd.read_csv(in_domain_path)
        for _, r in id_df.iterrows():
            grp = str(r["group"])
            if "Shenzhen" in grp:
                in_row = {
                    "setting": "In-domain (Shenzhen 5-Fold OOF)",
                    "train_source": "Shenzhen",
                    "test_source": "Shenzhen",
                    "n_samples": int(r["n_samples"]),
                    "tn": int(r["tn"]),
                    "fp": int(r["fp"]),
                    "fn": int(r["fn"]),
                    "tp": int(r["tp"]),
                }
                for m in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]:
                    in_row[m] = float(r[m])
                    in_row[f"{m}_ci_lower"] = float(r[f"{m}_ci_lower"])
                    in_row[f"{m}_ci_upper"] = float(r[f"{m}_ci_upper"])
                cross_results.append(in_row)
            elif "Montgomery" in grp:
                in_row = {
                    "setting": "In-domain (Montgomery 5-Fold OOF)",
                    "train_source": "Montgomery",
                    "test_source": "Montgomery",
                    "n_samples": int(r["n_samples"]),
                    "tn": int(r["tn"]),
                    "fp": int(r["fp"]),
                    "fn": int(r["fn"]),
                    "tp": int(r["tp"]),
                }
                for m in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]:
                    in_row[m] = float(r[m])
                    in_row[f"{m}_ci_lower"] = float(r[f"{m}_ci_lower"])
                    in_row[f"{m}_ci_upper"] = float(r[f"{m}_ci_upper"])
                cross_results.append(in_row)

    # Save cross-scanner results
    out_csv = RESULTS_DIR / "classifier_crossscanner.csv"
    res_df = pd.DataFrame(cross_results)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved cross-scanner metrics with in-domain comparison to {out_csv}")
    print(res_df[["setting", "train_source", "test_source", "accuracy", "recall", "specificity", "auc"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TB ResNet18 classifier across 5 folds.")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch budget per fold (default: 15).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val AUC (default: 5).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers (default: 2).")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch of fold 0 to measure timing.")
    parser.add_argument("--cross-scanner", action="store_true", help="Run cross-scanner generalization experiments.")
    parser.add_argument("--force", action="store_true", help="Force retrain existing checkpoints.")
    args = parser.parse_args()

    # Hardware check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Training on GPU is mandatory for RTX 2050.")

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)} with mixed precision.")

    set_seed(42)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    if args.cross_scanner:
        run_cross_scanner_experiments(
            manifest_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
        )
        return

    if args.smoke:
        print("\n--- Running 1-epoch smoke test on fold 0 ---")
        t0 = time.time()
        _ = train_fold(
            fold=0,
            manifest_df=manifest_df,
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
            save_path=CLASSIFIER_DIR / "smoke_fold0.pt",
            history_path=RESULTS_DIR / "classifier_smoke_fold0.csv",
        )
        total_time = time.time() - t0
        print(f"\n[Smoke Test Complete] 1 epoch of fold 0 took: {total_time:.2f} seconds.")
        return

    # Full 5-fold cross-validation
    total_start = time.time()
    for k in range(5):
        ckpt_path = CLASSIFIER_DIR / f"fold{k}.pt"
        history_path = RESULTS_DIR / f"classifier_training_fold{k}.csv"

        if ckpt_path.is_file() and not args.force:
            print(f"Skipping fold {k}: checkpoint {ckpt_path.name} already exists.")
            continue

        train_fold(
            fold=k,
            manifest_df=manifest_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
            save_path=ckpt_path,
            history_path=history_path,
        )

    print(f"\n5-fold training finished in {(time.time() - total_start)/60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
