"""5-fold cross-validation training and out-of-fold evaluation for independent Judge classifier.

Architecture:
- torchvision EfficientNet-B0 with ImageNet pretrained weights.
- Classifier head: Dropout(p=0.2) + Linear(1280, 1).
- Resolution: 320x320 square padding.
- Different seed from steering classifier (default: 1337).
- Same 5 folds as manifest.csv.
- Same training recipe as train_classifier.py:
  10% stratified validation split for early stopping (patience 5 on val AUC),
  AdamW, Cosine Annealing, mixed precision (AMP on CUDA), NO horizontal flips.

Outputs:
- work/judge/fold{k}.pt
- results/judge_oof.csv
- results/judge_metrics.csv
- results/judge_summary.txt
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

from medcomp.classifier import bootstrap_ci_metrics, compute_binary_metrics
from medcomp.cls_data import TBClassificationDataset
from medcomp.config import DATA_DIR, RESULTS_DIR, WORK_DIR
from medcomp.judge import JUDGE_DIR, get_judge, load_judge_model, predict_proba_judge


def set_seed(seed: int = 1337) -> None:
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
    seed: int = 1337,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation with Windows fallback."""
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
    """Train judge model for one epoch using automatic mixed precision."""
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
    """Evaluate validation loss and validation AUC on the validation split."""
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


def train_judge_fold(
    fold: int,
    manifest_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    save_path: Path,
) -> list[dict[str, Any]]:
    """Train single fold of judge classifier with 10% stratified validation split."""
    print(f"\n{'='*30} Training Judge Fold {fold} {'='*30}")
    # STRICT ISOLATION: Held-out fold k is NEVER used in training or validation
    held_out_mask = (manifest_df["fold"] == fold)
    train_pool = manifest_df[~held_out_mask].copy()

    # Verify 0 overlap with held-out fold
    assert len(set(train_pool["stem"]).intersection(set(manifest_df[held_out_mask]["stem"]))) == 0

    strat_key = train_pool["source"] + "_" + train_pool["label"].astype(str)
    train_df, val_df = train_test_split(
        train_pool,
        test_size=0.10,
        stratify=strat_key,
        random_state=seed + fold,
    )
    print(
        f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, "
        f"Strictly held-out test samples: {len(manifest_df[held_out_mask])}"
    )

    train_loader, val_loader = create_dataloaders(
        train_df, val_df, batch_size=batch_size, num_workers=num_workers, seed=seed + fold
    )

    model = get_judge(pretrained=True).to(device)
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
            print(f"  -> Best judge model saved to {save_path.name} (val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep} (patience={patience}, best AUC={best_val_auc:.4f})")
                break

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    return history


def compute_oof_predictions(
    manifest_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Compute out-of-fold predictions on original images using trained judge checkpoints."""
    print("\nComputing out-of-fold predictions for all 800 images using Judge models...")
    oof_rows: list[dict[str, Any]] = []

    for k in range(5):
        fold_df = manifest_df[manifest_df["fold"] == k].copy().sort_values("stem").reset_index(drop=True)
        ckpt_path = JUDGE_DIR / f"fold{k}.pt"
        assert ckpt_path.is_file(), f"Judge checkpoint for fold {k} not found at {ckpt_path}"

        model = load_judge_model(k, device=device)
        ds = TBClassificationDataset(fold_df, is_train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        probs: list[float] = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                with autocast("cuda"):
                    logits = model(imgs)
                batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                probs.extend(np.atleast_1d(batch_probs).tolist())

        del model, ds, loader
        torch.cuda.empty_cache()

        for idx, row in fold_df.iterrows():
            prob_val = float(probs[idx])
            oof_rows.append({
                "stem": str(row["stem"]),
                "source": str(row["source"]),
                "label": int(row["label"]),
                "fold": k,
                "prob_tb": prob_val,
                "pred": int(prob_val >= 0.5),
            })

    oof_df = pd.DataFrame(oof_rows).sort_values("stem").reset_index(drop=True)
    return oof_df


def evaluate_and_summarize_judge(oof_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compute metrics with bootstrap CIs for overall cohort, per-source, and per-fold."""
    subsets = [
        ("Overall", oof_df),
        ("Source: Shenzhen", oof_df[oof_df["source"] == "Shenzhen"]),
        ("Source: Montgomery", oof_df[oof_df["source"] == "Montgomery"]),
    ]
    for k in range(5):
        subsets.append((f"Fold {k}", oof_df[oof_df["fold"] == k]))

    results: list[dict[str, Any]] = []
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]

    for group_name, sub_df in subsets:
        yt = sub_df["label"].values.astype(int)
        yp = sub_df["prob_tb"].values.astype(float)
        n = len(sub_df)

        pt = compute_binary_metrics(yt, yp, threshold=0.5)
        row: dict[str, Any] = {
            "group": group_name,
            "n_samples": n,
            "tn": pt["tn"],
            "fp": pt["fp"],
            "fn": pt["fn"],
            "tp": pt["tp"],
        }

        # Bootstrap CIs for overall and per-source; point metrics for individual folds
        if group_name.startswith("Overall") or group_name.startswith("Source:"):
            ci_res = bootstrap_ci_metrics(yt, yp, threshold=0.5, n_bootstraps=1000, seed=1337)
            for m in metric_keys:
                row[m] = ci_res[m]["point"]
                row[f"{m}_ci_lower"] = ci_res[m]["ci_lower"]
                row[f"{m}_ci_upper"] = ci_res[m]["ci_upper"]
        else:
            for m in metric_keys:
                row[m] = pt[m]
                row[f"{m}_ci_lower"] = float("nan")
                row[f"{m}_ci_upper"] = float("nan")

        results.append(row)

    metrics_df = pd.DataFrame(results)

    # Format summary text
    ov = metrics_df[metrics_df["group"] == "Overall"].iloc[0]
    sz = metrics_df[metrics_df["group"] == "Source: Shenzhen"].iloc[0]
    mg = metrics_df[metrics_df["group"] == "Source: Montgomery"].iloc[0]

    summary_lines = [
        "=" * 80,
        "TUBERCULOSIS (TB) EFFICIENTNET-B0 JUDGE CLASSIFIER EVALUATION SUMMARY",
        "=" * 80,
        "Evaluation Model: EfficientNet-B0 (ImageNet Pretrained backbone, 320x320 resolution)",
        "Architecture: Dropout(0.2) + Linear(1280, 1), ImageNet weights, Seed=1337",
        "Cross-Validation: 5-Fold Patient-Level Stratified Out-of-Fold (800 Images Total)",
        "Decision Threshold: 0.50",
        "Confidence Intervals: 95% Non-Parametric Bootstrap (1000 Resamples, Seed=1337)",
        "",
        "1. OVERALL COHORT PERFORMANCE (N = 800)",
        "-" * 80,
        f"Accuracy:    {ov['accuracy']*100:.2f}%  [95% CI: {ov['accuracy_ci_lower']*100:.2f}% - {ov['accuracy_ci_upper']*100:.2f}%]",
        f"Sensitivity: {ov['recall']*100:.2f}%  [95% CI: {ov['recall_ci_lower']*100:.2f}% - {ov['recall_ci_upper']*100:.2f}%]",
        f"Specificity: {ov['specificity']*100:.2f}%  [95% CI: {ov['specificity_ci_lower']*100:.2f}% - {ov['specificity_ci_upper']*100:.2f}%]",
        f"Precision:   {ov['precision']*100:.2f}%  [95% CI: {ov['precision_ci_lower']*100:.2f}% - {ov['precision_ci_upper']*100:.2f}%]",
        f"F1-Score:    {ov['f1']:.4f}   [95% CI: {ov['f1_ci_lower']:.4f} - {ov['f1_ci_upper']:.4f}]",
        f"ROC AUC:     {ov['auc']:.4f}   [95% CI: {ov['auc_ci_lower']:.4f} - {ov['auc_ci_upper']:.4f}]",
        "",
        "Confusion Matrix:",
        "               Pred Normal    Pred TB",
        f"  True Normal     {ov['tn']:<12} {ov['fp']:<12}",
        f"  True TB         {ov['fn']:<12} {ov['tp']:<12}",
        "",
        "2. PER-SOURCE PERFORMANCE BREAKDOWN",
        "-" * 80,
        f"Shenzhen Hospital (N = 662):",
        f"  Accuracy:    {sz['accuracy']*100:.2f}%  [95% CI: {sz['accuracy_ci_lower']*100:.2f}% - {sz['accuracy_ci_upper']*100:.2f}%]",
        f"  Sensitivity: {sz['recall']*100:.2f}%  [95% CI: {sz['recall_ci_lower']*100:.2f}% - {sz['recall_ci_upper']*100:.2f}%]",
        f"  Specificity: {sz['specificity']*100:.2f}%  [95% CI: {sz['specificity_ci_lower']*100:.2f}% - {sz['specificity_ci_upper']*100:.2f}%]",
        f"  ROC AUC:     {sz['auc']:.4f}   [95% CI: {sz['auc_ci_lower']:.4f} - {sz['auc_ci_upper']:.4f}]",
        "",
        f"Montgomery County (N = 138):",
        f"  Accuracy:    {mg['accuracy']*100:.2f}%  [95% CI: {mg['accuracy_ci_lower']*100:.2f}% - {mg['accuracy_ci_upper']*100:.2f}%]",
        f"  Sensitivity: {mg['recall']*100:.2f}%  [95% CI: {mg['recall_ci_lower']*100:.2f}% - {mg['recall_ci_upper']*100:.2f}%]",
        f"  Specificity: {mg['specificity']*100:.2f}%  [95% CI: {mg['specificity_ci_lower']*100:.2f}% - {mg['specificity_ci_upper']*100:.2f}%]",
        f"  ROC AUC:     {mg['auc']:.4f}   [95% CI: {mg['auc_ci_lower']:.4f} - {mg['auc_ci_upper']:.4f}]",
        "",
        "3. PER-FOLD PERFORMANCE SUMMARY",
        "-" * 80,
    ]

    for k in range(5):
        f_row = metrics_df[metrics_df["group"] == f"Fold {k}"].iloc[0]
        summary_lines.append(
            f"Fold {k}     (N={f_row['n_samples']}): "
            f"Accuracy={f_row['accuracy']*100:.2f}%, Sens={f_row['recall']*100:.2f}%, "
            f"Spec={f_row['specificity']*100:.2f}%, AUC={f_row['auc']:.4f}"
        )

    summary_lines.append("=" * 80)
    summary_text = "\n".join(summary_lines) + "\n"
    return metrics_df, summary_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TB EfficientNet-B0 Judge classifier across 5 folds.")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch budget per fold (default: 15).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val AUC (default: 5).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers (default: 2).")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed (default: 1337).")
    parser.add_argument("--smoke", action="store_true", help="Measure time per epoch on fold 0 and estimate total budget.")
    parser.add_argument("--force", action="store_true", help="Force retrain existing checkpoints.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is NOT available! RTX 2050 CUDA GPU is mandatory.")

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)} with mixed precision.")

    set_seed(args.seed)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.is_file(), f"Manifest missing at {manifest_path}"
    manifest_df = pd.read_csv(manifest_path)

    if args.smoke:
        print("\n--- Measuring Time Per Epoch for Judge (Fold 0) ---")
        t0 = time.time()
        _ = train_judge_fold(
            fold=0,
            manifest_df=manifest_df,
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            save_path=JUDGE_DIR / "smoke_fold0.pt",
        )
        t_epoch = time.time() - t0
        print(f"\n[Judge Timing Benchmark] 1 epoch took: {t_epoch:.2f} seconds.")
        print(f"Estimated time for 5 folds x 15 epochs (75 epochs max): {(75 * t_epoch) / 60.0:.2f} minutes.")
        print(f"Estimated time for 5 folds x 20 epochs (100 epochs max): {(100 * t_epoch) / 60.0:.2f} minutes.")
        return

    # Train all 5 folds
    total_start = time.time()
    for k in range(5):
        ckpt_path = JUDGE_DIR / f"fold{k}.pt"
        if ckpt_path.is_file() and not args.force:
            print(f"Skipping fold {k}: checkpoint {ckpt_path.name} already exists.")
            continue

        train_judge_fold(
            fold=k,
            manifest_df=manifest_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            save_path=ckpt_path,
        )

    total_training_minutes = (time.time() - total_start) / 60.0
    print(f"\n5-fold Judge training completed in {total_training_minutes:.2f} minutes.")

    # Compute out-of-fold predictions on original images
    oof_df = compute_oof_predictions(manifest_df, device=device, batch_size=args.batch_size)
    oof_path = RESULTS_DIR / "judge_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved out-of-fold predictions to {oof_path}")

    # Compute metrics and summary
    metrics_df, summary_text = evaluate_and_summarize_judge(oof_df)
    metrics_path = RESULTS_DIR / "judge_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved judge metrics with bootstrap CIs to {metrics_path}")

    summary_path = RESULTS_DIR / "judge_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved judge summary to {summary_path}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
