"""Generate out-of-fold predictions on original processed images using trained fold models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import load_model, predict_proba
from medcomp.cls_data import TBClassificationDataset
from medcomp.config import DATA_DIR, RESULTS_DIR


def predict_out_of_fold(
    manifest_df: pd.DataFrame,
    batch_size: int = 16,
    device: torch.device = None,
) -> pd.DataFrame:
    """Predict out-of-fold probabilities for all images across 5 folds.

    Each image in fold k is evaluated strictly by the fold k model (which was trained
    only on the remaining folds).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_records: list[dict] = []

    for fold in range(5):
        fold_df = manifest_df[manifest_df["fold"] == fold].copy().reset_index(drop=True)
        print(f"Predicting Fold {fold} ({len(fold_df)} images) using checkpoint fold{fold}.pt...")

        model = load_model(fold, device=device)
        model.eval()

        fold_ds = TBClassificationDataset(fold_df, is_train=False)
        fold_loader = DataLoader(fold_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        fold_probs: list[float] = []
        with torch.no_grad():
            for imgs, _, _ in fold_loader:
                probs = predict_proba(model, imgs, device=device, batch_size=batch_size)
                fold_probs.extend(probs.tolist())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for idx, row in fold_df.iterrows():
            prob = float(fold_probs[idx])
            oof_records.append({
                "stem": row["stem"],
                "source": row["source"],
                "label": int(row["label"]),
                "fold": int(row["fold"]),
                "prob_tb": prob,
                "pred": int(prob >= 0.5),
            })

    oof_df = pd.DataFrame(oof_records)
    # Ensure ordering matches original manifest
    oof_df = manifest_df[["stem"]].merge(oof_df, on="stem", how="left")
    return oof_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate out-of-fold predictions on original images.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for classifier inference on RTX 2050!")

    device = torch.device("cuda")
    manifest_path = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    oof_df = predict_out_of_fold(manifest_df, batch_size=args.batch_size, device=device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "classifier_oof.csv"
    oof_df.to_csv(out_path, index=False)
    print(f"\nSaved out-of-fold predictions for {len(oof_df)} images to {out_path}")
    print(oof_df.head())


if __name__ == "__main__":
    main()
