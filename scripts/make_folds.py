"""Assign cross-validation folds and cross-scanner splits to dataset manifest."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, RESULTS_DIR


def make_folds() -> pd.DataFrame:
    """Add 5-fold StratifiedKFold and cross-scanner splits to data/manifest.csv."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}. Run build_manifest.py first."

    df = pd.read_csv(manifest_csv)
    assert len(df) == 800, f"Expected 800 rows in manifest, got {len(df)}"

    # Create stratification target: source + label
    df["strat_group"] = df["source"].astype(str) + "_" + df["label"].astype(str)

    # 5-fold StratifiedKFold (shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["strat_group"])):
        df.loc[val_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Unassigned folds detected"
    assert (df["fold"] <= 4).all(), "Invalid fold indices detected"

    # Cross-scanner split: Shenzhen for train, Montgomery for test
    df["split_xscanner"] = np.where(df["source"] == "Shenzhen", "train", "test")

    # Drop temporary column before saving
    df_save = df.drop(columns=["strat_group"])
    df_save.to_csv(manifest_csv, index=False)
    print(f"Updated manifest with folds saved to {manifest_csv}")

    # Generate results/fold_summary.txt
    summary_lines = [
        "=== FOLD SUMMARY (5-FOLD STRATIFIED K-FOLD) ===",
        f"Total images: {len(df)}",
        f"Random state: 42 (shuffled)",
        f"Stratification key: source + label",
        "",
        "--- OVERALL FOLD SIZES ---",
    ]

    for f_idx in range(5):
        cnt = (df["fold"] == f_idx).sum()
        summary_lines.append(f"Fold {f_idx}: {cnt} images")

    summary_lines.extend([
        "",
        "--- COUNTS PER FOLD BY SOURCE, LABEL, AND HAS_MASK ---",
    ])

    for f_idx in range(5):
        f_df = df[df["fold"] == f_idx]
        summary_lines.append(f"Fold {f_idx} (Total = {len(f_df)}):")
        for src_name, s_group in f_df.groupby("source"):
            summary_lines.append(
                f"  {src_name}: Total={len(s_group)}, "
                f"Class 0={int((s_group['label'] == 0).sum())}, "
                f"Class 1={int((s_group['label'] == 1).sum())}, "
                f"has_mask={int(s_group['has_mask'].sum())}, "
                f"no_mask={int((~s_group['has_mask']).sum())}"
            )

    summary_lines.extend([
        "",
        "--- CROSS-SCANNER SPLIT COUNTS ---",
        f"Train (Shenzhen):   {(df['split_xscanner'] == 'train').sum()}",
        f"Test  (Montgomery): {(df['split_xscanner'] == 'test').sum()}",
    ])

    summary_text = "\n".join(summary_lines) + "\n"
    summary_file = RESULTS_DIR / "fold_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")
    print(f"Fold summary written to {summary_file}")
    print(summary_text)

    return df_save


if __name__ == "__main__":
    make_folds()
