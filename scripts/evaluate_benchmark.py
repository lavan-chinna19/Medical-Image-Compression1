"""Evaluate task-based benchmark results with paired bootstrap confidence intervals and operating points."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

BENCHMARK_CSV = RESULTS_DIR / "benchmark_per_image.csv"


def compute_subset_metrics(df_variant: pd.DataFrame, df_orig: Optional[pd.DataFrame] = None) -> dict[str, float]:
    """Compute point metrics for a variant on a specific set of images."""
    y_true = df_variant["label"].values.astype(int)
    y_prob = df_variant["prob_dec"].values.astype(float)
    y_pred = df_variant["pred_dec"].values.astype(int)
    total = len(y_true)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float((tp + tn) / total) if total > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    flip_rate = float(df_variant["flip"].mean())
    mean_abs_diff = float(df_variant["abs_prob_diff"].mean())
    mean_bpp = float(df_variant["bpp"].mean())

    res = {
        "mean_bpp": mean_bpp,
        "auc": auc,
        "accuracy": acc,
        "recall": rec,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "flip_rate": flip_rate,
        "mean_abs_prob_diff": mean_abs_diff,
    }

    if df_orig is not None:
        orig_true = df_orig["label"].values.astype(int)
        orig_prob = df_orig["prob_dec"].values.astype(float)
        orig_pred = df_orig["pred_dec"].values.astype(int)
        orig_tp = int(np.sum((orig_true == 1) & (orig_pred == 1)))
        orig_fn = int(np.sum((orig_true == 1) & (orig_pred == 0)))
        orig_rec = float(orig_tp / (orig_tp + orig_fn)) if (orig_tp + orig_fn) > 0 else 0.0
        orig_auc = float(roc_auc_score(orig_true, orig_prob)) if len(np.unique(orig_true)) > 1 else float("nan")

        res["delta_auc"] = float(auc - orig_auc)
        res["delta_recall"] = float(rec - orig_rec)
    else:
        res["delta_auc"] = 0.0
        res["delta_recall"] = 0.0

    return res


def evaluate_with_paired_bootstrap(
    df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute metrics and 95% paired bootstrap CIs across all codec settings."""
    unique_stems = np.array(sorted(df["stem"].unique()))
    n_stems = len(unique_stems)

    # Group dataframe by (codec, setting) indexed by stem for fast lookup
    grouped = {key: grp.set_index("stem") for key, grp in df.groupby(["codec", "setting"])}
    orig_df = grouped[("original", 0)]

    # Point estimates
    point_results: list[dict[str, Any]] = []
    keys = list(grouped.keys())

    for key in keys:
        codec, setting = key
        v_df = grouped[key]
        pt = compute_subset_metrics(v_df, orig_df)
        pt["codec"] = codec
        pt["setting"] = setting
        point_results.append(pt)

    # Paired Bootstrap
    metric_keys = [
        "mean_bpp",
        "auc",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "flip_rate",
        "mean_abs_prob_diff",
        "delta_auc",
        "delta_recall",
    ]

    boot_records: dict[tuple[str, Any], dict[str, list[float]]] = {
        k: {m: [] for m in metric_keys} for k in keys
    }

    rng = np.random.default_rng(seed)
    print(f"Running {n_bootstraps} paired bootstrap resamples over {n_stems} images...")

    for _ in range(n_bootstraps):
        sampled_stems = rng.choice(unique_stems, size=n_stems, replace=True)
        sampled_orig = orig_df.loc[sampled_stems]

        # Precompute orig metrics for this sample
        orig_y = sampled_orig["label"].values.astype(int)
        orig_prob = sampled_orig["prob_dec"].values.astype(float)
        orig_pred = sampled_orig["pred_dec"].values.astype(int)
        orig_tp = np.sum((orig_y == 1) & (orig_pred == 1))
        orig_fn = np.sum((orig_y == 1) & (orig_pred == 0))
        orig_rec = float(orig_tp / (orig_tp + orig_fn)) if (orig_tp + orig_fn) > 0 else 0.0
        has_two_classes = len(np.unique(orig_y)) > 1
        orig_auc = float(roc_auc_score(orig_y, orig_prob)) if has_two_classes else float("nan")

        for key in keys:
            v_df = grouped[key].loc[sampled_stems]
            y_t = v_df["label"].values.astype(int)
            y_p = v_df["prob_dec"].values.astype(float)
            y_hat = v_df["pred_dec"].values.astype(int)
            tot = len(y_t)

            tp = np.sum((y_t == 1) & (y_hat == 1))
            tn = np.sum((y_t == 0) & (y_hat == 0))
            fp = np.sum((y_t == 0) & (y_hat == 1))
            fn = np.sum((y_t == 1) & (y_hat == 0))

            acc = float((tp + tn) / tot)
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            auc = float(roc_auc_score(y_t, y_p)) if has_two_classes else float("nan")

            b_entry = boot_records[key]
            b_entry["mean_bpp"].append(float(v_df["bpp"].mean()))
            b_entry["auc"].append(auc)
            b_entry["accuracy"].append(acc)
            b_entry["recall"].append(rec)
            b_entry["specificity"].append(spec)
            b_entry["precision"].append(prec)
            b_entry["f1"].append(f1)
            b_entry["flip_rate"].append(float(v_df["flip"].mean()))
            b_entry["mean_abs_prob_diff"].append(float(v_df["abs_prob_diff"].mean()))
            b_entry["delta_auc"].append(float(auc - orig_auc) if not np.isnan(auc) and not np.isnan(orig_auc) else float("nan"))
            b_entry["delta_recall"].append(float(rec - orig_rec))

    # Assemble summary table
    summary_rows: list[dict[str, Any]] = []
    for pt in point_results:
        key = (pt["codec"], pt["setting"])
        row = {
            "codec": pt["codec"],
            "setting": pt["setting"],
            "n_samples": n_stems,
        }
        b_entry = boot_records[key]

        for m in metric_keys:
            val = pt[m]
            b_vals = np.array(b_entry[m], dtype=float)
            b_vals = b_vals[~np.isnan(b_vals)]
            if len(b_vals) > 0:
                low = float(np.percentile(b_vals, 2.5))
                high = float(np.percentile(b_vals, 97.5))
            else:
                low, high = float("nan"), float("nan")

            row[m] = val
            row[f"{m}_ci_lower"] = low
            row[f"{m}_ci_upper"] = high

        summary_rows.append(row)

    res_df = pd.DataFrame(summary_rows)
    # Order by codec and bpp
    res_df = res_df.sort_values(by=["codec", "mean_bpp"]).reset_index(drop=True)
    return res_df


def interpolate_operating_point(
    bpp_arr: np.ndarray,
    val_arr: np.ndarray,
    target_threshold: float,
    mode: str = "le",  # 'le' for val <= threshold, 'ge' for val >= threshold
) -> Union[float, str]:
    """Find smallest mean bpp satisfying the threshold using linear interpolation in log(bpp)."""
    # Sort strictly by increasing bpp
    sort_idx = np.argsort(bpp_arr)
    bpp_sorted = bpp_arr[sort_idx]
    val_sorted = val_arr[sort_idx]

    cond = (val_sorted <= target_threshold) if mode == "le" else (val_sorted >= target_threshold)

    # If first point already satisfies
    if cond[0]:
        return float(bpp_sorted[0])

    # If even last (highest bitrate) does not satisfy
    if not cond[-1]:
        return "not reached"

    # Find the crossing point
    for i in range(len(bpp_sorted) - 1):
        c1, c2 = cond[i], cond[i + 1]
        if not c1 and c2:
            # Crossed between i and i+1
            x1 = np.log(bpp_sorted[i])
            x2 = np.log(bpp_sorted[i + 1])
            y1 = val_sorted[i]
            y2 = val_sorted[i + 1]

            if abs(y2 - y1) < 1e-9:
                return float(bpp_sorted[i])

            t = (target_threshold - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)
            x_star = x1 + t * (x2 - x1)
            return float(np.exp(x_star))

    # Fallback to the first satisfying point
    first_idx = np.where(cond)[0][0]
    return float(bpp_sorted[first_idx])


def compute_operating_points(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute smallest mean bpp per codec where flip rate <= 5%, <= 2%, and AUC loss <= 0.01."""
    codecs = [c for c in summary_df["codec"].unique() if c != "original"]
    records: list[dict[str, Any]] = []

    for c in codecs:
        sub = summary_df[summary_df["codec"] == c].copy()
        bpps = sub["mean_bpp"].values
        flips = sub["flip_rate"].values
        # AUC loss relative to original: -delta_auc (loss is positive)
        auc_losses = -sub["delta_auc"].values

        bpp_flip_5 = interpolate_operating_point(bpps, flips, target_threshold=0.05, mode="le")
        bpp_flip_2 = interpolate_operating_point(bpps, flips, target_threshold=0.02, mode="le")
        bpp_auc_loss_001 = interpolate_operating_point(bpps, auc_losses, target_threshold=0.01, mode="le")

        records.append({
            "codec": c,
            "bpp_flip_le_5pct": round(bpp_flip_5, 4) if isinstance(bpp_flip_5, float) else bpp_flip_5,
            "bpp_flip_le_2pct": round(bpp_flip_2, 4) if isinstance(bpp_flip_2, float) else bpp_flip_2,
            "bpp_auc_loss_le_0.01": round(bpp_auc_loss_001, 4) if isinstance(bpp_auc_loss_001, float) else bpp_auc_loss_001,
        })

    op_df = pd.DataFrame(records)
    return op_df


def main() -> None:
    if not BENCHMARK_CSV.is_file():
        raise FileNotFoundError(f"Benchmark results not found: {BENCHMARK_CSV}")

    df = pd.read_csv(BENCHMARK_CSV)
    print(f"Loaded {len(df)} benchmark rows from {BENCHMARK_CSV}")

    # 1. Overall summary
    print("\nEvaluating Overall Cohort...")
    summary_df = evaluate_with_paired_bootstrap(df, n_bootstraps=1000, seed=42)
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved overall summary to {summary_path}")

    # 2. Per-source summary
    print("\nEvaluating Per-Source Subsets...")
    per_source_rows = []
    for src in sorted(df["source"].unique()):
        print(f"  Source: {src}...")
        sub_df = df[df["source"] == src]
        src_summary = evaluate_with_paired_bootstrap(sub_df, n_bootstraps=1000, seed=42)
        src_summary.insert(0, "source", src)
        per_source_rows.append(src_summary)

    per_source_df = pd.concat(per_source_rows, ignore_index=True)
    per_source_path = RESULTS_DIR / "benchmark_summary_per_source.csv"
    per_source_df.to_csv(per_source_path, index=False)
    print(f"Saved per-source summary to {per_source_path}")

    # 3. Operating points
    print("\nComputing Operating Points per Codec...")
    op_df = compute_operating_points(summary_df)
    op_path = RESULTS_DIR / "benchmark_operating_points.csv"
    op_df.to_csv(op_path, index=False)
    print(f"Saved operating points to {op_path}")
    print(op_df)


if __name__ == "__main__":
    main()
