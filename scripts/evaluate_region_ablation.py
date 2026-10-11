"""Evaluate region ablation experiment with paired bootstrap CIs and comparison to ROI-DWT."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

ABLATION_CSV = RESULTS_DIR / "region_ablation_per_image.csv"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_per_image.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

VARIANT_COLORS = {
    "lungs_lossy": "#d62728",        # Red
    "background_lossy": "#1f77b4",   # Blue
    "all_lossy": "#2ca02c",          # Green
}

VARIANT_LABELS = {
    "lungs_lossy": "Lungs Lossy (Original BG)",
    "background_lossy": "Background Lossy (Original Lungs)",
    "all_lossy": "All Lossy (Whole Image)",
}


def compute_paired_ablation_metrics(
    df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute point estimates and 95% paired bootstrap CIs for each variant and quality."""
    unique_stems = np.array(sorted(df["stem"].unique()))
    n_stems = len(unique_stems)

    # Reference original probabilities are recorded in each row
    stem_meta = df[["stem", "label", "prob_orig"]].drop_duplicates("stem").set_index("stem")
    orig_y = stem_meta["label"].values.astype(int)
    orig_p = stem_meta["prob_orig"].values.astype(float)
    orig_pred = (orig_p >= 0.5).astype(int)
    orig_tp = np.sum((orig_y == 1) & (orig_pred == 1))
    orig_fn = np.sum((orig_y == 1) & (orig_pred == 0))
    orig_rec = float(orig_tp / (orig_tp + orig_fn)) if (orig_tp + orig_fn) > 0 else 0.0
    orig_auc = float(roc_auc_score(orig_y, orig_p))

    grouped = {key: grp.set_index("stem") for key, grp in df.groupby(["variant", "quality"])}
    keys = list(grouped.keys())

    # Point estimates
    point_rows: list[dict[str, Any]] = []
    for var, q in keys:
        v_df = grouped[(var, q)].loc[unique_stems]
        y_t = v_df["label"].values.astype(int)
        y_p = v_df["prob_dec"].values.astype(float)
        y_hat = (y_p >= 0.5).astype(int)
        tot = len(y_t)

        tp = np.sum((y_t == 1) & (y_hat == 1))
        tn = np.sum((y_t == 0) & (y_hat == 0))
        fp = np.sum((y_t == 0) & (y_hat == 1))
        fn = np.sum((y_t == 1) & (y_hat == 0))

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        auc = float(roc_auc_score(y_t, y_p))
        flip_rate = float(v_df["flip"].mean())
        abs_diff = float(v_df["abs_prob_diff"].mean())

        point_rows.append({
            "variant": var,
            "quality": int(q),
            "n_samples": tot,
            "auc": auc,
            "recall": rec,
            "specificity": spec,
            "flip_rate": flip_rate,
            "mean_abs_prob_diff": abs_diff,
            "delta_auc": auc - orig_auc,
            "delta_recall": rec - orig_rec,
        })

    # Paired Bootstrap
    metric_keys = [
        "auc",
        "recall",
        "specificity",
        "flip_rate",
        "mean_abs_prob_diff",
        "delta_auc",
        "delta_recall",
    ]

    boot_records: dict[tuple[str, int], dict[str, list[float]]] = {
        k: {m: [] for m in metric_keys} for k in keys
    }

    rng = np.random.default_rng(seed)
    for _ in range(n_bootstraps):
        sampled_stems = rng.choice(unique_stems, size=n_stems, replace=True)
        samp_orig = stem_meta.loc[sampled_stems]
        s_orig_y = samp_orig["label"].values.astype(int)
        s_orig_p = samp_orig["prob_orig"].values.astype(float)
        s_orig_pred = (s_orig_p >= 0.5).astype(int)
        s_orig_tp = np.sum((s_orig_y == 1) & (s_orig_pred == 1))
        s_orig_fn = np.sum((s_orig_y == 1) & (s_orig_pred == 0))
        s_orig_rec = float(s_orig_tp / (s_orig_tp + s_orig_fn)) if (s_orig_tp + s_orig_fn) > 0 else 0.0
        has_two_classes = len(np.unique(s_orig_y)) > 1
        s_orig_auc = float(roc_auc_score(s_orig_y, s_orig_p)) if has_two_classes else float("nan")

        for key in keys:
            v_df = grouped[key].loc[sampled_stems]
            y_t = v_df["label"].values.astype(int)
            y_p = v_df["prob_dec"].values.astype(float)
            y_hat = (y_p >= 0.5).astype(int)

            tp = np.sum((y_t == 1) & (y_hat == 1))
            tn = np.sum((y_t == 0) & (y_hat == 0))
            fp = np.sum((y_t == 0) & (y_hat == 1))
            fn = np.sum((y_t == 1) & (y_hat == 0))

            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            auc = float(roc_auc_score(y_t, y_p)) if has_two_classes else float("nan")

            b_entry = boot_records[key]
            b_entry["auc"].append(auc)
            b_entry["recall"].append(rec)
            b_entry["specificity"].append(spec)
            b_entry["flip_rate"].append(float(v_df["flip"].mean()))
            b_entry["mean_abs_prob_diff"].append(float(v_df["abs_prob_diff"].mean()))
            b_entry["delta_auc"].append(float(auc - s_orig_auc) if not np.isnan(auc) and not np.isnan(s_orig_auc) else float("nan"))
            b_entry["delta_recall"].append(float(rec - s_orig_rec))

    # Compile result rows
    summary_rows: list[dict[str, Any]] = []
    for pt in point_rows:
        key = (pt["variant"], pt["quality"])
        row = {
            "variant": pt["variant"],
            "quality": pt["quality"],
            "n_samples": pt["n_samples"],
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

    return pd.DataFrame(summary_rows).sort_values(["variant", "quality"]).reset_index(drop=True)


def compare_background_lossy_with_roi_dwt(
    ablation_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare plain composite background_lossy against full ROI-DWT codec at same qualities."""
    # Filter ROI-DWT rows at qualities [10, 30, 50, 70]
    roi_df = benchmark_df[
        (benchmark_df["codec"] == "roi_dwt") & (benchmark_df["setting"].astype(int).isin([10, 30, 50, 70]))
    ].copy()
    roi_df["quality"] = roi_df["setting"].astype(int)

    bg_df = ablation_df[ablation_df["variant"] == "background_lossy"].copy()

    # Merge on (stem, quality)
    merged = pd.merge(
        bg_df[["stem", "quality", "source", "label", "flip", "abs_prob_diff"]],
        roi_df[["stem", "quality", "flip", "abs_prob_diff"]],
        on=["stem", "quality"],
        suffixes=("_bg_lossy", "_roi"),
    )

    unique_stems = np.array(sorted(merged["stem"].unique()))
    n_stems = len(unique_stems)
    rng = np.random.default_rng(seed)

    comp_records: list[dict[str, Any]] = []

    for q in [10, 30, 50, 70]:
        sub = merged[merged["quality"] == q].set_index("stem").loc[unique_stems]

        # Paired differences: ROI-DWT minus background_lossy
        diff_abs = sub["abs_prob_diff_roi"] - sub["abs_prob_diff_bg_lossy"]
        diff_fl = sub["flip_roi"] - sub["flip_bg_lossy"]

        pt_diff_abs = float(diff_abs.mean())
        pt_diff_fl = float(diff_fl.mean())
        pt_mean_abs_roi = float(sub["abs_prob_diff_roi"].mean())
        pt_mean_abs_bg = float(sub["abs_prob_diff_bg_lossy"].mean())
        pt_flip_roi = float(sub["flip_roi"].mean())
        pt_flip_bg = float(sub["flip_bg_lossy"].mean())

        # Bootstrap
        b_diff_abs: list[float] = []
        b_diff_fl: list[float] = []
        for _ in range(n_bootstraps):
            samp = rng.choice(unique_stems, size=n_stems, replace=True)
            samp_sub = sub.loc[samp]
            b_diff_abs.append(float((samp_sub["abs_prob_diff_roi"] - samp_sub["abs_prob_diff_bg_lossy"]).mean()))
            b_diff_fl.append(float((samp_sub["flip_roi"] - samp_sub["flip_bg_lossy"]).mean()))

        comp_records.append({
            "quality": q,
            "mean_abs_prob_diff_bg_lossy": pt_mean_abs_bg,
            "mean_abs_prob_diff_roi": pt_mean_abs_roi,
            "paired_diff_abs_prob": pt_diff_abs,
            "diff_abs_prob_ci_lower": float(np.percentile(b_diff_abs, 2.5)),
            "diff_abs_prob_ci_upper": float(np.percentile(b_diff_abs, 97.5)),
            "flip_rate_bg_lossy": pt_flip_bg,
            "flip_rate_roi": pt_flip_roi,
            "paired_diff_flip": pt_diff_fl,
            "diff_flip_ci_lower": float(np.percentile(b_diff_fl, 2.5)),
            "diff_flip_ci_upper": float(np.percentile(b_diff_fl, 97.5)),
        })

    return pd.DataFrame(comp_records)


def plot_region_ablation(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot Delta AUC and Flip rate vs quality for the three variants across sources."""
    sources = ["Shenzhen", "Montgomery"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for col_idx, src in enumerate(sources):
        src_df = summary_df[summary_df["source"] == src]

        # Row 0: Delta AUC
        ax_auc = axes[0, col_idx]
        ax_auc.axhline(0.0, color="gray", linestyle="--", lw=1.2, label="Original Reference (0.0)")

        for var in ["lungs_lossy", "background_lossy", "all_lossy"]:
            sub = src_df[src_df["variant"] == var].sort_values("quality")
            x = sub["quality"].values
            y = sub["delta_auc"].values
            y_low = sub["delta_auc_ci_lower"].values
            y_high = sub["delta_auc_ci_upper"].values
            c = VARIANT_COLORS[var]
            lbl = VARIANT_LABELS[var]

            ax_auc.plot(x, y, marker="o", lw=2.2, color=c, label=lbl)
            ax_auc.fill_between(x, y_low, y_high, color=c, alpha=0.18)

        ax_auc.set_title(f"{src}: Delta AUC vs DWT Quality", fontsize=13, fontweight="bold")
        ax_auc.set_xlabel("DWT Quality Setting", fontsize=11)
        ax_auc.set_ylabel("Delta AUC vs Original", fontsize=11)
        ax_auc.grid(True, alpha=0.3)
        ax_auc.legend(fontsize=9, loc="lower right")

        # Row 1: Flip Rate
        ax_flip = axes[1, col_idx]
        ax_flip.axhline(0.05, color="gray", linestyle=":", lw=1.2, label="5% Threshold")
        ax_flip.axhline(0.02, color="gray", linestyle="--", lw=1.0, label="2% Threshold")

        for var in ["lungs_lossy", "background_lossy", "all_lossy"]:
            sub = src_df[src_df["variant"] == var].sort_values("quality")
            x = sub["quality"].values
            y = sub["flip_rate"].values
            y_low = sub["flip_rate_ci_lower"].values
            y_high = sub["flip_rate_ci_upper"].values
            c = VARIANT_COLORS[var]
            lbl = VARIANT_LABELS[var]

            ax_flip.plot(x, y, marker="s", lw=2.2, color=c, label=lbl)
            ax_flip.fill_between(x, y_low, y_high, color=c, alpha=0.18)

        ax_flip.set_title(f"{src}: Diagnostic Flip Rate vs DWT Quality", fontsize=13, fontweight="bold")
        ax_flip.set_xlabel("DWT Quality Setting", fontsize=11)
        ax_flip.set_ylabel("Diagnostic Flip Rate", fontsize=11)
        ax_flip.grid(True, alpha=0.3)
        ax_flip.legend(fontsize=9, loc="upper right")

    plt.suptitle("Region Ablation: Diagnostic Impact of Lung Fields vs Background Compression", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")


def main() -> None:
    if not ABLATION_CSV.is_file():
        raise FileNotFoundError(f"Missing {ABLATION_CSV}")
    if not BENCHMARK_CSV.is_file():
        raise FileNotFoundError(f"Missing {BENCHMARK_CSV}")

    ablation_df = pd.read_csv(ABLATION_CSV)
    benchmark_df = pd.read_csv(BENCHMARK_CSV)
    print(f"Loaded {len(ablation_df)} ablation rows from {ABLATION_CSV}")

    # 1. Overall evaluation
    print("\nComputing overall region ablation summary with 1000 paired bootstrap resamples...")
    ov_summary = compute_paired_ablation_metrics(ablation_df, n_bootstraps=1000, seed=42)
    ov_summary.insert(0, "source", "Overall")

    # 2. Per-source evaluation
    per_source_rows = [ov_summary]
    for src in sorted(ablation_df["source"].unique()):
        print(f"Computing {src} ablation summary...")
        sub = ablation_df[ablation_df["source"] == src]
        s_summary = compute_paired_ablation_metrics(sub, n_bootstraps=1000, seed=42)
        s_summary.insert(0, "source", src)
        per_source_rows.append(s_summary)

    full_summary = pd.concat(per_source_rows, ignore_index=True)
    out_summary_csv = RESULTS_DIR / "region_ablation_summary.csv"
    full_summary.to_csv(out_summary_csv, index=False)
    print(f"Saved region ablation summary to {out_summary_csv}")

    # 3. Compare background_lossy with ROI-DWT
    print("\nComparing background_lossy with ROI-DWT rows from benchmark...")
    comp_df = compare_background_lossy_with_roi_dwt(ablation_df, benchmark_df, n_bootstraps=1000, seed=42)
    comp_csv = RESULTS_DIR / "region_ablation_roi_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"Saved ROI comparison table to {comp_csv}")
    print(comp_df)

    # 4. Generate plot
    plot_path = PLOTS_DIR / "region_ablation.png"
    print("\nGenerating region ablation plot...")
    plot_region_ablation(full_summary, plot_path)


if __name__ == "__main__":
    main()
