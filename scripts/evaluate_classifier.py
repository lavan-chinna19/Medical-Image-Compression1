"""Evaluate classifier out-of-fold predictions with bootstrap confidence intervals and plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.classifier import bootstrap_ci_metrics, compute_binary_metrics
from medcomp.config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"


def evaluate_subset(
    df: pd.DataFrame,
    group_name: str,
    calc_bootstrap: bool = True,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate point metrics, confusion matrix, and optional bootstrap CIs."""
    y_true = df["label"].values.astype(int)
    y_prob = df["prob_tb"].values.astype(float)
    n = len(df)

    point_res = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    row: dict[str, Any] = {
        "group": group_name,
        "n_samples": n,
        "tn": point_res["tn"],
        "fp": point_res["fp"],
        "fn": point_res["fn"],
        "tp": point_res["tp"],
    }

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]

    if calc_bootstrap:
        ci_res = bootstrap_ci_metrics(y_true, y_prob, threshold=0.5, n_bootstraps=n_bootstraps, seed=seed)
        for k in metric_keys:
            row[k] = ci_res[k]["point"]
            row[f"{k}_ci_lower"] = ci_res[k]["ci_lower"]
            row[f"{k}_ci_upper"] = ci_res[k]["ci_upper"]
    else:
        for k in metric_keys:
            row[k] = point_res[k]
            row[f"{k}_ci_lower"] = float("nan")
            row[f"{k}_ci_upper"] = float("nan")

    return row


def generate_roc_curves_plot(df: pd.DataFrame, save_path: Path) -> None:
    """Plot ROC curves for overall and per-source subsets."""
    plt.figure(figsize=(8, 6))

    subsets = [
        ("Overall (All 800)", df, "#1f77b4"),
        ("Shenzhen (662)", df[df["source"] == "Shenzhen"], "#2ca02c"),
        ("Montgomery (138)", df[df["source"] == "Montgomery"], "#ff7f0e"),
    ]

    for label, sub_df, color in subsets:
        yt = sub_df["label"].values.astype(int)
        yp = sub_df["prob_tb"].values.astype(float)
        fpr, tpr, _ = roc_curve(yt, yp)
        m = compute_binary_metrics(yt, yp)
        auc_val = m["auc"]
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Out-of-Fold Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_confusion_matrices_plot(df: pd.DataFrame, save_path: Path) -> None:
    """Plot confusion matrices for overall and per-source subsets."""
    subsets = [
        ("Overall", df),
        ("Shenzhen", df[df["source"] == "Shenzhen"]),
        ("Montgomery", df[df["source"] == "Montgomery"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (title, sub_df) in zip(axes, subsets):
        yt = sub_df["label"].values.astype(int)
        yp = sub_df["prob_tb"].values.astype(float)
        m = compute_binary_metrics(yt, yp)
        cm = m["confusion_matrix"]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{title} (N={len(sub_df)})", fontsize=13, fontweight="bold")
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Normal (0)", "TB (1)"], fontsize=11)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Normal (0)", "TB (1)"], fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

        # Annotate counts and percentages
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / len(sub_df) * 100.0
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_probability_histograms_plot(df: pd.DataFrame, save_path: Path) -> None:
    """Plot probability histograms separated by true class."""
    subsets = [
        ("Overall", df),
        ("Shenzhen", df[df["source"] == "Shenzhen"]),
        ("Montgomery", df[df["source"] == "Montgomery"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0.0, 1.0, 26)

    for ax, (title, sub_df) in zip(axes, subsets):
        normal_probs = sub_df[sub_df["label"] == 0]["prob_tb"].values
        tb_probs = sub_df[sub_df["label"] == 1]["prob_tb"].values

        ax.hist(normal_probs, bins=bins, alpha=0.65, label=f"Normal (n={len(normal_probs)})", color="#1f77b4", edgecolor="black")
        ax.hist(tb_probs, bins=bins, alpha=0.65, label=f"TB (n={len(tb_probs)})", color="#d62728", edgecolor="black")
        ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")

        ax.set_title(f"{title} Prediction Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Probability of TB", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(loc="upper center", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TB classifier out-of-fold predictions.")
    parser.add_argument("--oof-file", type=str, default=str(RESULTS_DIR / "classifier_oof.csv"), help="OOF predictions file.")
    args = parser.parse_args()

    oof_path = Path(args.oof_file)
    if not oof_path.is_file():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_path}")

    df = pd.read_csv(oof_path)
    print(f"Loaded {len(df)} predictions from {oof_path}")

    # Compute evaluation rows
    rows: list[dict[str, Any]] = []

    # 1. Overall (with 95% bootstrap CI)
    print("Computing Overall metrics with 1000 bootstrap resamples...")
    rows.append(evaluate_subset(df, "Overall", calc_bootstrap=True, n_bootstraps=1000, seed=42))

    # 2. Per source (with 95% bootstrap CI)
    for source in ["Shenzhen", "Montgomery"]:
        print(f"Computing {source} metrics with 1000 bootstrap resamples...")
        sub = df[df["source"] == source]
        rows.append(evaluate_subset(sub, f"Source: {source}", calc_bootstrap=True, n_bootstraps=1000, seed=42))

    # 3. Per fold
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        rows.append(evaluate_subset(sub, f"Fold {fold}", calc_bootstrap=False))

    metrics_df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "classifier_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics table to {metrics_path}")

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    roc_plot_path = PLOTS_DIR / "classifier_roc_curves.png"
    cm_plot_path = PLOTS_DIR / "classifier_confusion_matrices.png"
    hist_plot_path = PLOTS_DIR / "classifier_prob_histograms.png"

    print("Generating ROC curve plot...")
    generate_roc_curves_plot(df, roc_plot_path)

    print("Generating Confusion Matrix plot...")
    generate_confusion_matrices_plot(df, cm_plot_path)

    print("Generating Probability Histograms plot...")
    generate_probability_histograms_plot(df, hist_plot_path)

    # Write summary text
    summary_path = RESULTS_DIR / "classifier_summary.txt"
    ov = rows[0]
    sh = rows[1]
    mo = rows[2]

    summary_text = f"""================================================================================
TUBERCULOSIS (TB) RESNET18 CLASSIFIER EVALUATION SUMMARY
================================================================================
Evaluation Model: ResNet18 (ImageNet Pretrained backbone, 320x320 resolution)
Cross-Validation: 5-Fold Patient-Level Stratified Out-of-Fold (800 Images Total)
Decision Threshold: 0.50
Confidence Intervals: 95% Non-Parametric Bootstrap (1000 Resamples, Seed=42)

1. OVERALL COHORT PERFORMANCE (N = {ov['n_samples']})
--------------------------------------------------------------------------------
Accuracy:    {ov['accuracy']*100:.2f}%  [95% CI: {ov['accuracy_ci_lower']*100:.2f}% - {ov['accuracy_ci_upper']*100:.2f}%]
Sensitivity: {ov['recall']*100:.2f}%  [95% CI: {ov['recall_ci_lower']*100:.2f}% - {ov['recall_ci_upper']*100:.2f}%]
Specificity: {ov['specificity']*100:.2f}%  [95% CI: {ov['specificity_ci_lower']*100:.2f}% - {ov['specificity_ci_upper']*100:.2f}%]
Precision:   {ov['precision']*100:.2f}%  [95% CI: {ov['precision_ci_lower']*100:.2f}% - {ov['precision_ci_upper']*100:.2f}%]
F1-Score:    {ov['f1']:.4f}   [95% CI: {ov['f1_ci_lower']:.4f} - {ov['f1_ci_upper']:.4f}]
ROC AUC:     {ov['auc']:.4f}   [95% CI: {ov['auc_ci_lower']:.4f} - {ov['auc_ci_upper']:.4f}]

Confusion Matrix:
               Pred Normal    Pred TB
  True Normal     {ov['tn']:<12} {ov['fp']:<12}
  True TB         {ov['fn']:<12} {ov['tp']:<12}

2. PER-SOURCE PERFORMANCE BREAKDOWN
--------------------------------------------------------------------------------
Shenzhen Hospital (N = {sh['n_samples']}):
  Accuracy:    {sh['accuracy']*100:.2f}%  [95% CI: {sh['accuracy_ci_lower']*100:.2f}% - {sh['accuracy_ci_upper']*100:.2f}%]
  Sensitivity: {sh['recall']*100:.2f}%  [95% CI: {sh['recall_ci_lower']*100:.2f}% - {sh['recall_ci_upper']*100:.2f}%]
  Specificity: {sh['specificity']*100:.2f}%  [95% CI: {sh['specificity_ci_lower']*100:.2f}% - {sh['specificity_ci_upper']*100:.2f}%]
  ROC AUC:     {sh['auc']:.4f}   [95% CI: {sh['auc_ci_lower']:.4f} - {sh['auc_ci_upper']:.4f}]

Montgomery County (N = {mo['n_samples']}):
  Accuracy:    {mo['accuracy']*100:.2f}%  [95% CI: {mo['accuracy_ci_lower']*100:.2f}% - {mo['accuracy_ci_upper']*100:.2f}%]
  Sensitivity: {mo['recall']*100:.2f}%  [95% CI: {mo['recall_ci_lower']*100:.2f}% - {mo['recall_ci_upper']*100:.2f}%]
  Specificity: {mo['specificity']*100:.2f}%  [95% CI: {mo['specificity_ci_lower']*100:.2f}% - {mo['specificity_ci_upper']*100:.2f}%]
  ROC AUC:     {mo['auc']:.4f}   [95% CI: {mo['auc_ci_lower']:.4f} - {mo['auc_ci_upper']:.4f}]

3. PER-FOLD PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
"""
    for r in rows[3:]:
        summary_text += f"{r['group']:<10} (N={r['n_samples']:<3}): Accuracy={r['accuracy']*100:.2f}%, Sens={r['recall']*100:.2f}%, Spec={r['specificity']*100:.2f}%, AUC={r['auc']:.4f}\n"

    summary_text += f"""
4. GENERATED PLOTS
--------------------------------------------------------------------------------
- ROC Curves:             {roc_plot_path}
- Confusion Matrices:     {cm_plot_path}
- Probability Histograms: {hist_plot_path}
================================================================================
"""
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nWritten summary to {summary_path}")
    print(summary_text)


if __name__ == "__main__":
    main()
