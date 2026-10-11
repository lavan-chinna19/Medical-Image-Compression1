"""Diagnose the degenerate-rung failure mode for DWT compression.

Investigates:
1. Probability collapse at low rungs (Q=5, 10, 15) vs original distributions.
2. Fraction of images whose original probability aligns within tolerance of the collapsed median.
3. Cross-model fragility correlation per rung (|steering delta prob| vs |judge delta prob|).
4. Scatter plots of steering delta prob vs judge delta prob across low, mid, and high rungs.

Outputs:
- results/plots/dwt_degenerate_histograms.png
- results/plots/dwt_delta_prob_scatter.png
- results/controller_diagnosis.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

LADDERS_CSV = RESULTS_DIR / "ladders.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
DIAGNOSIS_CSV = RESULTS_DIR / "controller_diagnosis.csv"


def run_dwt_diagnosis() -> None:
    assert LADDERS_CSV.is_file(), f"Ladders file missing at {LADDERS_CSV}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LADDERS_CSV)
    dwt_df = df[df["family"] == "dwt_only"].copy()
    assert len(dwt_df) > 0, "No dwt_only records found in ladders.csv"

    # Distinct images original probabilities
    orig_df = dwt_df[dwt_df["quality"] == 5].copy().sort_values("stem").reset_index(drop=True)
    p_orig_st = orig_df["prob_orig_steering"].values
    p_orig_jd = orig_df["prob_orig_judge"].values
    n_images = len(orig_df)

    # -------------------------------------------------------------------------
    # 1. Histograms: Steering and Judge at lowest 3 rungs vs Original
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    low_qs = [5, 10, 15]

    for col_idx, q in enumerate(low_qs):
        sub_q = dwt_df[dwt_df["quality"] == q].sort_values("stem").reset_index(drop=True)
        p_st = sub_q["prob_steering"].values
        p_jd = sub_q["prob_judge"].values

        # Top row: Steering
        ax_st = axes[0, col_idx]
        ax_st.hist(p_orig_st, bins=25, alpha=0.5, color="#1f77b4", label="Original", density=True)
        ax_st.hist(p_st, bins=25, alpha=0.6, color="#d62728", label=f"DWT Q={q}", density=True)
        ax_st.set_title(f"Steering Classifier (Q={q})", fontsize=12, fontweight="bold")
        ax_st.set_xlabel("Probability TB", fontsize=11)
        ax_st.grid(True, alpha=0.3)
        ax_st.legend(fontsize=10)
        med_st = np.median(p_st)
        ax_st.axvline(med_st, color="darkred", linestyle="--", label=f"Median: {med_st:.2f}")

        # Bottom row: Judge
        ax_jd = axes[1, col_idx]
        ax_jd.hist(p_orig_jd, bins=25, alpha=0.5, color="#2ca02c", label="Original", density=True)
        ax_jd.hist(p_jd, bins=25, alpha=0.6, color="#ff7f0e", label=f"DWT Q={q}", density=True)
        ax_jd.set_title(f"Independent Judge (Q={q})", fontsize=12, fontweight="bold")
        ax_jd.set_xlabel("Probability TB", fontsize=11)
        ax_jd.grid(True, alpha=0.3)
        ax_jd.legend(fontsize=10)
        med_jd = np.median(p_jd)
        ax_jd.axvline(med_jd, color="darkorange", linestyle="--", label=f"Median: {med_jd:.2f}")

    axes[0, 0].set_ylabel("Density (Steering)", fontsize=12)
    axes[1, 0].set_ylabel("Density (Judge)", fontsize=12)
    fig.suptitle(
        "Degenerate Collapse at Lowest Rungs: Output Probability Distribution Shift",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    hist_plot_path = PLOTS_DIR / "dwt_degenerate_histograms.png"
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()
    print(f"Saved degenerate histograms to {hist_plot_path}")

    # -------------------------------------------------------------------------
    # 2. Fraction of images whose original steering probability lies within
    #    tolerance of the median steering output at lowest rung (Q=5)
    # -------------------------------------------------------------------------
    q5_df = dwt_df[dwt_df["quality"] == 5].sort_values("stem").reset_index(drop=True)
    med_steering_q5 = float(np.median(q5_df["prob_steering"]))
    med_judge_q5 = float(np.median(q5_df["prob_judge"]))

    tolerances = [0.02, 0.05, 0.10]
    frac_steering_match = {}
    frac_judge_match = {}

    for tol in tolerances:
        st_match = np.abs(p_orig_st - med_steering_q5) <= tol
        jd_match = np.abs(p_orig_jd - med_judge_q5) <= tol
        frac_steering_match[tol] = float(np.mean(st_match))
        frac_judge_match[tol] = float(np.mean(jd_match))

    print("\n--- Degenerate Collision Analysis at Q=5 ---")
    print(f"Median Steering Probability at Q=5: {med_steering_q5:.4f}")
    for tol in tolerances:
        print(f"  Tolerance {tol:.2f}: {frac_steering_match[tol]*100:.2f}% of cohort images naturally match Q=5 median")

    # -------------------------------------------------------------------------
    # 3. Per-Rung Correlation between |steering delta prob| and |judge delta prob|
    # -------------------------------------------------------------------------
    diagnosis_rows: list[dict[str, Any]] = []
    all_qs = sorted(dwt_df["quality"].unique())

    for q in all_qs:
        sub_q = dwt_df[dwt_df["quality"] == q].sort_values("stem").reset_index(drop=True)
        d_st = np.abs(sub_q["prob_steering"].values - sub_q["prob_orig_steering"].values)
        d_jd = np.abs(sub_q["prob_judge"].values - sub_q["prob_orig_judge"].values)

        r_val, r_p = pearsonr(d_st, d_jd)
        rho_val, rho_p = spearmanr(d_st, d_jd)

        diagnosis_rows.append({
            "quality": q,
            "rung": f"dwt_q{q}",
            "mean_bpp": float(np.mean(sub_q["bpp"])),
            "mean_steering_abs_delta": float(np.mean(d_st)),
            "std_steering_abs_delta": float(np.std(d_st)),
            "mean_judge_abs_delta": float(np.mean(d_jd)),
            "std_judge_abs_delta": float(np.std(d_jd)),
            "pearson_r": float(r_val),
            "pearson_p": float(r_p),
            "spearman_rho": float(rho_val),
            "spearman_p": float(rho_p),
            "frac_orig_within_tol_02": frac_steering_match[0.02] if q == 5 else float("nan"),
            "frac_orig_within_tol_05": frac_steering_match[0.05] if q == 5 else float("nan"),
            "frac_orig_within_tol_10": frac_steering_match[0.10] if q == 5 else float("nan"),
        })

    diag_df = pd.DataFrame(diagnosis_rows)
    diag_df.to_csv(DIAGNOSIS_CSV, index=False)
    print(f"\nSaved diagnosis table to {DIAGNOSIS_CSV}")
    print(diag_df[["quality", "mean_bpp", "mean_steering_abs_delta", "mean_judge_abs_delta", "pearson_r", "spearman_rho"]])

    # -------------------------------------------------------------------------
    # 4. Scatter of steering delta prob vs judge delta prob at three rungs
    # -------------------------------------------------------------------------
    scatter_qs = [5, 40, 90]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

    for ax, q in zip(axes, scatter_qs):
        sub_q = dwt_df[dwt_df["quality"] == q].sort_values("stem").reset_index(drop=True)
        d_st = np.abs(sub_q["prob_steering"].values - sub_q["prob_orig_steering"].values)
        d_jd = np.abs(sub_q["prob_judge"].values - sub_q["prob_orig_judge"].values)

        r_val, _ = pearsonr(d_st, d_jd)
        rho_val, _ = spearmanr(d_st, d_jd)

        ax.scatter(d_st, d_jd, alpha=0.45, color="#1f77b4", edgecolors="none")
        ax.set_title(f"DWT Q={q} (r={r_val:.2f}, rho={rho_val:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("|Steering Delta Prob|", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # Plot 45-degree reference line
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", alpha=0.6)

    axes[0].set_ylabel("|Judge Delta Prob|", fontsize=12)
    fig.suptitle(
        "Cross-Classifier Diagnostic Fragility: |Steering Delta Prob| vs |Judge Delta Prob|",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    scatter_plot_path = PLOTS_DIR / "dwt_delta_prob_scatter.png"
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"Saved delta prob scatter plot to {scatter_plot_path}")


if __name__ == "__main__":
    run_dwt_diagnosis()
