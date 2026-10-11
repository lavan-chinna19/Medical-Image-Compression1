"""Evaluate closed-loop rate controller and compare against fixed-setting baselines.

Metrics computed:
1. Controller Performance (per family & tolerance):
   - Mean BPP and distribution (p25, median, p75, min, max, std)
   - Fraction unsatisfied
   - Judge classifier metrics:
     * Flip rate vs judge's own prediction on original
     * Mean absolute probability deviation
     * ROC AUC, Sensitivity (Recall), Specificity
     * Paired bootstrap 95% CIs for Delta AUC and Delta Recall vs original (1000 resamples, fixed seed)
   - Steering classifier flip rate (for reference)
2. Fixed-Setting Baselines:
   - For every rung: judge metrics and mean BPP
   - Interpolated byte saving relative to fixed rung at matching judge flip rate (overall & per source)
3. Confidence Correlation:
   - Pearson and Spearman correlation between steering confidence |prob_orig - 0.5| and chosen BPP
4. Visualizations:
   - Judge flip rate vs mean BPP (fixed lines + controller points)
   - Judge delta AUC vs mean BPP
   - Histogram of chosen BPP per family
   - Chosen BPP vs confidence scatter plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR
from medcomp.controller import choose_setting, compare_bisection_vs_exhaustive

LADDERS_CSV = RESULTS_DIR / "ladders.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

FAMILIES = ["dwt_only", "graded_d2_4_bp8", "lossless_dilated_bp8"]
TOLERANCES = [0.02, 0.05, 0.10]


def paired_bootstrap_delta_metrics(
    y_true: np.ndarray,
    y_prob_chosen: np.ndarray,
    y_prob_orig: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute point estimates and paired bootstrap 95% CIs for delta AUC and delta Recall."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    pred_chosen = (y_prob_chosen >= 0.5).astype(int)
    pred_orig = (y_prob_orig >= 0.5).astype(int)

    tp_ch = np.sum((y_true == 1) & (pred_chosen == 1))
    fn_ch = np.sum((y_true == 1) & (pred_chosen == 0))
    tn_ch = np.sum((y_true == 0) & (pred_chosen == 0))
    fp_ch = np.sum((y_true == 0) & (pred_chosen == 1))

    tp_or = np.sum((y_true == 1) & (pred_orig == 1))
    fn_or = np.sum((y_true == 1) & (pred_orig == 0))

    rec_ch = float(tp_ch / (tp_ch + fn_ch)) if (tp_ch + fn_ch) > 0 else 0.0
    rec_or = float(tp_or / (tp_or + fn_or)) if (tp_or + fn_or) > 0 else 0.0
    spec_ch = float(tn_ch / (tn_ch + fp_ch)) if (tn_ch + fp_ch) > 0 else 0.0

    auc_ch = float(roc_auc_score(y_true, y_prob_chosen)) if len(np.unique(y_true)) > 1 else 0.5
    auc_or = float(roc_auc_score(y_true, y_prob_orig)) if len(np.unique(y_true)) > 1 else 0.5

    delta_auc_pt = auc_ch - auc_or
    delta_rec_pt = rec_ch - rec_or

    d_auc_samples: list[float] = []
    d_rec_samples: list[float] = []
    auc_samples: list[float] = []
    rec_samples: list[float] = []
    spec_samples: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_s = y_true[idx]
        if len(np.unique(yt_s)) < 2:
            continue

        yp_ch_s = y_prob_chosen[idx]
        yp_or_s = y_prob_orig[idx]
        pr_ch_s = pred_chosen[idx]
        pr_or_s = pred_orig[idx]

        a_ch = float(roc_auc_score(yt_s, yp_ch_s))
        a_or = float(roc_auc_score(yt_s, yp_or_s))
        d_auc_samples.append(a_ch - a_or)
        auc_samples.append(a_ch)

        tp_c = np.sum((yt_s == 1) & (pr_ch_s == 1))
        fn_c = np.sum((yt_s == 1) & (pr_ch_s == 0))
        tp_o = np.sum((yt_s == 1) & (pr_or_s == 1))
        fn_o = np.sum((yt_s == 1) & (pr_or_s == 0))
        tn_c = np.sum((yt_s == 0) & (pr_ch_s == 0))
        fp_c = np.sum((yt_s == 0) & (pr_ch_s == 1))

        r_c = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0
        r_o = float(tp_o / (tp_o + fn_o)) if (tp_o + fn_o) > 0 else 0.0
        s_c = float(tn_c / (tn_c + fp_c)) if (tn_c + fp_c) > 0 else 0.0

        d_rec_samples.append(r_c - r_o)
        rec_samples.append(r_c)
        spec_samples.append(s_c)

    def calc_ci(pt: float, samples: list[float]) -> dict[str, float]:
        if len(samples) > 0:
            return {
                "point": pt,
                "ci_lower": float(np.percentile(samples, 2.5)),
                "ci_upper": float(np.percentile(samples, 97.5)),
            }
        return {"point": pt, "ci_lower": float("nan"), "ci_upper": float("nan")}

    return {
        "auc": calc_ci(auc_ch, auc_samples),
        "recall": calc_ci(rec_ch, rec_samples),
        "specificity": calc_ci(spec_ch, spec_samples),
        "delta_auc": calc_ci(delta_auc_pt, d_auc_samples),
        "delta_recall": calc_ci(delta_rec_pt, d_rec_samples),
    }


def evaluate_controller_selections(ladder_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    """Run controller across all images, families, and tolerances and compute performance metrics."""
    print("\nRunning Rate Controller across families and tolerances...")
    stems = sorted(ladder_df["stem"].unique())
    summary_rows: list[dict[str, Any]] = []
    chosen_dfs: dict[tuple[str, float], pd.DataFrame] = {}

    # Define columns accessible to controller (NEVER include judge or label)
    controller_cols = [
        "stem",
        "family",
        "quality",
        "rung",
        "total_bytes",
        "bpp",
        "prob_steering",
        "prob_orig_steering",
    ]

    for fam in FAMILIES:
        fam_df = ladder_df[ladder_df["family"] == fam].copy()
        if len(fam_df) == 0:
            continue

        for tol in TOLERANCES:
            records_chosen: list[dict[str, Any]] = []

            for stem in stems:
                img_sub = fam_df[fam_df["stem"] == stem]
                if len(img_sub) == 0:
                    continue

                # Pass only steering data to controller (strictly no judge / no label)
                steering_sub = img_sub[controller_cols].copy()
                choice = choose_setting(steering_sub, tolerance=tol, mode="exhaustive")

                # Reconnect chosen rung with judge and label data using unique key (stem, rung)
                full_row = img_sub[img_sub["rung"] == choice["rung"]].iloc[0]

                records_chosen.append({
                    "stem": stem,
                    "source": full_row["source"],
                    "label": int(full_row["label"]),
                    "fold": int(full_row["fold"]),
                    "family": fam,
                    "tolerance": tol,
                    "quality": full_row["quality"],
                    "rung": choice["rung"],
                    "total_bytes": choice["total_bytes"],
                    "bpp": choice["bpp"],
                    "unsatisfied": bool(choice["unsatisfied"]),
                    "prob_orig_steering": full_row["prob_orig_steering"],
                    "prob_orig_judge": full_row["prob_orig_judge"],
                    "prob_steering": full_row["prob_steering"],
                    "prob_judge": full_row["prob_judge"],
                })

            chosen_df = pd.DataFrame(records_chosen)
            chosen_dfs[(fam, tol)] = chosen_df

            # Metrics
            y_true = chosen_df["label"].values.astype(int)
            yp_jd = chosen_df["prob_judge"].values.astype(float)
            yp_jd_orig = chosen_df["prob_orig_judge"].values.astype(float)
            yp_st = chosen_df["prob_steering"].values.astype(float)
            yp_st_orig = chosen_df["prob_orig_steering"].values.astype(float)

            # Flip rates
            pred_jd_orig = (yp_jd_orig >= 0.5).astype(int)
            pred_jd_ch = (yp_jd >= 0.5).astype(int)
            judge_flip_rate = float(np.mean(pred_jd_orig != pred_jd_ch))

            pred_st_orig = (yp_st_orig >= 0.5).astype(int)
            pred_st_ch = (yp_st >= 0.5).astype(int)
            steering_flip_rate = float(np.mean(pred_st_orig != pred_st_ch))

            mean_abs_prob_diff_judge = float(np.mean(np.abs(yp_jd - yp_jd_orig)))
            mean_bpp = float(np.mean(chosen_df["bpp"]))
            mean_bytes = float(np.mean(chosen_df["total_bytes"]))
            frac_unsatisfied = float(np.mean(chosen_df["unsatisfied"]))

            # Distribution of chosen BPP
            bpps = chosen_df["bpp"].values
            bpp_p25 = float(np.percentile(bpps, 25))
            bpp_median = float(np.median(bpps))
            bpp_p75 = float(np.percentile(bpps, 75))
            bpp_std = float(np.std(bpps))
            bpp_min = float(np.min(bpps))
            bpp_max = float(np.max(bpps))

            # Bootstrap CIs for judge metrics
            ci_dict = paired_bootstrap_delta_metrics(y_true, yp_jd, yp_jd_orig, n_bootstraps=1000, seed=42)

            # Steering confidence correlation: |prob_orig_steering - 0.5| vs chosen BPP
            steering_conf = np.abs(yp_st_orig - 0.5)
            r_corr, r_pval = pearsonr(steering_conf, bpps)
            rho_corr, rho_pval = spearmanr(steering_conf, bpps)

            summary_rows.append({
                "family": fam,
                "tolerance": tol,
                "n_images": len(chosen_df),
                "mean_bpp": mean_bpp,
                "bpp_std": bpp_std,
                "bpp_min": bpp_min,
                "bpp_p25": bpp_p25,
                "bpp_median": bpp_median,
                "bpp_p75": bpp_p75,
                "bpp_max": bpp_max,
                "mean_bytes": mean_bytes,
                "frac_unsatisfied": frac_unsatisfied,
                "judge_flip_rate": judge_flip_rate,
                "judge_mean_abs_prob_diff": mean_abs_prob_diff_judge,
                "steering_flip_rate": steering_flip_rate,
                "judge_auc": ci_dict["auc"]["point"],
                "judge_auc_ci_lower": ci_dict["auc"]["ci_lower"],
                "judge_auc_ci_upper": ci_dict["auc"]["ci_upper"],
                "judge_recall": ci_dict["recall"]["point"],
                "judge_recall_ci_lower": ci_dict["recall"]["ci_lower"],
                "judge_recall_ci_upper": ci_dict["recall"]["ci_upper"],
                "judge_specificity": ci_dict["specificity"]["point"],
                "judge_specificity_ci_lower": ci_dict["specificity"]["ci_lower"],
                "judge_specificity_ci_upper": ci_dict["specificity"]["ci_upper"],
                "judge_delta_auc": ci_dict["delta_auc"]["point"],
                "judge_delta_auc_ci_lower": ci_dict["delta_auc"]["ci_lower"],
                "judge_delta_auc_ci_upper": ci_dict["delta_auc"]["ci_upper"],
                "judge_delta_recall": ci_dict["delta_recall"]["point"],
                "judge_delta_recall_ci_lower": ci_dict["delta_recall"]["ci_lower"],
                "judge_delta_recall_ci_upper": ci_dict["delta_recall"]["ci_upper"],
                "confidence_bpp_pearson_r": float(r_corr),
                "confidence_bpp_pearson_p": float(r_pval),
                "confidence_bpp_spearman_rho": float(rho_corr),
                "confidence_bpp_spearman_p": float(rho_pval),
            })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, chosen_dfs


def evaluate_fixed_baselines_and_savings(
    ladder_df: pd.DataFrame,
    controller_summary_df: pd.DataFrame,
    chosen_dfs: dict[tuple[str, float], pd.DataFrame],
) -> pd.DataFrame:
    """Compute fixed-setting baselines and evaluate byte saving of controller at matching judge flip rate."""
    print("\nComputing fixed-setting baselines and comparative byte savings...")

    # Compute fixed rung metrics per family and source
    fixed_results: list[dict[str, Any]] = []

    subsets = [
        ("Overall", ladder_df),
        ("Shenzhen", ladder_df[ladder_df["source"] == "Shenzhen"]),
        ("Montgomery", ladder_df[ladder_df["source"] == "Montgomery"]),
    ]

    for scope_name, sub_df in subsets:
        for fam in FAMILIES:
            fam_sub = sub_df[sub_df["family"] == fam]
            if len(fam_sub) == 0:
                continue

            rungs = fam_sub["rung"].unique()
            for rg in rungs:
                rg_df = fam_sub[fam_sub["rung"] == rg]
                yt = rg_df["label"].values.astype(int)
                yp_jd = rg_df["prob_judge"].values.astype(float)
                yp_jd_or = rg_df["prob_orig_judge"].values.astype(float)

                flip = float(np.mean((yp_jd >= 0.5) != (yp_jd_or >= 0.5)))
                auc = float(roc_auc_score(yt, yp_jd)) if len(np.unique(yt)) > 1 else 0.5
                rec = float(np.sum((yt == 1) & (yp_jd >= 0.5)) / max(1, np.sum(yt == 1)))
                spec = float(np.sum((yt == 0) & (yp_jd < 0.5)) / max(1, np.sum(yt == 0)))
                m_bpp = float(np.mean(rg_df["bpp"]))
                m_bytes = float(np.mean(rg_df["total_bytes"]))

                fixed_results.append({
                    "scope": scope_name,
                    "family": fam,
                    "rung": rg,
                    "quality": rg_df["quality"].iloc[0],
                    "mean_bpp": m_bpp,
                    "mean_bytes": m_bytes,
                    "judge_flip_rate": flip,
                    "judge_auc": auc,
                    "judge_recall": rec,
                    "judge_specificity": spec,
                })

    fixed_df = pd.DataFrame(fixed_results)

    # For each controller result, calculate saving relative to fixed rung with matching judge flip rate
    vs_fixed_rows: list[dict[str, Any]] = []

    for _, ctrl_row in controller_summary_df.iterrows():
        fam = str(ctrl_row["family"])
        tol = float(ctrl_row["tolerance"])
        target_flip = float(ctrl_row["judge_flip_rate"])
        ctrl_mean_bpp = float(ctrl_row["mean_bpp"])
        ctrl_mean_bytes = float(ctrl_row["mean_bytes"])

        # Also get per-source controller numbers from chosen_dfs
        chosen_df = chosen_dfs[(fam, tol)]

        for scope_name in ["Overall", "Shenzhen", "Montgomery"]:
            if scope_name == "Overall":
                c_bytes = ctrl_mean_bytes
                c_bpp = ctrl_mean_bpp
                scope_chosen = chosen_df
            else:
                scope_chosen = chosen_df[chosen_df["source"] == scope_name]
                c_bytes = float(np.mean(scope_chosen["total_bytes"])) if len(scope_chosen) > 0 else float("nan")
                c_bpp = float(np.mean(scope_chosen["bpp"])) if len(scope_chosen) > 0 else float("nan")

            scope_target_flip = float(np.mean(
                (scope_chosen["prob_judge"] >= 0.5) != (scope_chosen["prob_orig_judge"] >= 0.5)
            )) if len(scope_chosen) > 0 else target_flip

            f_family = fixed_df[(fixed_df["scope"] == scope_name) & (fixed_df["family"] == fam)].copy()
            if len(f_family) < 2:
                continue

            # Sort by flip rate for interpolation
            f_family = f_family.sort_values("judge_flip_rate").reset_index(drop=True)
            min_flip = f_family["judge_flip_rate"].min()
            max_flip = f_family["judge_flip_rate"].max()

            if scope_target_flip < min_flip or scope_target_flip > max_flip:
                saving_status = "not reached"
                interp_fixed_bpp = float("nan")
                interp_fixed_bytes = float("nan")
                saving_bytes_pct = float("nan")
                saving_bpp_pct = float("nan")
            else:
                saving_status = "interpolated"
                # Interpolate in log(bpp) and log(bytes)
                log_bpps = np.log(f_family["mean_bpp"].values)
                log_bytes = np.log(f_family["mean_bytes"].values)
                flips = f_family["judge_flip_rate"].values

                interp_log_bpp = float(np.interp(scope_target_flip, flips, log_bpps))
                interp_log_bytes = float(np.interp(scope_target_flip, flips, log_bytes))

                interp_fixed_bpp = float(np.exp(interp_log_bpp))
                interp_fixed_bytes = float(np.exp(interp_log_bytes))

                saving_bytes_pct = float(((interp_fixed_bytes - c_bytes) / interp_fixed_bytes) * 100.0)
                saving_bpp_pct = float(((interp_fixed_bpp - c_bpp) / interp_fixed_bpp) * 100.0)

            vs_fixed_rows.append({
                "family": fam,
                "tolerance": tol,
                "scope": scope_name,
                "controller_judge_flip_rate": scope_target_flip,
                "controller_mean_bpp": c_bpp,
                "controller_mean_bytes": c_bytes,
                "fixed_min_flip": min_flip,
                "fixed_max_flip": max_flip,
                "status": saving_status,
                "interp_fixed_bpp": interp_fixed_bpp,
                "interp_fixed_bytes": interp_fixed_bytes,
                "bytes_saving_pct": saving_bytes_pct,
                "bpp_saving_pct": saving_bpp_pct,
            })

    vs_fixed_df = pd.DataFrame(vs_fixed_rows)
    return vs_fixed_df


def generate_plots(
    ladder_df: pd.DataFrame,
    controller_summary_df: pd.DataFrame,
    chosen_dfs: dict[tuple[str, float], pd.DataFrame],
) -> list[Path]:
    """Generate and save publication-grade diagnostic plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    palette = {
        "dwt_only": "#1f77b4",
        "graded_d2_4_bp8": "#2ca02c",
        "lossless_dilated_bp8": "#ff7f0e",
    }
    tol_markers = {
        0.02: "o",
        0.05: "s",
        0.10: "^",
    }

    # 1. Plot: Judge Flip Rate vs Mean BPP
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in FAMILIES:
        fam_df = ladder_df[ladder_df["family"] == fam]
        if len(fam_df) == 0:
            continue

        rung_means = (
            fam_df.groupby(["rung", "quality"])[["bpp", "prob_judge", "prob_orig_judge"]]
            .apply(lambda g: pd.Series({
                "mean_bpp": g["bpp"].mean(),
                "flip_rate": np.mean((g["prob_judge"] >= 0.5) != (g["prob_orig_judge"] >= 0.5)),
            }))
            .reset_index()
            .sort_values("mean_bpp")
        )

        ax.plot(
            rung_means["mean_bpp"],
            rung_means["flip_rate"] * 100.0,
            marker=".",
            linestyle="-",
            color=palette[fam],
            label=f"Fixed {fam}",
            alpha=0.7,
            linewidth=1.8,
        )

        # Plot controller points
        c_sub = controller_summary_df[controller_summary_df["family"] == fam]
        for _, r in c_sub.iterrows():
            tol_val = float(r["tolerance"])
            ax.scatter(
                r["mean_bpp"],
                r["judge_flip_rate"] * 100.0,
                color=palette[fam],
                marker=tol_markers[tol_val],
                s=90,
                edgecolors="black",
                zorder=5,
                label=f"Controller {fam} (tol={tol_val})" if fam == "dwt_only" else None,
            )

    ax.set_xlabel("Mean Bitrate (bpp)", fontsize=12)
    ax.set_ylabel("Judge Flip Rate (%)", fontsize=12)
    ax.set_title("Judge Classifier Flip Rate vs Mean BPP", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    p1 = PLOTS_DIR / "controller_judge_flip_vs_bpp.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    plot_paths.append(p1)

    # 2. Plot: Judge Delta AUC vs Mean BPP
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in FAMILIES:
        c_sub = controller_summary_df[controller_summary_df["family"] == fam].sort_values("mean_bpp")
        if len(c_sub) == 0:
            continue

        ax.errorbar(
            c_sub["mean_bpp"],
            c_sub["judge_delta_auc"],
            yerr=[
                c_sub["judge_delta_auc"] - c_sub["judge_delta_auc_ci_lower"],
                c_sub["judge_delta_auc_ci_upper"] - c_sub["judge_delta_auc"],
            ],
            fmt="o-",
            color=palette[fam],
            ecolor="gray",
            capsize=4,
            label=f"Controller {fam}",
            linewidth=1.8,
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Bitrate (bpp)", fontsize=12)
    ax.set_ylabel("Judge Delta AUC vs Original", fontsize=12)
    ax.set_title("Judge Delta AUC (with 95% Bootstrap CIs) vs Mean BPP", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = PLOTS_DIR / "controller_judge_delta_auc_vs_bpp.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    plot_paths.append(p2)

    # 3. Plot: Histogram of Chosen BPP per Family (at tolerance 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        key = (fam, 0.05)
        if key in chosen_dfs:
            c_df = chosen_dfs[key]
            ax.hist(c_df["bpp"], bins=20, color=palette[fam], edgecolor="black", alpha=0.75)
            ax.set_title(f"{fam} (tol=0.05)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Chosen bpp", fontsize=11)
            ax.grid(True, alpha=0.3)
            mean_val = float(c_df["bpp"].mean())
            ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.legend(fontsize=10)
    axes[0].set_ylabel("Count", fontsize=12)
    fig.suptitle("Distribution of Rate Controller Chosen BPP Across Cohort (Tolerance 0.05)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3 = PLOTS_DIR / "controller_bpp_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    plot_paths.append(p3)

    # 4. Plot: Chosen BPP vs Steering Confidence Scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, fam in zip(axes, FAMILIES):
        key = (fam, 0.05)
        if key in chosen_dfs:
            c_df = chosen_dfs[key]
            conf = np.abs(c_df["prob_orig_steering"].values - 0.5)
            bpps = c_df["bpp"].values

            ax.scatter(conf, bpps, color=palette[fam], alpha=0.5, edgecolors="none")
            r, _ = pearsonr(conf, bpps)

            # Fit trend line
            m_slope, b_intercept = np.polyfit(conf, bpps, 1)
            x_line = np.linspace(0, 0.5, 50)
            ax.plot(x_line, m_slope * x_line + b_intercept, color="black", linestyle="--", label=f"r = {r:.2f}")

            ax.set_title(f"{fam} (tol=0.05)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Steering Confidence |p_orig - 0.5|", fontsize=11)
            ax.set_ylabel("Chosen bpp", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

    fig.suptitle("Rate Controller Adaptivity: Chosen BPP vs Diagnostic Confidence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = PLOTS_DIR / "controller_bpp_vs_confidence.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    plot_paths.append(p4)

    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate closed-loop rate controller on ladder results.")
    parser.add_argument("--bisection-check", action="store_true", help="Run practical bisection search validation.")
    args = parser.parse_args()

    assert LADDERS_CSV.is_file(), f"Ladder results file not found at {LADDERS_CSV}"
    print(f"Loading ladder results from {LADDERS_CSV}...")
    ladder_df = pd.read_csv(LADDERS_CSV)
    print(f"Loaded {len(ladder_df)} ladder rows ({ladder_df['stem'].nunique()} unique images).")

    # Evaluate controller choices
    summary_df, chosen_dfs = evaluate_controller_selections(ladder_df)
    summary_path = RESULTS_DIR / "controller_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved controller summary to {summary_path}")

    # Evaluate fixed baselines and comparative savings
    vs_fixed_df = evaluate_fixed_baselines_and_savings(ladder_df, summary_df, chosen_dfs)
    vs_fixed_path = RESULTS_DIR / "controller_vs_fixed.csv"
    vs_fixed_df.to_csv(vs_fixed_path, index=False)
    print(f"Saved controller vs fixed baseline savings to {vs_fixed_path}")

    # Practical bisection search validation
    print("\nEvaluating practical bisection search efficiency and agreement...")
    bisection_input_cols = [
        "stem",
        "family",
        "quality",
        "rung",
        "total_bytes",
        "bpp",
        "prob_steering",
        "prob_orig_steering",
    ]
    bisection_comp_df = compare_bisection_vs_exhaustive(
        ladder_df[bisection_input_cols],
        tolerances=TOLERANCES,
    )
    bisection_csv_path = RESULTS_DIR / "controller_bisection_comparison.csv"
    bisection_comp_df.to_csv(bisection_csv_path, index=False)
    print(f"Saved bisection vs exhaustive comparison to {bisection_csv_path}")
    print("\nBisection Agreement & Efficiency Table:")
    print(bisection_comp_df.to_string(index=False))

    # Generate plots
    plot_paths = generate_plots(ladder_df, summary_df, chosen_dfs)
    print("\nGenerated plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
