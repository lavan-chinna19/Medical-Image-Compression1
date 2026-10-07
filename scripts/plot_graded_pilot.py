"""Generate publication-quality evaluation plots for graded near-lossless compression pilot.

Plots generated into results/plots/:
1. graded_pilot.png:
   - Left: Rate-distortion curve (bpp vs mean abs_prob_diff).
   - Right: Rate-fidelity curve (bpp vs flip rate %).
   - Shows all variant families, DWT-only reference, lossless_dilated curves, and marks Pareto points.
2. graded_pilot_error.png:
   - Rate vs 99th-percentile true lung pixel error for all families, with bounded-error threshold guides.
3. graded_pilot_bytes_by_zone.png:
   - Stacked bar chart showing bytes per zone (Header, Mask, Core, Band, Background) for 5 representative variants.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

SUMMARY_CSV = RESULTS_DIR / "graded_pilot_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

# Curated palette for variant families
FAMILY_STYLES = {
    "dwt_only": {"color": "#1f2937", "marker": "o", "ls": "--", "label": "DWT-only Reference", "lw": 2.2},
    "lossless_dilated_bp4": {"color": "#0284c7", "marker": "s", "ls": "-", "label": "Lossless Dilated (bp=4)", "lw": 1.8},
    "lossless_dilated_bp8": {"color": "#0369a1", "marker": "s", "ls": "--", "label": "Lossless Dilated (bp=8)", "lw": 1.8},
    "graded_d0_2_bp4": {"color": "#7c3aed", "marker": "^", "ls": "-", "label": "Graded (0,2) bp=4", "lw": 1.6},
    "graded_d0_2_bp8": {"color": "#6d28d9", "marker": "^", "ls": "--", "label": "Graded (0,2) bp=8", "lw": 1.6},
    "graded_d0_4_bp4": {"color": "#8b5cf6", "marker": "v", "ls": "-", "label": "Graded (0,4) bp=4", "lw": 1.6},
    "graded_d0_4_bp8": {"color": "#7c3aed", "marker": "v", "ls": "--", "label": "Graded (0,4) bp=8", "lw": 1.6},
    "graded_d1_3_bp4": {"color": "#059669", "marker": "D", "ls": "-", "label": "Graded (1,3) bp=4", "lw": 1.6},
    "graded_d1_3_bp8": {"color": "#047857", "marker": "D", "ls": "--", "label": "Graded (1,3) bp=8", "lw": 1.6},
    "graded_d2_4_bp4": {"color": "#d97706", "marker": "p", "ls": "-", "label": "Graded (2,4) bp=4", "lw": 1.6},
    "graded_d2_4_bp8": {"color": "#b45309", "marker": "p", "ls": "--", "label": "Graded (2,4) bp=8", "lw": 1.6},
    "graded_d3_6_bp4": {"color": "#dc2626", "marker": "h", "ls": "-", "label": "Graded (3,6) bp=4", "lw": 1.6},
    "graded_d3_6_bp8": {"color": "#b91c1c", "marker": "h", "ls": "--", "label": "Graded (3,6) bp=8", "lw": 1.6},
}


def find_pareto_front(candidates: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    """Return non-dominated points minimizing both x_key and y_key."""
    sorted_cand = sorted(candidates, key=lambda c: (c[x_key], c[y_key]))
    pareto: list[dict[str, Any]] = []
    best_y = float("inf")
    for c in sorted_cand:
        if c[y_key] < best_y:
            pareto.append(c)
            best_y = c[y_key]
    return pareto


def plot_graded_pilot(df: pd.DataFrame) -> Path:
    """Generate graded_pilot.png: bpp vs abs_prob_diff and bpp vs flip rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Plot families
    for fam, style in FAMILY_STYLES.items():
        sub = df[df["family"] == fam].sort_values("bpp")
        if sub.empty:
            continue
        ax1.plot(
            sub["bpp"],
            sub["mean_abs_prob_diff"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
            alpha=0.85,
            markersize=6,
        )
        ax2.plot(
            sub["bpp"],
            sub["flip_rate"] * 100.0,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
            alpha=0.85,
            markersize=6,
        )

    # Identify and highlight Pareto points
    cand = df.to_dict("records")
    pareto_diff = find_pareto_front(cand, "bpp", "mean_abs_prob_diff")
    pareto_flip = find_pareto_front(cand, "bpp", "flip_rate")

    for p in pareto_diff:
        ax1.scatter(
            p["bpp"],
            p["mean_abs_prob_diff"],
            color="#e11d48",
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
            marker="*",
        )
        ax1.annotate(
            p["variant"],
            (p["bpp"], p["mean_abs_prob_diff"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="bold",
            color="#9f1239",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe4e6", alpha=0.8, edgecolor="#f43f5e", lw=0.5),
        )

    for p in pareto_flip:
        ax2.scatter(
            p["bpp"],
            p["flip_rate"] * 100.0,
            color="#e11d48",
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
            marker="*",
        )
        ax2.annotate(
            p["variant"],
            (p["bpp"], p["flip_rate"] * 100.0),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="bold",
            color="#9f1239",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe4e6", alpha=0.8, edgecolor="#f43f5e", lw=0.5),
        )

    ax1.set_title("Probability Shift vs Bitrate", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Bitrate (bpp)", fontsize=12)
    ax1.set_ylabel("Mean Absolute Probability Difference", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_title("Diagnostic Flip Rate vs Bitrate", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Bitrate (bpp)", fontsize=12)
    ax2.set_ylabel("Decision Flip Rate (%)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Place combined legend outside
    handles, labels = ax1.get_legend_handles_labels()
    # Add Pareto handle
    star_proxy = matplotlib.lines.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e11d48", markersize=12, label="Pareto-Optimal")
    handles.append(star_proxy)
    labels.append("Pareto-Optimal")

    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    out_path = PLOTS_DIR / "graded_pilot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_graded_pilot_error(df: pd.DataFrame) -> Path:
    """Generate graded_pilot_error.png: bpp vs 99th-percentile lung error."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    for fam, style in FAMILY_STYLES.items():
        sub = df[df["family"] == fam].sort_values("bpp")
        if sub.empty:
            continue
        ax.plot(
            sub["bpp"],
            sub["p99_lung_err"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
            alpha=0.85,
            markersize=7,
        )

    # Horizontal guide thresholds for gray level error
    for th, color in [(2.0, "#059669"), (4.0, "#d97706"), (8.0, "#dc2626")]:
        ax.axhline(th, color=color, linestyle=":", linewidth=1.4, alpha=0.8)
        ax.annotate(
            f"Error ≤ {int(th)} GL",
            xy=(0.02, th),
            xycoords=("axes fraction", "data"),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    # Highlight Pareto points on error
    cand = df.to_dict("records")
    pareto_err = find_pareto_front(cand, "bpp", "p99_lung_err")
    for p in pareto_err:
        ax.scatter(
            p["bpp"],
            p["p99_lung_err"],
            color="#e11d48",
            s=130,
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
            marker="*",
        )
        ax.annotate(
            p["variant"],
            (p["bpp"], p["p99_lung_err"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="bold",
            color="#9f1239",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe4e6", alpha=0.8, edgecolor="#f43f5e", lw=0.5),
        )

    ax.set_title("99th-Percentile True Lung Pixel Error vs Bitrate", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Bitrate (bpp)", fontsize=12)
    ax.set_ylabel("99th-Percentile Pixel Error in True Lungs (Gray Levels)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    star_proxy = matplotlib.lines.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e11d48", markersize=12, label="Pareto-Optimal")
    handles.append(star_proxy)
    labels.append("Pareto-Optimal")

    ax.legend(handles, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    out_path = PLOTS_DIR / "graded_pilot_error.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_graded_bytes_by_zone(df: pd.DataFrame) -> Path:
    """Generate graded_pilot_bytes_by_zone.png: stacked bars for 5 representative variants."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 5 representative variants
    rep_variants = [
        "dwt_q30",
        "lossless_dilated_bp4_q30",
        "graded_d0_2_bp4_q30",
        "graded_d1_3_bp4_q30",
        "graded_d2_4_bp8_q30",
    ]

    labels = [
        "DWT-only (q=30)",
        "Lossless Dilated\n(bp=4, q=30)",
        "Graded (0,2)\n(bp=4, q=30)",
        "Graded (1,3)\n(bp=4, q=30)",
        "Graded (2,4)\n(bp=8, q=30)",
    ]

    sub = df.set_index("variant").loc[rep_variants].reset_index()

    indices = np.arange(len(rep_variants))
    bar_width = 0.55

    headers = sub["bytes_header"].values
    masks = sub["bytes_mask"].values
    cores = sub["bytes_core"].values
    bands = sub["bytes_band"].values
    bgs = sub["bytes_bg"].values
    bpps = sub["bpp"].values

    # Colors for zones
    c_header = "#64748b"
    c_mask = "#f59e0b"
    c_core = "#3b82f6"
    c_band = "#10b981"
    c_bg = "#8b5cf6"

    # Stacked bars
    p1 = ax.bar(indices, headers, bar_width, label="Header", color=c_header, edgecolor="white", linewidth=0.5)
    p2 = ax.bar(indices, masks, bar_width, bottom=headers, label="Mask", color=c_mask, edgecolor="white", linewidth=0.5)
    p3 = ax.bar(indices, cores, bar_width, bottom=headers + masks, label="Core (ROI)", color=c_core, edgecolor="white", linewidth=0.5)
    p4 = ax.bar(indices, bands, bar_width, bottom=headers + masks + cores, label="Band", color=c_band, edgecolor="white", linewidth=0.5)
    p5 = ax.bar(indices, bgs, bar_width, bottom=headers + masks + cores + bands, label="Background", color=c_bg, edgecolor="white", linewidth=0.5)

    totals = headers + masks + cores + bands + bgs
    for i, (tot, bpp_val) in enumerate(zip(totals, bpps)):
        ax.annotate(
            f"{tot/1024:.1f} KB\n({bpp_val:.3f} bpp)",
            xy=(i, tot),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_title("Compressed Bytes Allocation by Zone (5 Representative Variants)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Compressed Size (Bytes)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(totals) * 1.22)

    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()

    out_path = PLOTS_DIR / "graded_pilot_bytes_by_zone.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} variants from {SUMMARY_CSV.name}")

    p1 = plot_graded_pilot(df)
    print(f"Saved rate-distortion plot: {p1}")

    p2 = plot_graded_pilot_error(df)
    print(f"Saved rate-error plot: {p2}")

    p3 = plot_graded_bytes_by_zone(df)
    print(f"Saved byte allocation plot: {p3}")


if __name__ == "__main__":
    main()
