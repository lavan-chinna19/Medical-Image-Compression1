"""Plot task-based compression benchmark results across codecs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"

CODEC_COLORS = {
    "jpeg": "#e377c2",       # Pink/Magenta
    "jpeg2000": "#1f77b4",   # Blue
    "dct": "#ff7f0e",        # Orange
    "dwt": "#2ca02c",        # Green
    "roi_dwt": "#d62728",    # Red
}

CODEC_LABELS = {
    "jpeg": "JPEG (Standard)",
    "jpeg2000": "JPEG 2000 (OpenJPEG)",
    "dct": "Custom DCT",
    "dwt": "Tuned DWT (bior2.2)",
    "roi_dwt": "ROI-DWT (Lossless ROI)",
}


def plot_metric_vs_bpp(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: Path,
    ref_orig: bool = True,
    ref_levels: list[float] | None = None,
) -> None:
    """Plot metric vs bpp for all codecs with 95% bootstrap CI bands."""
    plt.figure(figsize=(9, 6))

    orig_row = summary_df[summary_df["codec"] == "original"]
    if ref_orig and not orig_row.empty:
        orig_val = float(orig_row[metric].iloc[0])
        plt.axhline(
            orig_val,
            color="#333333",
            linestyle="--",
            lw=1.8,
            label=f"Original (Uncompressed = {orig_val:.3f})",
        )

    if ref_levels:
        for lvl in ref_levels:
            plt.axhline(
                lvl,
                color="gray",
                linestyle=":",
                lw=1.2,
                label=f"Target {lvl*100:.0f}%",
            )

    codecs = ["jpeg", "jpeg2000", "dct", "dwt", "roi_dwt"]
    for c in codecs:
        sub = summary_df[summary_df["codec"] == c].sort_values("mean_bpp")
        if sub.empty:
            continue
        color = CODEC_COLORS.get(c, "black")
        lbl = CODEC_LABELS.get(c, c)

        x = sub["mean_bpp"].values
        y = sub[metric].values
        y_low = sub[f"{metric}_ci_lower"].values
        y_high = sub[f"{metric}_ci_upper"].values

        plt.plot(x, y, marker="o", lw=2.2, color=color, label=lbl)
        plt.fill_between(x, y_low, y_high, color=color, alpha=0.18)

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right" if "auc" in metric or "recall" in metric else "upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path.name}")


def plot_metric_vs_bpp_per_source(
    src_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title_prefix: str,
    save_path: Path,
    ref_orig: bool = True,
    ref_levels: list[float] | None = None,
) -> None:
    """Plot metric vs bpp side-by-side for Shenzhen and Montgomery."""
    sources = sorted(src_df["source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(16, 6))

    for ax, src in zip(axes, sources):
        sub_src = src_df[src_df["source"] == src]
        orig_row = sub_src[sub_src["codec"] == "original"]
        if ref_orig and not orig_row.empty:
            orig_val = float(orig_row[metric].iloc[0])
            ax.axhline(
                orig_val,
                color="#333333",
                linestyle="--",
                lw=1.8,
                label=f"Original ({orig_val:.3f})",
            )

        if ref_levels:
            for lvl in ref_levels:
                ax.axhline(
                    lvl,
                    color="gray",
                    linestyle=":",
                    lw=1.2,
                    label=f"Target {lvl*100:.0f}%",
                )

        codecs = ["jpeg", "jpeg2000", "dct", "dwt", "roi_dwt"]
        for c in codecs:
            sub = sub_src[sub_src["codec"] == c].sort_values("mean_bpp")
            if sub.empty:
                continue
            color = CODEC_COLORS.get(c, "black")
            lbl = CODEC_LABELS.get(c, c)

            x = sub["mean_bpp"].values
            y = sub[metric].values
            y_low = sub[f"{metric}_ci_lower"].values
            y_high = sub[f"{metric}_ci_upper"].values

            ax.plot(x, y, marker="o", lw=2.2, color=color, label=lbl)
            ax.fill_between(x, y_low, y_high, color=color, alpha=0.18)

        ax.set_xlabel("Bitrate (bpp)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{src} ({sub_src['n_samples'].iloc[0]} images)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right" if "auc" in metric or "recall" in metric else "upper right", fontsize=9)

    fig.suptitle(f"{title_prefix} by Scanner Source", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path.name}")


def plot_operating_points_barchart(op_df: pd.DataFrame, save_path: Path) -> None:
    """Plot bar chart of required bitrate/file size per codec at equal flip rates."""
    # Filter valid rows
    codecs = [c for c in op_df["codec"] if c != "original"]
    bpp_5 = []
    bpp_2 = []
    names = []

    for c in codecs:
        row = op_df[op_df["codec"] == c].iloc[0]
        v5 = row["bpp_flip_le_5pct"]
        v2 = row["bpp_flip_le_2pct"]
        bpp_5.append(float(v5) if isinstance(v5, (int, float)) else np.nan)
        bpp_2.append(float(v2) if isinstance(v2, (int, float)) else np.nan)
        names.append(CODEC_LABELS.get(c, c))

    x = np.arange(len(codecs))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, bpp_5, width, label="Flip Rate ≤ 5%", color="#1f77b4", alpha=0.85)
    bars2 = plt.bar(x + width / 2, bpp_2, width, label="Flip Rate ≤ 2%", color="#d62728", alpha=0.85)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylabel("Required Bitrate (bpp)", fontsize=12)
    plt.title("Operating Points: Minimum Bitrate at Equal Diagnostic Flip Rates", fontsize=14, fontweight="bold")
    plt.xticks(x, names, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path.name}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    per_source_path = RESULTS_DIR / "benchmark_summary_per_source.csv"
    op_path = RESULTS_DIR / "benchmark_operating_points.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not per_source_path.is_file():
        raise FileNotFoundError(f"Missing {per_source_path}")
    if not op_path.is_file():
        raise FileNotFoundError(f"Missing {op_path}")

    summary_df = pd.read_csv(summary_path)
    per_source_df = pd.read_csv(per_source_path)
    op_df = pd.read_csv(op_path)

    # 1. Overall plots
    plot_metric_vs_bpp(
        summary_df,
        metric="auc",
        ylabel="ROC Area Under the Curve (AUC)",
        title="Diagnostic ROC AUC vs Bitrate Across Codecs",
        save_path=PLOTS_DIR / "benchmark_auc_vs_bpp.png",
        ref_orig=True,
    )

    plot_metric_vs_bpp(
        summary_df,
        metric="recall",
        ylabel="Sensitivity / Recall",
        title="Diagnostic Sensitivity vs Bitrate Across Codecs",
        save_path=PLOTS_DIR / "benchmark_recall_vs_bpp.png",
        ref_orig=True,
    )

    plot_metric_vs_bpp(
        summary_df,
        metric="flip_rate",
        ylabel="Diagnostic Flip Rate",
        title="Diagnosis Flip Rate vs Bitrate Across Codecs",
        save_path=PLOTS_DIR / "benchmark_flip_vs_bpp.png",
        ref_orig=False,
        ref_levels=[0.05, 0.02],
    )

    # 2. Per-source plots
    plot_metric_vs_bpp_per_source(
        per_source_df,
        metric="auc",
        ylabel="ROC AUC",
        title_prefix="Diagnostic ROC AUC vs Bitrate",
        save_path=PLOTS_DIR / "benchmark_auc_vs_bpp_per_source.png",
        ref_orig=True,
    )

    plot_metric_vs_bpp_per_source(
        per_source_df,
        metric="recall",
        ylabel="Sensitivity / Recall",
        title_prefix="Diagnostic Sensitivity vs Bitrate",
        save_path=PLOTS_DIR / "benchmark_recall_vs_bpp_per_source.png",
        ref_orig=True,
    )

    plot_metric_vs_bpp_per_source(
        per_source_df,
        metric="flip_rate",
        ylabel="Diagnostic Flip Rate",
        title_prefix="Diagnosis Flip Rate vs Bitrate",
        save_path=PLOTS_DIR / "benchmark_flip_vs_bpp_per_source.png",
        ref_orig=False,
        ref_levels=[0.05, 0.02],
    )

    # 3. Bar chart of operating points
    plot_operating_points_barchart(
        op_df,
        save_path=PLOTS_DIR / "benchmark_operating_points_barchart.png",
    )


if __name__ == "__main__":
    main()
