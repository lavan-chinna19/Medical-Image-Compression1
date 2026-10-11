"""Generate visualization plots for lossless ROI and mask coding evaluation."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

LOSSLESS_RESULTS_CSV = RESULTS_DIR / "lossless_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_roi_bytes_vs_pixels(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot of ROI bytes vs ROI pixel count colored by source."""
    plt.figure(figsize=(8, 6), dpi=300)

    sources = sorted(df["source"].unique())
    colors = {"Montgomery": "#1f77b4", "Shenzhen": "#ff7f0e"}

    for s in sources:
        sub = df[df["source"] == s]
        plt.scatter(
            sub["roi_pixels"],
            sub["roi_bytes"],
            alpha=0.6,
            s=25,
            color=colors.get(s, "#2ca02c"),
            label=f"{s} (N={len(sub)})",
            edgecolors="none",
        )

    # Reference slope lines (3 bpp, 4 bpp, 5 bpp)
    x_ref = np.linspace(df["roi_pixels"].min(), df["roi_pixels"].max(), 100)
    plt.plot(x_ref, x_ref * 3 / 8, "--", color="#888888", lw=1.2, label="3.0 bits/pixel")
    plt.plot(x_ref, x_ref * 4 / 8, ":", color="#555555", lw=1.2, label="4.0 bits/pixel")
    plt.plot(x_ref, x_ref * 5 / 8, "-.", color="#333333", lw=1.2, label="5.0 bits/pixel")

    plt.xlabel("ROI Pixel Count", fontsize=11, fontweight="bold")
    plt.ylabel("Lossless ROI Compressed Size (Bytes)", fontsize=11, fontweight="bold")
    plt.title("Lossless ROI Bytes vs ROI Pixel Count", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved ROI bytes vs pixels scatter to {out_path}")


def plot_bpp_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Comparison of bits per ROI pixel for MED+Huffman, Context-MED, and PNG."""
    plt.figure(figsize=(8.5, 5.5), dpi=300)

    med_bpp = df["roi_bits_per_roi_pixel"].dropna()
    ctx_bpp = (df["context_roi_bytes"] * 8 / df["roi_pixels"]).dropna()
    png_bpp = df["png_bpp"].dropna()

    data = [med_bpp, ctx_bpp, png_bpp]
    labels = [
        f"MED + Huffman\n(Mean: {med_bpp.mean():.2f} bpp)",
        f"Context MED\n(Mean: {ctx_bpp.mean():.2f} bpp)",
        f"Full-Frame PNG\n(Mean: {png_bpp.mean():.2f} bpp)",
    ]

    box = plt.boxplot(data, patch_artist=True, labels=labels, showmeans=True, meanline=True)

    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    for median in box["medians"]:
        median.set(color="black", linewidth=1.5)
    for mean in box["means"]:
        mean.set(color="blue", linewidth=1.8, linestyle="--")

    plt.ylabel("Bits per Pixel (bpp)", fontsize=11, fontweight="bold")
    plt.title("Rate Comparison: ROI Lossless vs Full PNG Baseline", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved BPP comparison boxplot to {out_path}")


def plot_mask_overhead(df: pd.DataFrame, out_path: Path) -> None:
    """Mask bytes as a percentage of ROI bytes, comparing RLE and BBox variants."""
    plt.figure(figsize=(8, 5.5), dpi=300)

    pct_rle = (df["mask_rle_bytes"] / df["roi_bytes"]) * 100
    pct_bbox = (df["mask_bbox_bytes"] / df["roi_bytes"]) * 100
    pct_min = (df["mask_bytes"] / df["roi_bytes"]) * 100

    plt.hist(pct_rle, bins=30, alpha=0.5, color="#1f77b4", label=f"Row RLE (Mean: {pct_rle.mean():.2f}%)")
    plt.hist(pct_bbox, bins=30, alpha=0.5, color="#ff7f0e", label=f"BBox RLE (Mean: {pct_bbox.mean():.2f}%)")

    plt.axvline(pct_min.mean(), color="black", linestyle="--", lw=1.8, label=f"Optimal Mask (Mean: {pct_min.mean():.2f}%)")

    plt.xlabel("Mask Overhead as % of ROI Compressed Bytes", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title("Mask Coding Overhead Relative to Lossless ROI Data", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved mask overhead histogram to {out_path}")


def main():
    assert LOSSLESS_RESULTS_CSV.is_file(), f"Results CSV not found: {LOSSLESS_RESULTS_CSV}"
    df = pd.read_csv(LOSSLESS_RESULTS_CSV)
    print(f"Loaded {len(df)} lossless evaluation rows from {LOSSLESS_RESULTS_CSV}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_roi_bytes_vs_pixels(df, PLOTS_DIR / "lossless_roi_bytes_vs_pixels.png")
    plot_bpp_comparison(df, PLOTS_DIR / "lossless_bpp_comparison.png")
    plot_mask_overhead(df, PLOTS_DIR / "lossless_mask_overhead.png")


if __name__ == "__main__":
    main()
