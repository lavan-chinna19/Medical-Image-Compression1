"""Generate Rate-Distortion curves for compression baselines."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric dataframe filtering out infinities and missing values."""
    res = df.copy()
    for col in ["bpp", "psnr_full", "ssim_full", "psnr_roi", "ssim_roi", "ssim_background"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col].replace("inf", np.nan), errors="coerce")
    return res


def plot_rd_psnr(df: pd.DataFrame, out_path: Path) -> None:
    """Plot Rate-Distortion curve: bpp vs PSNR full image."""
    plt.figure(figsize=(8, 5.5), dpi=300)

    colors = {"JPEG": "#1f77b4", "JPEG2000": "#ff7f0e", "PNG": "#2ca02c"}
    markers = {"JPEG": "o", "JPEG2000": "s", "PNG": "^"}

    for codec in ["JPEG", "JPEG2000"]:
        sub = df[df["codec"] == codec].dropna(subset=["bpp", "psnr_full"])
        if sub.empty:
            continue

        # Group by setting and compute mean and standard error
        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_psnr=("psnr_full", "mean"),
            sem_psnr=("psnr_full", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(agg["mean_bpp"], agg["mean_psnr"], marker=markers[codec], color=colors[codec], label=f"{codec} (DCT/DWT)", lw=2)
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_psnr"] - agg["sem_psnr"],
            agg["mean_psnr"] + agg["sem_psnr"],
            color=colors[codec],
            alpha=0.2,
        )

    # Note PNG lossless reference point
    png_sub = df[df["codec"] == "PNG"].dropna(subset=["bpp"])
    if not png_sub.empty:
        png_bpp = png_sub["bpp"].mean()
        plt.axvline(x=png_bpp, color=colors["PNG"], linestyle="--", alpha=0.7, label=f"PNG Lossless ({png_bpp:.2f} bpp, PSNR=∞)")

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR, dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Baseline Codecs on Chest X-Rays (PSNR)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD PSNR plot to {out_path}")


def plot_rd_ssim(df: pd.DataFrame, out_path: Path) -> None:
    """Plot Rate-Distortion curve: bpp vs SSIM full image."""
    plt.figure(figsize=(8, 5.5), dpi=300)

    colors = {"JPEG": "#1f77b4", "JPEG2000": "#ff7f0e", "PNG": "#2ca02c"}
    markers = {"JPEG": "o", "JPEG2000": "s", "PNG": "*"}

    for codec in ["JPEG", "JPEG2000"]:
        sub = df[df["codec"] == codec].dropna(subset=["bpp", "ssim_full"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_ssim=("ssim_full", "mean"),
            sem_ssim=("ssim_full", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(agg["mean_bpp"], agg["mean_ssim"], marker=markers[codec], color=colors[codec], label=f"{codec}", lw=2)
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_ssim"] - agg["sem_ssim"],
            agg["mean_ssim"] + agg["sem_ssim"],
            color=colors[codec],
            alpha=0.2,
        )

    # Plot PNG point (SSIM = 1.0)
    png_sub = df[df["codec"] == "PNG"].dropna(subset=["bpp", "ssim_full"])
    if not png_sub.empty:
        png_bpp = png_sub["bpp"].mean()
        plt.scatter([png_bpp], [1.0], color=colors["PNG"], marker=markers["PNG"], s=150, zorder=5, label=f"PNG Lossless ({png_bpp:.2f} bpp, SSIM=1.0)")

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Baseline Codecs on Chest X-Rays (SSIM)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD SSIM plot to {out_path}")


def plot_rd_roi_psnr(df: pd.DataFrame, out_path: Path) -> None:
    """Plot ROI-only PSNR vs bpp on masked images."""
    plt.figure(figsize=(8, 5.5), dpi=300)

    colors = {"JPEG": "#1f77b4", "JPEG2000": "#ff7f0e", "PNG": "#2ca02c"}
    markers = {"JPEG": "o", "JPEG2000": "s"}

    masked_df = df[df["has_mask"] == True].copy()

    for codec in ["JPEG", "JPEG2000"]:
        sub = masked_df[masked_df["codec"] == codec].dropna(subset=["bpp", "psnr_roi"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_roi=("psnr_roi", "mean"),
            sem_roi=("psnr_roi", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(agg["mean_bpp"], agg["mean_roi"], marker=markers[codec], color=colors[codec], label=f"{codec} (ROI)", lw=2)
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_roi"] - agg["sem_roi"],
            agg["mean_roi"] + agg["sem_roi"],
            color=colors[codec],
            alpha=0.2,
        )

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Lung ROI PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Lung ROI Distortion vs Bitrate (Masked Images)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD ROI PSNR plot to {out_path}")


def plot_rd_by_source(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel rate-distortion plot: Montgomery vs Shenzhen."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    sources = [("Montgomery", axes[0]), ("Shenzhen", axes[1])]
    colors = {"JPEG": "#1f77b4", "JPEG2000": "#ff7f0e"}
    markers = {"JPEG": "o", "JPEG2000": "s"}

    for src_name, ax in sources:
        src_df = df[df["source"] == src_name]
        for codec in ["JPEG", "JPEG2000"]:
            sub = src_df[src_df["codec"] == codec].dropna(subset=["bpp", "psnr_full"])
            if sub.empty:
                continue

            agg = sub.groupby("setting").agg(
                mean_bpp=("bpp", "mean"),
                mean_psnr=("psnr_full", "mean"),
                sem_psnr=("psnr_full", "sem"),
            ).sort_values("mean_bpp")

            ax.plot(agg["mean_bpp"], agg["mean_psnr"], marker=markers[codec], color=colors[codec], label=f"{codec}", lw=2)
            ax.fill_between(
                agg["mean_bpp"],
                agg["mean_psnr"] - agg["sem_psnr"],
                agg["mean_psnr"] + agg["sem_psnr"],
                color=colors[codec],
                alpha=0.2,
            )

        # PNG vertical line
        png_sub = src_df[src_df["codec"] == "PNG"].dropna(subset=["bpp"])
        if not png_sub.empty:
            png_bpp = png_sub["bpp"].mean()
            ax.axvline(x=png_bpp, color="#2ca02c", linestyle="--", alpha=0.7, label=f"PNG Lossless ({png_bpp:.2f} bpp)")

        ax.set_title(f"{src_name} Set", fontsize=12, fontweight="bold")
        ax.set_xlabel("Bitrate (bpp)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    plt.suptitle("Rate-Distortion Comparison Across Hospital Datasets", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD by Source plot to {out_path}")


def plot_rd_roi_ssim(df: pd.DataFrame, out_path: Path) -> None:
    """Plot ROI-only SSIM vs bpp on masked images."""
    plt.figure(figsize=(8, 5.5), dpi=300)

    colors = {"JPEG": "#1f77b4", "JPEG2000": "#ff7f0e", "PNG": "#2ca02c"}
    markers = {"JPEG": "o", "JPEG2000": "s"}

    masked_df = df[df["has_mask"] == True].copy()

    for codec in ["JPEG", "JPEG2000"]:
        sub = masked_df[masked_df["codec"] == codec].dropna(subset=["bpp", "ssim_roi"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_roi=("ssim_roi", "mean"),
            sem_roi=("ssim_roi", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(agg["mean_bpp"], agg["mean_roi"], marker=markers[codec], color=colors[codec], label=f"{codec} (ROI)", lw=2)
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_roi"] - agg["sem_roi"],
            agg["mean_roi"] + agg["sem_roi"],
            color=colors[codec],
            alpha=0.2,
        )

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Lung ROI SSIM", fontsize=11, fontweight="bold")
    plt.title("Lung ROI Structural Similarity vs Bitrate (Masked Images)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD ROI SSIM plot to {out_path}")


def generate_all_plots() -> None:
    """Generate all baseline rate-distortion plots."""
    assert BASELINES_CSV.is_file(), f"Baselines CSV not found: {BASELINES_CSV}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(BASELINES_CSV)
    df = safe_numeric_df(df_raw)

    plot_rd_psnr(df, PLOTS_DIR / "rd_curve_psnr.png")
    plot_rd_ssim(df, PLOTS_DIR / "rd_curve_ssim.png")
    plot_rd_roi_psnr(df, PLOTS_DIR / "rd_curve_psnr_roi.png")
    plot_rd_roi_ssim(df, PLOTS_DIR / "rd_curve_ssim_roi.png")
    plot_rd_by_source(df, PLOTS_DIR / "rd_curve_psnr_by_source.png")


if __name__ == "__main__":
    generate_all_plots()
