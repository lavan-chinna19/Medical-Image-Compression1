"""Plot Rate-Distortion curves comparing Custom DCT codec against JPEG and JPEG 2000 baselines."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.baselines import jpeg_codec
from medcomp.config import PROCESSED_IMAGES_DIR, RESULTS_DIR
from medcomp.dct_codec import dct_encode
from medcomp.io_utils import load_image
from medcomp.metrics import bits_per_pixel, psnr, ssim

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
DCT_RESULTS_CSV = RESULTS_DIR / "dct_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric conversion for rate and distortion metrics."""
    res = df.copy()
    for col in ["bpp", "psnr_full", "ssim_full", "psnr_roi", "ssim_roi", "ssim_background"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col].replace("inf", np.nan), errors="coerce")
    return res


def plot_combined_rd_psnr(df_all: pd.DataFrame, out_path: Path) -> None:
    """Plot bpp vs psnr_full for Custom DCT, JPEG, and JPEG 2000."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    codec_styles = {
        "Custom DCT": {"color": "#d62728", "marker": "D", "lw": 2.2, "ls": "-"},
        "JPEG": {"color": "#1f77b4", "marker": "o", "lw": 1.8, "ls": "--"},
        "JPEG2000": {"color": "#ff7f0e", "marker": "s", "lw": 1.8, "ls": "-."},
    }

    for codec_name, style in codec_styles.items():
        sub = df_all[df_all["codec_label"] == codec_name].dropna(subset=["bpp", "psnr_full"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_psnr=("psnr_full", "mean"),
            sem_psnr=("psnr_full", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(
            agg["mean_bpp"], agg["mean_psnr"],
            marker=style["marker"], color=style["color"],
            linestyle=style["ls"], lw=style["lw"],
            label=codec_name,
        )
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_psnr"] - agg["sem_psnr"],
            agg["mean_psnr"] + agg["sem_psnr"],
            color=style["color"], alpha=0.15,
        )

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR, dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion (PSNR): Custom DCT vs Standard JPEG and JPEG 2000", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD PSNR plot to {out_path}")


def plot_combined_rd_ssim(df_all: pd.DataFrame, out_path: Path) -> None:
    """Plot bpp vs ssim_full for Custom DCT, JPEG, and JPEG 2000."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    codec_styles = {
        "Custom DCT": {"color": "#d62728", "marker": "D", "lw": 2.2, "ls": "-"},
        "JPEG": {"color": "#1f77b4", "marker": "o", "lw": 1.8, "ls": "--"},
        "JPEG2000": {"color": "#ff7f0e", "marker": "s", "lw": 1.8, "ls": "-."},
    }

    for codec_name, style in codec_styles.items():
        sub = df_all[df_all["codec_label"] == codec_name].dropna(subset=["bpp", "ssim_full"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_ssim=("ssim_full", "mean"),
            sem_ssim=("ssim_full", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(
            agg["mean_bpp"], agg["mean_ssim"],
            marker=style["marker"], color=style["color"],
            linestyle=style["ls"], lw=style["lw"],
            label=codec_name,
        )
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_ssim"] - agg["sem_ssim"],
            agg["mean_ssim"] + agg["sem_ssim"],
            color=style["color"], alpha=0.15,
        )

    plt.xlabel("Bitrate (bits per pixel, bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion (SSIM): Custom DCT vs Standard JPEG and JPEG 2000", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD SSIM plot to {out_path}")


def plot_combined_rd_roi_psnr(df_all: pd.DataFrame, out_path: Path) -> None:
    """Plot lung ROI PSNR vs bpp on masked images."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    masked_df = df_all[df_all["has_mask"] == True].copy()
    codec_styles = {
        "Custom DCT": {"color": "#d62728", "marker": "D", "lw": 2.2, "ls": "-"},
        "JPEG": {"color": "#1f77b4", "marker": "o", "lw": 1.8, "ls": "--"},
        "JPEG2000": {"color": "#ff7f0e", "marker": "s", "lw": 1.8, "ls": "-."},
    }

    for codec_name, style in codec_styles.items():
        sub = masked_df[masked_df["codec_label"] == codec_name].dropna(subset=["bpp", "psnr_roi"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_roi=("psnr_roi", "mean"),
            sem_roi=("psnr_roi", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(
            agg["mean_bpp"], agg["mean_roi"],
            marker=style["marker"], color=style["color"],
            linestyle=style["ls"], lw=style["lw"],
            label=f"{codec_name} (ROI)",
        )
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_roi"] - agg["sem_roi"],
            agg["mean_roi"] + agg["sem_roi"],
            color=style["color"], alpha=0.15,
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


def plot_combined_by_source(df_all: pd.DataFrame, out_path: Path) -> None:
    """Two-panel rate-distortion plot (Montgomery vs Shenzhen)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

    sources = [("Montgomery", axes[0]), ("Shenzhen", axes[1])]
    codec_styles = {
        "Custom DCT": {"color": "#d62728", "marker": "D", "lw": 2.2, "ls": "-"},
        "JPEG": {"color": "#1f77b4", "marker": "o", "lw": 1.8, "ls": "--"},
        "JPEG2000": {"color": "#ff7f0e", "marker": "s", "lw": 1.8, "ls": "-."},
    }

    for src_name, ax in sources:
        src_df = df_all[df_all["source"] == src_name]
        for codec_name, style in codec_styles.items():
            sub = src_df[src_df["codec_label"] == codec_name].dropna(subset=["bpp", "psnr_full"])
            if sub.empty:
                continue

            agg = sub.groupby("setting").agg(
                mean_bpp=("bpp", "mean"),
                mean_psnr=("psnr_full", "mean"),
                sem_psnr=("psnr_full", "sem"),
            ).sort_values("mean_bpp")

            ax.plot(
                agg["mean_bpp"], agg["mean_psnr"],
                marker=style["marker"], color=style["color"],
                linestyle=style["ls"], lw=style["lw"],
                label=codec_name,
            )
            ax.fill_between(
                agg["mean_bpp"],
                agg["mean_psnr"] - agg["sem_psnr"],
                agg["mean_psnr"] + agg["sem_psnr"],
                color=style["color"], alpha=0.15,
            )

        ax.set_title(f"{src_name} Dataset", fontsize=12, fontweight="bold")
        ax.set_xlabel("Bitrate (bpp)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    plt.suptitle("Rate-Distortion Performance Across Hospital Datasets", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD by Source plot to {out_path}")


def generate_side_by_side_comparison(out_path: Path) -> None:
    """Side-by-side visual comparison of one X-ray from each source at ~0.5 bpp: Original, DCT, JPEG."""
    mcu_images = sorted(PROCESSED_IMAGES_DIR.glob("MCUCXR_*.png"))
    chn_images = sorted(PROCESSED_IMAGES_DIR.glob("CHNCXR_*.png"))

    if not mcu_images or not chn_images:
        print("Skipping side-by-side plot: processed images not found.")
        return

    mcu_img = load_image(mcu_images[0])
    chn_img = load_image(chn_images[0])

    # Compress at quality=50 (yields ~0.48-0.50 bpp)
    # 1. Montgomery
    mcu_dct_bytes, mcu_dct_rec = dct_encode(mcu_img, quality=50)
    mcu_jpeg_bytes, mcu_jpeg_rec = jpeg_codec(mcu_img, quality=50)

    mcu_dct_bpp = bits_per_pixel(len(mcu_dct_bytes), mcu_img.size)
    mcu_dct_psnr = psnr(mcu_img, mcu_dct_rec)
    mcu_jpeg_bpp = bits_per_pixel(len(mcu_jpeg_bytes), mcu_img.size)
    mcu_jpeg_psnr = psnr(mcu_img, mcu_jpeg_rec)

    # 2. Shenzhen
    chn_dct_bytes, chn_dct_rec = dct_encode(chn_img, quality=50)
    chn_jpeg_bytes, chn_jpeg_rec = jpeg_codec(chn_img, quality=50)

    chn_dct_bpp = bits_per_pixel(len(chn_dct_bytes), chn_img.size)
    chn_dct_psnr = psnr(chn_img, chn_dct_rec)
    chn_jpeg_bpp = bits_per_pixel(len(chn_jpeg_bytes), chn_img.size)
    chn_jpeg_psnr = psnr(chn_img, chn_jpeg_rec)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), dpi=300)

    # Row 0: Montgomery
    axes[0, 0].imshow(mcu_img, cmap="gray")
    axes[0, 0].set_title(f"Montgomery Original\n({mcu_images[0].stem})", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mcu_dct_rec, cmap="gray")
    axes[0, 1].set_title(f"Custom DCT (Q=50)\n{mcu_dct_bpp:.2f} bpp | {mcu_dct_psnr:.2f} dB", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mcu_jpeg_rec, cmap="gray")
    axes[0, 2].set_title(f"JPEG Baseline (Q=50)\n{mcu_jpeg_bpp:.2f} bpp | {mcu_jpeg_psnr:.2f} dB", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 1: Shenzhen
    axes[1, 0].imshow(chn_img, cmap="gray")
    axes[1, 0].set_title(f"Shenzhen Original\n({chn_images[0].stem})", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(chn_dct_rec, cmap="gray")
    axes[1, 1].set_title(f"Custom DCT (Q=50)\n{chn_dct_bpp:.2f} bpp | {chn_dct_psnr:.2f} dB", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(chn_jpeg_rec, cmap="gray")
    axes[1, 2].set_title(f"JPEG Baseline (Q=50)\n{chn_jpeg_bpp:.2f} bpp | {chn_jpeg_psnr:.2f} dB", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle("Visual Fidelity Comparison at ~0.5 bpp: Original vs Custom DCT vs Standard JPEG", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved side-by-side comparison to {out_path}")


def plot_combined_rd_roi_ssim(df_all: pd.DataFrame, out_path: Path) -> None:
    """Plot lung ROI SSIM vs bpp on masked images."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    masked_df = df_all[df_all["has_mask"] == True].copy()
    codec_styles = {
        "Custom DCT": {"color": "#d62728", "marker": "D", "lw": 2.2, "ls": "-"},
        "JPEG": {"color": "#1f77b4", "marker": "o", "lw": 1.8, "ls": "--"},
        "JPEG2000": {"color": "#ff7f0e", "marker": "s", "lw": 1.8, "ls": "-."},
    }

    for codec_name, style in codec_styles.items():
        sub = masked_df[masked_df["codec_label"] == codec_name].dropna(subset=["bpp", "ssim_roi"])
        if sub.empty:
            continue

        agg = sub.groupby("setting").agg(
            mean_bpp=("bpp", "mean"),
            mean_roi=("ssim_roi", "mean"),
            sem_roi=("ssim_roi", "sem"),
        ).sort_values("mean_bpp")

        plt.plot(
            agg["mean_bpp"], agg["mean_roi"],
            marker=style["marker"], color=style["color"],
            linestyle=style["ls"], lw=style["lw"],
            label=f"{codec_name} (ROI)",
        )
        plt.fill_between(
            agg["mean_bpp"],
            agg["mean_roi"] - agg["sem_roi"],
            agg["mean_roi"] + agg["sem_roi"],
            color=style["color"], alpha=0.15,
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


def generate_all_dct_plots() -> None:
    """Generate all DCT vs Baseline plots."""
    assert BASELINES_CSV.is_file(), f"Baselines CSV not found: {BASELINES_CSV}"
    assert DCT_RESULTS_CSV.is_file(), f"DCT results CSV not found: {DCT_RESULTS_CSV}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_base = safe_numeric(pd.read_csv(BASELINES_CSV))
    df_dct = safe_numeric(pd.read_csv(DCT_RESULTS_CSV))

    # Add codec_label for distinct grouping
    df_base["codec_label"] = df_base["codec"]
    df_dct["codec_label"] = "Custom DCT"

    # Filter out lossless PNG for continuous RD curves
    df_base_lossy = df_base[df_base["codec"].isin(["JPEG", "JPEG2000"])]
    df_combined = pd.concat([df_dct, df_base_lossy], ignore_index=True)

    plot_combined_rd_psnr(df_combined, PLOTS_DIR / "rd_dct_vs_baselines_psnr.png")
    plot_combined_rd_ssim(df_combined, PLOTS_DIR / "rd_dct_vs_baselines_ssim.png")
    plot_combined_rd_roi_psnr(df_combined, PLOTS_DIR / "rd_dct_vs_baselines_roi_psnr.png")
    plot_combined_rd_roi_ssim(df_combined, PLOTS_DIR / "rd_dct_vs_baselines_roi_ssim.png")
    plot_combined_by_source(df_combined, PLOTS_DIR / "rd_dct_vs_baselines_by_source.png")
    generate_side_by_side_comparison(PLOTS_DIR / "side_by_side_comparison.png")


if __name__ == "__main__":
    generate_all_dct_plots()
