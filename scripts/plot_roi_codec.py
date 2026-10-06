"""Generate comparative plots for hybrid ROI codec evaluation."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.baselines import jpeg2000_codec
from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RESULTS_DIR,
    compute_original_bytes,
)
from medcomp.io_utils import load_image
from medcomp.metrics import bits_per_pixel, psnr
from medcomp.roi_codec import roi_decode, roi_encode

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
ROI_RESULTS_CSV = RESULTS_DIR / "roi_codec_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric values for rate and distortion metrics."""
    res = df.copy()
    for col in ["bpp", "psnr_full", "ssim_full", "psnr_background", "ssim_background", "roi_fraction", "compression_ratio"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col].replace("inf", np.nan), errors="coerce")
    return res


def plot_rd_background_psnr(roi_df: pd.DataFrame, base_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bpp vs psnr_background for ROI-DCT, ROI-DWT, JPEG, JPEG2000."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    # 1. ROI-DWT
    sub_dwt = roi_df[roi_df["method"] == "dwt"].dropna(subset=["bpp", "psnr_background"])
    agg_dwt = sub_dwt.groupby("quality").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_background", "mean")).sort_values("mean_bpp")
    plt.plot(agg_dwt["mean_bpp"], agg_dwt["mean_psnr"], marker="^", color="#2ca02c", lw=2.2, label="Hybrid ROI-DWT")

    # 2. ROI-DCT
    sub_dct = roi_df[roi_df["method"] == "dct"].dropna(subset=["bpp", "psnr_background"])
    agg_dct = sub_dct.groupby("quality").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_background", "mean")).sort_values("mean_bpp")
    plt.plot(agg_dct["mean_bpp"], agg_dct["mean_psnr"], marker="D", color="#d62728", lw=2.0, label="Hybrid ROI-DCT")

    # 3. JPEG
    sub_jpeg = base_df[base_df["codec"] == "JPEG"].dropna(subset=["bpp", "psnr_background"])
    agg_jpeg = sub_jpeg.groupby("setting").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_background", "mean")).sort_values("mean_bpp")
    plt.plot(agg_jpeg["mean_bpp"], agg_jpeg["mean_psnr"], marker="o", color="#1f77b4", lw=1.8, linestyle="--", label="JPEG (Full Image)")

    # 4. JPEG2000
    sub_j2k = base_df[base_df["codec"] == "JPEG2000"].dropna(subset=["bpp", "psnr_background"])
    agg_j2k = sub_j2k.groupby("setting").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_background", "mean")).sort_values("mean_bpp")
    plt.plot(agg_j2k["mean_bpp"], agg_j2k["mean_psnr"], marker="s", color="#ff7f0e", lw=1.8, linestyle="-.", label="JPEG 2000 (Full Image)")

    plt.xlabel("Overall Bit-Rate (bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Background PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Background Fidelity vs Total Bit-Rate", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD background PSNR plot to {out_path}")


def plot_rd_full_psnr(roi_df: pd.DataFrame, base_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bpp vs psnr_full for ROI-DCT, ROI-DWT, JPEG, JPEG2000."""
    plt.figure(figsize=(8.5, 6), dpi=300)

    sub_dwt = roi_df[roi_df["method"] == "dwt"].dropna(subset=["bpp", "psnr_full"])
    agg_dwt = sub_dwt.groupby("quality").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_full", "mean")).sort_values("mean_bpp")
    plt.plot(agg_dwt["mean_bpp"], agg_dwt["mean_psnr"], marker="^", color="#2ca02c", lw=2.2, label="Hybrid ROI-DWT (Lossless ROI)")

    sub_dct = roi_df[roi_df["method"] == "dct"].dropna(subset=["bpp", "psnr_full"])
    agg_dct = sub_dct.groupby("quality").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_full", "mean")).sort_values("mean_bpp")
    plt.plot(agg_dct["mean_bpp"], agg_dct["mean_psnr"], marker="D", color="#d62728", lw=2.0, label="Hybrid ROI-DCT (Lossless ROI)")

    sub_jpeg = base_df[base_df["codec"] == "JPEG"].dropna(subset=["bpp", "psnr_full"])
    agg_jpeg = sub_jpeg.groupby("setting").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_full", "mean")).sort_values("mean_bpp")
    plt.plot(agg_jpeg["mean_bpp"], agg_jpeg["mean_psnr"], marker="o", color="#1f77b4", lw=1.8, linestyle="--", label="JPEG Baseline")

    sub_j2k = base_df[base_df["codec"] == "JPEG2000"].dropna(subset=["bpp", "psnr_full"])
    agg_j2k = sub_j2k.groupby("setting").agg(mean_bpp=("bpp", "mean"), mean_psnr=("psnr_full", "mean")).sort_values("mean_bpp")
    plt.plot(agg_j2k["mean_bpp"], agg_j2k["mean_psnr"], marker="s", color="#ff7f0e", lw=1.8, linestyle="-.", label="JPEG 2000 Baseline")

    plt.xlabel("Overall Bit-Rate (bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Full-Image PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Full-Image PSNR (Boosted by Lossless ROI)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved RD full PSNR plot to {out_path}")


def plot_cr_vs_roi_fraction(roi_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of compression ratio vs roi_fraction per method at Q=50."""
    plt.figure(figsize=(8, 6), dpi=300)

    sub_q50 = roi_df[roi_df["quality"] == 50]
    for method, col, mark in [("dwt", "#2ca02c", "^"), ("dct", "#d62728", "D")]:
        m_sub = sub_q50[sub_q50["method"] == method]
        plt.scatter(
            m_sub["roi_fraction"],
            m_sub["compression_ratio"],
            alpha=0.5,
            s=22,
            color=col,
            marker=mark,
            label=f"ROI-{method.upper()} (Q=50)",
            edgecolors="none",
        )

    plt.xlabel("ROI Area Fraction (Mask Pixels / Total Pixels)", fontsize=11, fontweight="bold")
    plt.ylabel("Compression Ratio", fontsize=11, fontweight="bold")
    plt.title("Compression Ratio vs ROI Area Fraction (Q=50)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved CR vs ROI fraction scatter to {out_path}")


def plot_stacked_bytes_bars(roi_df: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart of mean bytes (mask, roi, background) per quality for ROI-DWT."""
    plt.figure(figsize=(8.5, 5.5), dpi=300)

    sub_dwt = roi_df[roi_df["method"] == "dwt"]
    agg = sub_dwt.groupby("quality").agg(
        mask_bytes=("mask_bytes", "mean"),
        roi_bytes=("roi_bytes", "mean"),
        bg_bytes=("background_bytes", "mean"),
    ).loc[[10, 20, 30, 50, 70, 90]]

    qualities = [str(q) for q in agg.index]
    x = np.arange(len(qualities))
    width = 0.55

    m_b = agg["mask_bytes"].values
    r_b = agg["roi_bytes"].values
    bg_b = agg["bg_bytes"].values

    p1 = plt.bar(x, m_b, width, label="Mask Overhead", color="#1f77b4")
    p2 = plt.bar(x, r_b, width, bottom=m_b, label="Lossless ROI Data", color="#2ca02c")
    p3 = plt.bar(x, bg_b, width, bottom=m_b + r_b, label="Lossy Background Data", color="#ff7f0e")

    plt.xticks(x, [f"Q={q}" for q in qualities], fontsize=10)
    plt.ylabel("Mean Compressed Size (Bytes)", fontsize=11, fontweight="bold")
    plt.title("Payload Breakdown by Quality (Hybrid ROI-DWT)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved stacked bytes breakdown bar plot to {out_path}")


def plot_side_by_side_visual(out_path: Path) -> None:
    """Generate visual side-by-side comparison for one image per source."""
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    manifest = manifest[manifest["has_mask"] == True]

    m_stem = manifest[manifest["source"] == "Montgomery"]["stem"].iloc[0]
    s_stem = manifest[manifest["source"] == "Shenzhen"]["stem"].iloc[0]

    stems = [("Montgomery", m_stem), ("Shenzhen", s_stem)]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=300)

    for row_idx, (src_name, stem) in enumerate(stems):
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

        img = load_image(img_path)
        mask = (load_image(mask_path) > 127)
        orig_bytes = compute_original_bytes(img)

        # Encode with ROI-DWT at Q=50 (~0.8 bpp)
        dwt_data, dwt_recon, _ = roi_encode(img, mask, method="dwt", quality=50, fill="inpaint")
        dwt_bpp = bits_per_pixel(len(dwt_data), img.size)

        # Match JPEG 2000 target ratio
        target_ratio = int(round(orig_bytes / len(dwt_data)))
        j2k_data, j2k_recon = jpeg2000_codec(img, ratio=max(1, target_ratio))
        j2k_bpp = bits_per_pixel(len(j2k_data), img.size)

        diff_dwt = np.clip(np.abs(img.astype(np.float32) - dwt_recon.astype(np.float32)) * 10.0, 0, 255).astype(np.uint8)
        diff_j2k = np.clip(np.abs(img.astype(np.float32) - j2k_recon.astype(np.float32)) * 10.0, 0, 255).astype(np.uint8)

        # Col 0: Original
        axes[row_idx, 0].imshow(img, cmap="gray")
        axes[row_idx, 0].set_title(f"{src_name}: Original", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: ROI-DWT Recon
        axes[row_idx, 1].imshow(dwt_recon, cmap="gray")
        axes[row_idx, 1].set_title(f"ROI-DWT ({dwt_bpp:.3f} bpp)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: J2K Recon
        axes[row_idx, 2].imshow(j2k_recon, cmap="gray")
        axes[row_idx, 2].set_title(f"JPEG 2000 ({j2k_bpp:.3f} bpp)", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Col 3: Diff ROI-DWT (pure black on lungs)
        axes[row_idx, 3].imshow(diff_dwt, cmap="inferno")
        axes[row_idx, 3].set_title(f"ROI-DWT Error (x10)\nLungs Pure Black", fontsize=10, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # Col 4: Diff J2K (visible errors on lungs)
        axes[row_idx, 4].imshow(diff_j2k, cmap="inferno")
        axes[row_idx, 4].set_title(f"JPEG 2000 Error (x10)\nLossy Lungs", fontsize=10, fontweight="bold")
        axes[row_idx, 4].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved side-by-side visual comparison to {out_path}")


def main():
    assert ROI_RESULTS_CSV.is_file(), f"ROI results not found: {ROI_RESULTS_CSV}"
    assert BASELINES_CSV.is_file(), f"Baselines CSV not found: {BASELINES_CSV}"

    roi_df = safe_numeric(pd.read_csv(ROI_RESULTS_CSV))
    base_df = safe_numeric(pd.read_csv(BASELINES_CSV))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_rd_background_psnr(roi_df, base_df, PLOTS_DIR / "rd_roi_vs_baselines_psnr_background.png")
    plot_rd_full_psnr(roi_df, base_df, PLOTS_DIR / "rd_roi_vs_baselines_psnr_full.png")
    plot_cr_vs_roi_fraction(roi_df, PLOTS_DIR / "roi_compression_ratio_vs_fraction.png")
    plot_stacked_bytes_bars(roi_df, PLOTS_DIR / "roi_bytes_stacked_bars.png")
    plot_side_by_side_visual(PLOTS_DIR / "side_by_side_roi_dwt_vs_jpeg2000.png")


if __name__ == "__main__":
    main()
