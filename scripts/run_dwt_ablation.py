"""Ablation study on wavelets, decomposition levels, and Huffman table strategies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    RESULTS_DIR,
    compute_original_bytes,
)
from medcomp.dwt_codec import dwt_encode_with_stats
from medcomp.io_utils import load_image
from medcomp.metrics import bits_per_pixel, psnr, ssim

WAVELETS = ["haar", "bior2.2", "bior4.4", "db2"]
LEVELS = [3, 4, 5]
TABLE_MODES = [False, True]  # False: single detail table, True: per-level detail tables
ABLATION_CSV = RESULTS_DIR / "dwt_ablation.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def evaluate_single_config(args: tuple) -> dict:
    """Evaluate one image under one ablation configuration."""
    stem, img_path, w, lvl, per_lvl = args
    img = load_image(img_path)
    orig_bytes = compute_original_bytes(img)
    num_pixels = img.size

    comp_bytes, recon, stats = dwt_encode_with_stats(
        img,
        quality=50,
        wavelet=w,
        levels=lvl,
        per_level_tables=per_lvl,
    )

    c_len = len(comp_bytes)
    bpp = bits_per_pixel(c_len, num_pixels)
    p = psnr(img, recon)
    s = ssim(img, recon)

    return {
        "stem": stem,
        "wavelet": w,
        "levels": lvl,
        "table_mode": "per_level" if per_lvl else "single",
        "orig_bytes": orig_bytes,
        "compressed_bytes": c_len,
        "bpp": round(bpp, 4),
        "psnr": round(p, 2),
        "ssim": round(s, 4),
        "ll_entropy": stats["ll_entropy"],
        "detail_entropy": stats["detail_entropy"],
    }


def run_ablation(workers: int = 4) -> pd.DataFrame:
    """Run ablation on a fixed stratified subset of 20 images."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)

    # Fixed seeded stratified sample of 20 images by source (seed=42)
    rng = np.random.default_rng(42)
    mcu_idx = rng.choice(df[df["source"] == "Montgomery"].index, size=4, replace=False)
    chn_idx = rng.choice(df[df["source"] == "Shenzhen"].index, size=16, replace=False)
    sample_idx = list(mcu_idx) + list(chn_idx)
    sample_df = df.loc[sample_idx].copy()

    items = []
    for _, row in sample_df.iterrows():
        stem = row["stem"]
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        for w in WAVELETS:
            for lvl in LEVELS:
                for per_lvl in TABLE_MODES:
                    items.append((stem, img_path, w, lvl, per_lvl))

    print(f"Running DWT ablation study across {len(items)} evaluations (20 images x 24 configs)...")

    results = []
    if workers <= 1:
        for item in tqdm(items, desc="DWT ablation"):
            results.append(evaluate_single_config(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_single_config, it) for it in items]
            for fut in tqdm(as_completed(futures), total=len(items), desc="DWT ablation"):
                results.append(fut.result())

    df_res = pd.DataFrame(results)
    df_res.sort_values(by=["wavelet", "levels", "table_mode", "stem"], inplace=True)
    df_res.to_csv(ABLATION_CSV, index=False)
    print(f"Saved {len(df_res)} ablation rows to {ABLATION_CSV}")

    # Compute aggregation
    agg = df_res.groupby(["wavelet", "levels", "table_mode"]).agg(
        mean_bpp=("bpp", "mean"),
        mean_psnr=("psnr", "mean"),
        mean_ssim=("ssim", "mean"),
    ).reset_index()

    agg_csv = RESULTS_DIR / "dwt_ablation_summary.csv"
    agg.to_csv(agg_csv, index=False)

    print("\n" + "=" * 80)
    print("                     DWT ABLATION STUDY SUMMARY (Q=50)")
    print("=" * 80)
    print(agg.to_string(index=False))
    print("=" * 80 + "\n")

    # Plot ablation bar charts
    plot_ablation_chart(agg, PLOTS_DIR / "dwt_ablation_bpp_psnr.png")

    return agg


def plot_ablation_chart(agg: pd.DataFrame, out_path: Path) -> None:
    """Generate bar chart comparing mean bpp and PSNR across all ablation configurations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    agg["config_label"] = agg["wavelet"] + " L" + agg["levels"].astype(str) + " (" + agg["table_mode"] + ")"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)

    x = np.arange(len(agg))
    colors = ["#1f77b4" if m == "single" else "#ff7f0e" for m in agg["table_mode"]]

    # PSNR plot
    ax1.bar(x, agg["mean_psnr"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    ax1.set_title("DWT Ablation Study: Distortion (PSNR) and Rate (bpp) at Q=50", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.set_ylim(bottom=agg["mean_psnr"].min() - 1.5, top=agg["mean_psnr"].max() + 1.0)

    # BPP plot
    ax2.bar(x, agg["mean_bpp"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Bitrate (bpp)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(agg["config_label"], rotation=45, ha="right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Custom legend for table modes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Single Table (all details)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Per-Level Tables"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved DWT ablation chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DWT ablation study.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    run_ablation(workers=args.workers)


if __name__ == "__main__":
    main()
