"""Controlled tuning experiments for DWT codec across 60 masked images."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
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
from medcomp.dwt_codec import dwt_encode
from medcomp.io_utils import load_image
from medcomp.metrics import bd_psnr, bits_per_pixel, psnr

QUALITY_GRID = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
TUNING_SUMMARY_CSV = RESULTS_DIR / "dwt_tuning_summary.csv"


def get_stratified_60_images() -> List[Tuple[str, Path]]:
    """Select a fixed seeded subset of 60 masked images stratified by source."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    masked_df = df[df["has_mask"] == True].copy()

    rng = np.random.default_rng(42)
    mcu = masked_df[masked_df["source"] == "Montgomery"]
    chn = masked_df[masked_df["source"] == "Shenzhen"]

    # Proportional sampling: Montgomery ~ 12, Shenzhen ~ 48
    n_mcu = int(round(60 * len(mcu) / len(masked_df)))
    n_chn = 60 - n_mcu

    chosen_mcu = rng.choice(mcu["stem"].values, size=n_mcu, replace=False)
    chosen_chn = rng.choice(chn["stem"].values, size=n_chn, replace=False)
    stems = list(chosen_mcu) + list(chosen_chn)

    return [(stem, PROCESSED_IMAGES_DIR / f"{stem}.png") for stem in stems]


def _eval_item(item: Tuple[str, Path, int, Dict[str, Any]]) -> Tuple[int, float, float]:
    """Top-level picklable worker function."""
    _, path, q, config = item
    img = load_image(path)
    comp_bytes, recon = dwt_encode(img, quality=q, **config)
    b = bits_per_pixel(len(comp_bytes), img.size)
    p = psnr(img, recon)
    return q, b, p


def evaluate_curve(
    images: List[Tuple[str, Path]],
    config: Dict[str, Any],
    workers: int = 4,
) -> Tuple[List[float], List[float]]:
    """Evaluate mean bpp and mean PSNR across quality grid for a given configuration."""
    tasks = [(stem, path, q, config) for stem, path in images for q in QUALITY_GRID]
    results_by_q = {q: ([], []) for q in QUALITY_GRID}

    if workers <= 1:
        for t in tasks:
            q, b, p = _eval_item(t)
            results_by_q[q][0].append(b)
            results_by_q[q][1].append(p)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_eval_item, t) for t in tasks]
            for fut in as_completed(futures):
                q, b, p = fut.result()
                results_by_q[q][0].append(b)
                results_by_q[q][1].append(p)

    mean_bpp = [float(np.mean(results_by_q[q][0])) for q in QUALITY_GRID]
    mean_psnr = [float(np.mean(results_by_q[q][1])) for q in QUALITY_GRID]
    return mean_bpp, mean_psnr


def run_tuning(workers: int = 4) -> pd.DataFrame:
    """Run four-stage controlled tuning study."""
    images = get_stratified_60_images()
    print(f"Loaded {len(images)} stratified masked images for DWT tuning.")

    records = []

    # 1. Baseline Reference: Legacy Default
    print("\n--- Measuring Baseline Reference (Legacy DWT: bior4.4, L4, theta=0.0, bias=0.25, legacy=True) ---")
    ref_config = {"legacy": True}
    ref_bpp, ref_psnr = evaluate_curve(images, ref_config, workers=workers)
    records.append({
        "group": "baseline",
        "configuration": "Legacy DWT (bior4.4, L4, theta=0, bias=0.25, legacy=True)",
        "bd_psnr_vs_legacy": 0.0,
        "mean_bpp_q50": ref_bpp[QUALITY_GRID.index(50)],
        "mean_psnr_q50": ref_psnr[QUALITY_GRID.index(50)],
    })

    # Group a: Subband weights
    print("\n--- Group a: Subband Weights ---")
    weight_schemes = ["default", "uniform", "fine_1.15", "fine_0.9"]
    best_weight = "default"
    best_weight_bd = -999.0

    for ws in weight_schemes:
        cfg = {"wavelet": "bior4.4", "levels": 4, "theta": 0.0, "bias": 0.25, "weight_scheme": ws, "context_mode": "none"}
        bpp_c, psnr_c = evaluate_curve(images, cfg, workers=workers)
        bd = bd_psnr(ref_bpp, ref_psnr, bpp_c, psnr_c)
        print(f"  Weight scheme '{ws}': BD-PSNR = {bd:+.4f} dB")
        records.append({
            "group": "a_weights",
            "configuration": f"weight_scheme={ws}",
            "bd_psnr_vs_legacy": round(bd, 4),
            "mean_bpp_q50": round(bpp_c[QUALITY_GRID.index(50)], 4),
            "mean_psnr_q50": round(psnr_c[QUALITY_GRID.index(50)], 2),
        })
        if bd > best_weight_bd:
            best_weight_bd = bd
            best_weight = ws

    print(f"-> Winner Group a: {best_weight} ({best_weight_bd:+.4f} dB)")

    # Group b: Dead-zone theta and reconstruction bias
    print(f"\n--- Group b: Dead-zone Theta & Bias (with weight_scheme={best_weight}) ---")
    theta_candidates = [0.5, 0.4, 0.33, 0.25]
    bias_candidates = [0.0, 0.1, 0.2, 0.3]
    best_theta = 0.0
    best_bias = 0.25
    best_tb_bd = best_weight_bd

    for th in theta_candidates:
        for bi in bias_candidates:
            cfg = {"wavelet": "bior4.4", "levels": 4, "theta": th, "bias": bi, "weight_scheme": best_weight, "context_mode": "none"}
            bpp_c, psnr_c = evaluate_curve(images, cfg, workers=workers)
            bd = bd_psnr(ref_bpp, ref_psnr, bpp_c, psnr_c)
            print(f"  theta={th:.2f}, bias={bi:.2f}: BD-PSNR = {bd:+.4f} dB")
            records.append({
                "group": "b_deadzone",
                "configuration": f"theta={th:.2f}, bias={bi:.2f} (weights={best_weight})",
                "bd_psnr_vs_legacy": round(bd, 4),
                "mean_bpp_q50": round(bpp_c[QUALITY_GRID.index(50)], 4),
                "mean_psnr_q50": round(psnr_c[QUALITY_GRID.index(50)], 2),
            })
            if bd > best_tb_bd:
                best_tb_bd = bd
                best_theta = th
                best_bias = bi

    print(f"-> Winner Group b: theta={best_theta}, bias={best_bias} ({best_tb_bd:+.4f} dB)")

    # Group c: Context-based entropy coding
    print(f"\n--- Group c: Context-based Entropy Coding (theta={best_theta}, bias={best_bias}, weights={best_weight}) ---")
    context_options = ["none", "run", "direct"]
    best_ctx = "none"
    best_ctx_bd = best_tb_bd

    for cm in context_options:
        cfg = {"wavelet": "bior4.4", "levels": 4, "theta": best_theta, "bias": best_bias, "weight_scheme": best_weight, "context_mode": cm}
        bpp_c, psnr_c = evaluate_curve(images, cfg, workers=workers)
        bd = bd_psnr(ref_bpp, ref_psnr, bpp_c, psnr_c)
        print(f"  context_mode='{cm}': BD-PSNR = {bd:+.4f} dB")
        records.append({
            "group": "c_context",
            "configuration": f"context_mode={cm}",
            "bd_psnr_vs_legacy": round(bd, 4),
            "mean_bpp_q50": round(bpp_c[QUALITY_GRID.index(50)], 4),
            "mean_psnr_q50": round(psnr_c[QUALITY_GRID.index(50)], 2),
        })
        if bd > best_ctx_bd:
            best_ctx_bd = bd
            best_ctx = cm

    print(f"-> Winner Group c: context_mode='{best_ctx}' ({best_ctx_bd:+.4f} dB)")

    # Group d: Wavelets and levels
    print(f"\n--- Group d: Wavelet & Levels (with theta={best_theta}, bias={best_bias}, weights={best_weight}, context={best_ctx}) ---")
    wavelets = ["bior2.2", "bior4.4", "db2"]
    levels_list = [3, 4, 5]
    best_wavelet = "bior4.4"
    best_lvl = 4
    best_overall_bd = -999.0

    group_d_results = []
    best_curve_bpp = []
    best_curve_psnr = []

    for w in wavelets:
        for lvl in levels_list:
            cfg = {"wavelet": w, "levels": lvl, "theta": best_theta, "bias": best_bias, "weight_scheme": best_weight, "context_mode": best_ctx}
            bpp_c, psnr_c = evaluate_curve(images, cfg, workers=workers)
            bd = bd_psnr(ref_bpp, ref_psnr, bpp_c, psnr_c)
            print(f"  wavelet={w}, levels={lvl}: BD-PSNR vs legacy = {bd:+.4f} dB")
            rec = {
                "group": "d_wavelet_levels",
                "configuration": f"{w} L{lvl}",
                "bd_psnr_vs_legacy": round(bd, 4),
                "mean_bpp_q50": round(bpp_c[QUALITY_GRID.index(50)], 4),
                "mean_psnr_q50": round(psnr_c[QUALITY_GRID.index(50)], 2),
            }
            records.append(rec)
            group_d_results.append((bd, w, lvl, bpp_c, psnr_c))
            if bd > best_overall_bd:
                best_overall_bd = bd
                best_wavelet = w
                best_lvl = lvl
                best_curve_bpp = bpp_c
                best_curve_psnr = psnr_c

    # Rank group d
    group_d_results.sort(key=lambda x: x[0], reverse=True)
    print("\n--- Group d Ranking (BD-PSNR vs Legacy) ---")
    for rank, (bd, w, lvl, _, _) in enumerate(group_d_results, 1):
        print(f"  Rank {rank}: {w} L{lvl} -> {bd:+.4f} dB")

    # Comparisons against JPEG2000 and DCT
    print("\n--- Benchmarking Best Tuned DWT against JPEG2000 and DCT ---")
    # Load baselines and DCT summaries for comparison
    base_summary = pd.read_csv(RESULTS_DIR / "baseline_summary.csv")
    j2k_sub = base_summary[(base_summary["source"] == "all") & (base_summary["codec"] == "JPEG2000")].sort_values("bpp")
    j2k_bpp = j2k_sub["bpp"].tolist()
    j2k_psnr = j2k_sub["psnr_full"].tolist()

    dct_summary = pd.read_csv(RESULTS_DIR / "dct_summary.csv")
    dct_sub = dct_summary[dct_summary["source"] == "all"].sort_values("bpp")
    dct_bpp = dct_sub["bpp"].tolist()
    dct_psnr = dct_sub["psnr_full"].tolist()

    bd_vs_j2k = bd_psnr(j2k_bpp, j2k_psnr, best_curve_bpp, best_curve_psnr)
    bd_vs_dct = bd_psnr(dct_bpp, dct_psnr, best_curve_bpp, best_curve_psnr)

    print(f"Best Tuned DWT ({best_wavelet} L{best_lvl}, theta={best_theta}, bias={best_bias}, weights={best_weight}, context={best_ctx}):")
    print(f"  BD-PSNR vs Legacy DWT : {best_overall_bd:+.4f} dB")
    print(f"  BD-PSNR vs Custom DCT : {bd_vs_dct:+.4f} dB")
    print(f"  BD-PSNR vs JPEG2000   : {bd_vs_j2k:+.4f} dB")

    records.append({
        "group": "final_benchmarks",
        "configuration": f"Best Tuned DWT vs Custom DCT",
        "bd_psnr_vs_legacy": round(bd_vs_dct, 4),
        "mean_bpp_q50": round(best_curve_bpp[QUALITY_GRID.index(50)], 4),
        "mean_psnr_q50": round(best_curve_psnr[QUALITY_GRID.index(50)], 2),
    })
    records.append({
        "group": "final_benchmarks",
        "configuration": f"Best Tuned DWT vs JPEG2000",
        "bd_psnr_vs_legacy": round(bd_vs_j2k, 4),
        "mean_bpp_q50": round(best_curve_bpp[QUALITY_GRID.index(50)], 4),
        "mean_psnr_q50": round(best_curve_psnr[QUALITY_GRID.index(50)], 2),
    })

    df_out = pd.DataFrame(records)
    df_out.to_csv(TUNING_SUMMARY_CSV, index=False)
    print(f"\nSaved tuning summary table to {TUNING_SUMMARY_CSV}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run DWT tuning study.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    run_tuning(workers=args.workers)


if __name__ == "__main__":
    main()
