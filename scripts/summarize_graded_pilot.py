"""Summarize graded near-lossless compression pilot results.

Generates:
1. results/graded_pilot_summary.csv:
   - Variant means for bpp, byte breakdowns, flip rate, confident flip rate,
     mean abs prob diff, 99th-percentile lung error, max lung error,
     fractions with error > 2 and > 4, lung fraction outside core+band, psnr_lung_true.
   - 95% bootstrap confidence intervals (1,000 cluster resamples over images, fixed seed).
2. results/graded_pilot_guarantee_cost.csv:
   - For each family: bpp required to achieve mean 99th-percentile lung error <= 2, 4, 8 gray levels
     (log-bpp interpolation) and flip rate <= 5%.
3. Prints Pareto-optimal variants for (bpp, mean_abs_prob_diff) and (bpp, p99_lung_err).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

PILOT_CSV = RESULTS_DIR / "graded_pilot.csv"
SUMMARY_CSV = RESULTS_DIR / "graded_pilot_summary.csv"
GUARANTEE_COST_CSV = RESULTS_DIR / "graded_pilot_guarantee_cost.csv"


def compute_bootstrap_cis(
    df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute 95% bootstrap confidence intervals for flip rate and mean abs_prob_diff.

    Resamples patient image stems with replacement.
    """
    rng = np.random.RandomState(seed)
    stems = df["stem"].unique()
    n_stems = len(stems)

    # Pre-index df by stem
    stem_groups = {s: g for s, g in df.groupby("stem")}

    variants = df["variant"].unique()
    boot_flips: dict[str, list[float]] = {v: [] for v in variants}
    boot_diffs: dict[str, list[float]] = {v: [] for v in variants}

    for _ in range(n_bootstraps):
        resampled_stems = rng.choice(stems, size=n_stems, replace=True)
        resampled_df = pd.concat([stem_groups[s] for s in resampled_stems], ignore_index=True)

        grouped = resampled_df.groupby("variant")
        flips = grouped["flip"].mean()
        diffs = grouped["abs_prob_diff"].mean()

        for v in variants:
            boot_flips[v].append(flips.get(v, np.nan))
            boot_diffs[v].append(diffs.get(v, np.nan))

    ci_dict: dict[str, dict[str, tuple[float, float]]] = {}
    for v in variants:
        f_vals = np.array(boot_flips[v])
        d_vals = np.array(boot_diffs[v])
        ci_dict[v] = {
            "flip": (float(np.nanpercentile(f_vals, 2.5)), float(np.nanpercentile(f_vals, 97.5))),
            "diff": (float(np.nanpercentile(d_vals, 2.5)), float(np.nanpercentile(d_vals, 97.5))),
        }

    return ci_dict


def interpolate_cost(
    points: list[tuple[float, float]],
    target: float,
) -> str:
    """Interpolate log(bpp) to find the bitrate where metric <= target.

    points: list of (bpp, metric_value) sorted by bpp ascending.
    """
    if not points:
        return "not reached"

    # If already <= target at the lowest bpp
    if points[0][1] <= target:
        return f"<={points[0][0]:.3f}"

    # If never <= target even at the highest bpp
    if points[-1][1] > target:
        return "not reached"

    # Search for crossing interval
    for i in range(len(points) - 1):
        bpp1, val1 = points[i]
        bpp2, val2 = points[i + 1]

        if val1 > target >= val2:
            # Linear interpolation in log(bpp)
            log1, log2 = np.log(bpp1), np.log(bpp2)
            t = (target - val1) / (val2 - val1)
            interp_log = log1 + t * (log2 - log1)
            interp_bpp = float(np.exp(interp_log))
            return f"{interp_bpp:.3f}"

    return "not reached"


def find_pareto_front(
    candidates: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> list[dict[str, Any]]:
    """Return non-dominated candidates minimizing both x_key and y_key."""
    sorted_cand = sorted(candidates, key=lambda c: (c[x_key], c[y_key]))
    pareto: list[dict[str, Any]] = []

    best_y = float("inf")
    for c in sorted_cand:
        if c[y_key] < best_y:
            pareto.append(c)
            best_y = c[y_key]

    return pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize graded near-lossless pilot.")
    args = parser.parse_args()

    if not PILOT_CSV.is_file():
        raise FileNotFoundError(f"Pilot results not found: {PILOT_CSV}")

    df = pd.read_csv(PILOT_CSV)
    print(f"Loaded {len(df)} rows from {PILOT_CSV.name} covering {df['stem'].nunique()} unique images.")

    # 1. Compute bootstrap CIs
    print("Computing 1,000 cluster bootstrap CIs...")
    ci_dict = compute_bootstrap_cis(df, n_bootstraps=1000, seed=42)

    # 2. Confident mask (prob_orig outside [0.3, 0.7])
    df["is_confident"] = (df["prob_orig"] < 0.3) | (df["prob_orig"] > 0.7)

    # 3. Group by variant
    summary_rows: list[dict[str, Any]] = []
    variants = df["variant"].unique()

    for v in variants:
        v_df = df[df["variant"] == v]
        first_row = v_df.iloc[0]

        # Masked images subset
        masked_v_df = v_df[v_df["has_true_mask"] == 1]
        confident_v_df = v_df[v_df["is_confident"]]

        bpp_mean = float(v_df["bpp"].mean())
        flip_rate = float(v_df["flip"].mean())
        flip_conf = float(confident_v_df["flip"].mean()) if len(confident_v_df) > 0 else 0.0
        diff_mean = float(v_df["abs_prob_diff"].mean())

        flip_ci = ci_dict[v]["flip"]
        diff_ci = ci_dict[v]["diff"]

        p99_err = float(masked_v_df["p99_err_true_lung"].mean()) if len(masked_v_df) > 0 else np.nan
        max_err = float(masked_v_df["max_err_true_lung"].mean()) if len(masked_v_df) > 0 else np.nan
        frac_gt_2 = float(masked_v_df["frac_err_gt_2"].mean()) if len(masked_v_df) > 0 else np.nan
        frac_gt_4 = float(masked_v_df["frac_err_gt_4"].mean()) if len(masked_v_df) > 0 else np.nan
        frac_outside = float(masked_v_df["frac_lung_in_bg"].mean()) if len(masked_v_df) > 0 else np.nan
        psnr_tl = float(masked_v_df["psnr_lung_true"].mean()) if len(masked_v_df) > 0 else np.nan

        summary_rows.append({
            "family": first_row["family"],
            "variant": v,
            "delta_core": first_row["delta_core"],
            "delta_band": first_row["delta_band"],
            "band_px": first_row["band_px"],
            "bg_quality": first_row["bg_quality"],
            "bpp": round(bpp_mean, 4),
            "bytes_header": int(round(v_df["bytes_header"].mean())),
            "bytes_mask": int(round(v_df["bytes_mask"].mean())),
            "bytes_core": int(round(v_df["bytes_core"].mean())),
            "bytes_band": int(round(v_df["bytes_band"].mean())),
            "bytes_bg": int(round(v_df["bytes_bg"].mean())),
            "total_bytes": int(round(v_df["total_bytes"].mean())),
            "flip_rate": round(flip_rate, 4),
            "flip_rate_ci_lower": round(flip_ci[0], 4),
            "flip_rate_ci_upper": round(flip_ci[1], 4),
            "flip_rate_confident": round(flip_conf, 4),
            "mean_abs_prob_diff": round(diff_mean, 4),
            "mean_abs_prob_diff_ci_lower": round(diff_ci[0], 4),
            "mean_abs_prob_diff_ci_upper": round(diff_ci[1], 4),
            "p99_lung_err": round(p99_err, 2),
            "max_lung_err": round(max_err, 2),
            "frac_lung_gt_2": round(frac_gt_2, 4),
            "frac_lung_gt_4": round(frac_gt_4, 4),
            "frac_lung_outside_roi": round(frac_outside, 4),
            "psnr_lung_true": round(psnr_tl, 2),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {len(sum_df)} variant summaries to {SUMMARY_CSV}")

    # 4. Guarantee cost table
    families = sum_df["family"].unique()
    cost_rows: list[dict[str, Any]] = []

    for fam in families:
        fam_df = sum_df[sum_df["family"] == fam].sort_values("bpp")
        pts_err = list(zip(fam_df["bpp"], fam_df["p99_lung_err"]))
        pts_flip = list(zip(fam_df["bpp"], fam_df["flip_rate"]))

        cost_rows.append({
            "family": fam,
            "bpp_for_p99_le_2": interpolate_cost(pts_err, 2.0),
            "bpp_for_p99_le_4": interpolate_cost(pts_err, 4.0),
            "bpp_for_p99_le_8": interpolate_cost(pts_err, 8.0),
            "bpp_for_flip_le_0.05": interpolate_cost(pts_flip, 0.05),
        })

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(GUARANTEE_COST_CSV, index=False)
    print(f"Saved guarantee cost results to {GUARANTEE_COST_CSV}")

    # 5. Pareto sets
    cand_records = sum_df.to_dict("records")
    pareto_diff = find_pareto_front(cand_records, "bpp", "mean_abs_prob_diff")
    pareto_err = find_pareto_front(cand_records, "bpp", "p99_lung_err")

    print("\n" + "=" * 80)
    print("PARETO SET: (bpp, mean_abs_prob_diff)")
    print("=" * 80)
    for p in pareto_diff:
        print(f"Variant: {p['variant']:<25} | bpp: {p['bpp']:.4f} | diff: {p['mean_abs_prob_diff']:.4f} (CI: [{p['mean_abs_prob_diff_ci_lower']:.4f}, {p['mean_abs_prob_diff_ci_upper']:.4f}]) | flip: {p['flip_rate']*100:.1f}%")

    print("\nTop 3 Pareto variants for (bpp, mean_abs_prob_diff):")
    for i, p in enumerate(pareto_diff[:3], 1):
        print(f"  {i}. {p['variant']} (bpp={p['bpp']:.4f}, diff={p['mean_abs_prob_diff']:.4f})")

    print("\n" + "=" * 80)
    print("PARETO SET: (bpp, 99th-percentile lung error)")
    print("=" * 80)
    for p in pareto_err:
        print(f"Variant: {p['variant']:<25} | bpp: {p['bpp']:.4f} | p99_err: {p['p99_lung_err']:.2f} | max_err: {p['max_lung_err']:.2f} | psnr_lung: {p['psnr_lung_true']:.2f} dB")

    print("\nTop 3 Pareto variants for (bpp, 99th-percentile lung error):")
    for i, p in enumerate(pareto_err[:3], 1):
        print(f"  {i}. {p['variant']} (bpp={p['bpp']:.4f}, p99_err={p['p99_lung_err']:.2f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
