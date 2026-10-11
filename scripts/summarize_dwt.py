"""Summarize custom DWT codec benchmark results overall and per source."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

DWT_RESULTS_CSV = RESULTS_DIR / "dwt_results.csv"
DWT_SUMMARY_CSV = RESULTS_DIR / "dwt_summary.csv"


def parse_numeric(series: pd.Series) -> pd.Series:
    """Safely convert strings with 'inf' and empty values to float."""
    return pd.to_numeric(series.replace("inf", np.inf), errors="coerce")


def compute_dwt_group_summary(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Compute mean metrics for each quality setting within a dataset subset."""
    numeric_df = df.copy()
    num_cols = [
        "bpp", "compression_ratio", "psnr_full", "ssim_full",
        "psnr_roi", "psnr_background", "ssim_roi", "ssim_background",
        "ll_entropy", "ll_avg_code_length",
        "detail_entropy", "detail_avg_code_length", "encode_time_s", "decode_time_s",
    ]
    for c in num_cols:
        if c in numeric_df.columns:
            numeric_df[c] = parse_numeric(numeric_df[c])

    numeric_df["roi_lossless_bool"] = numeric_df["roi_lossless"].apply(
        lambda x: True if str(x).strip().lower() in ("true", "1") else (False if str(x).strip().lower() in ("false", "0") else np.nan)
    )

    rows = []
    qualities = sorted(numeric_df["setting"].unique(), key=lambda q: int(q))

    for q in qualities:
        q_sub = numeric_df[numeric_df["setting"] == q]

        mean_bpp = q_sub["bpp"].mean()
        mean_cr = q_sub["compression_ratio"].mean()
        mean_psnr = q_sub["psnr_full"].mean()
        mean_ssim = q_sub["ssim_full"].mean()

        masked_sub = q_sub[q_sub["has_mask"] == True]
        if not masked_sub.empty:
            mean_psnr_roi = masked_sub["psnr_roi"].mean()
            mean_psnr_bg = masked_sub["psnr_background"].mean()
            mean_ssim_roi = masked_sub["ssim_roi"].mean() if "ssim_roi" in masked_sub.columns else np.nan
            mean_ssim_bg = masked_sub["ssim_background"].mean() if "ssim_background" in masked_sub.columns else np.nan
            lossless_frac = masked_sub["roi_lossless_bool"].mean()
        else:
            mean_psnr_roi = np.nan
            mean_psnr_bg = np.nan
            mean_ssim_roi = np.nan
            mean_ssim_bg = np.nan
            lossless_frac = np.nan

        mean_ll_ent = q_sub["ll_entropy"].mean()
        mean_ll_len = q_sub["ll_avg_code_length"].mean()
        mean_det_ent = q_sub["detail_entropy"].mean()
        mean_det_len = q_sub["detail_avg_code_length"].mean()
        mean_enc_time = q_sub["encode_time_s"].mean()
        mean_dec_time = q_sub["decode_time_s"].mean()

        rows.append({
            "source": source_label,
            "quality": int(q),
            "bpp": round(mean_bpp, 4),
            "compression_ratio": round(mean_cr, 3),
            "psnr_full": round(mean_psnr, 2),
            "ssim_full": round(mean_ssim, 4),
            "psnr_roi": round(mean_psnr_roi, 2) if not np.isnan(mean_psnr_roi) else "",
            "psnr_background": round(mean_psnr_bg, 2) if not np.isnan(mean_psnr_bg) else "",
            "ssim_roi": round(mean_ssim_roi, 4) if not np.isnan(mean_ssim_roi) else "",
            "ssim_background": round(mean_ssim_bg, 4) if not np.isnan(mean_ssim_bg) else "",
            "roi_lossless_fraction": round(lossless_frac, 4) if not np.isnan(lossless_frac) else "",
            "ll_entropy": round(mean_ll_ent, 4),
            "ll_avg_code_length": round(mean_ll_len, 4),
            "detail_entropy": round(mean_det_ent, 4),
            "detail_avg_code_length": round(mean_det_len, 4),
            "encode_time_s": round(mean_enc_time, 4),
            "decode_time_s": round(mean_dec_time, 4),
        })

    return pd.DataFrame(rows)


def summarize_dwt() -> pd.DataFrame:
    """Compute overall and per-source DWT summary tables."""
    assert DWT_RESULTS_CSV.is_file(), f"DWT results CSV not found: {DWT_RESULTS_CSV}"

    df = pd.read_csv(DWT_RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {DWT_RESULTS_CSV}")

    df_all = compute_dwt_group_summary(df, source_label="all")
    df_mcu = compute_dwt_group_summary(df[df["source"] == "Montgomery"], source_label="montgomery")
    df_chn = compute_dwt_group_summary(df[df["source"] == "Shenzhen"], source_label="shenzhen")

    summary_df = pd.concat([df_all, df_mcu, df_chn], ignore_index=True)
    summary_df.to_csv(DWT_SUMMARY_CSV, index=False)
    print(f"Saved DWT summary table to {DWT_SUMMARY_CSV}")

    print("\n" + "=" * 105)
    print("                           OVERALL CUSTOM DWT SUMMARY (source='all')")
    print("=" * 105)
    print(df_all.to_string(index=False))
    print("=" * 105 + "\n")

    return summary_df


if __name__ == "__main__":
    summarize_dwt()
