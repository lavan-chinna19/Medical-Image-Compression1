"""Summarize baseline compression performance overall and per source."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import RESULTS_DIR

BASELINES_CSV = RESULTS_DIR / "baselines.csv"
SUMMARY_CSV = RESULTS_DIR / "baseline_summary.csv"


def parse_numeric(series: pd.Series) -> pd.Series:
    """Safely convert strings with 'inf' and empty values to float."""
    return pd.to_numeric(series.replace("inf", np.inf), errors="coerce")


def compute_group_summary(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Compute mean metrics for each (codec, setting) pair within a dataset subset."""
    # Ensure numeric columns
    numeric_df = df.copy()
    num_cols = ["bpp", "compression_ratio", "psnr_full", "ssim_full", "psnr_roi", "psnr_background"]
    for c in num_cols:
        numeric_df[c] = parse_numeric(numeric_df[c])

    # Convert roi_lossless to boolean / NaN
    numeric_df["roi_lossless_bool"] = numeric_df["roi_lossless"].apply(
        lambda x: True if str(x).strip().lower() in ("true", "1") else (False if str(x).strip().lower() in ("false", "0") else np.nan)
    )

    rows = []
    # Preserve logical ordering: PNG, JPEG (10..90), JPEG2000 (5..80)
    codecs_order = ["PNG", "JPEG", "JPEG2000"]

    for codec in codecs_order:
        c_sub = numeric_df[numeric_df["codec"] == codec]
        if c_sub.empty:
            continue

        # Sort settings numerically if possible
        settings = c_sub["setting"].unique().tolist()
        try:
            settings = sorted(settings, key=lambda s: float(s))
        except ValueError:
            settings = sorted(settings)

        for setting in settings:
            s_sub = c_sub[c_sub["setting"] == setting]

            # Compute means
            mean_bpp = s_sub["bpp"].mean()
            mean_cr = s_sub["compression_ratio"].mean()
            mean_ssim = s_sub["ssim_full"].mean()

            # PSNR means: if any is inf, mean of inf is inf
            mean_psnr_full = np.inf if (s_sub["psnr_full"] == np.inf).any() else s_sub["psnr_full"].mean()
            
            # Masked metrics on masked images
            masked_sub = s_sub[s_sub["has_mask"] == True]
            if not masked_sub.empty:
                mean_psnr_roi = np.inf if (masked_sub["psnr_roi"] == np.inf).any() else masked_sub["psnr_roi"].mean()
                mean_psnr_bg = np.inf if (masked_sub["psnr_background"] == np.inf).any() else masked_sub["psnr_background"].mean()
                lossless_frac = masked_sub["roi_lossless_bool"].mean()
            else:
                mean_psnr_roi = np.nan
                mean_psnr_bg = np.nan
                lossless_frac = np.nan

            rows.append({
                "source": source_label,
                "codec": codec,
                "setting": str(setting),
                "bpp": round(mean_bpp, 4),
                "compression_ratio": round(mean_cr, 3),
                "psnr_full": round(mean_psnr_full, 2) if mean_psnr_full != np.inf else "inf",
                "ssim_full": round(mean_ssim, 4),
                "psnr_roi": round(mean_psnr_roi, 2) if (mean_psnr_roi != np.inf and not np.isnan(mean_psnr_roi)) else ("inf" if mean_psnr_roi == np.inf else ""),
                "psnr_background": round(mean_psnr_bg, 2) if (mean_psnr_bg != np.inf and not np.isnan(mean_psnr_bg)) else ("inf" if mean_psnr_bg == np.inf else ""),
                "roi_lossless_fraction": round(lossless_frac, 4) if not np.isnan(lossless_frac) else "",
            })

    return pd.DataFrame(rows)


def summarize_baselines() -> pd.DataFrame:
    """Compute overall and per-source baseline summary tables."""
    assert BASELINES_CSV.is_file(), f"Baselines CSV not found: {BASELINES_CSV}. Run run_baselines.py first."

    df = pd.read_csv(BASELINES_CSV)
    print(f"Loaded {len(df)} rows from {BASELINES_CSV}")

    df_all = compute_group_summary(df, source_label="all")
    df_mcu = compute_group_summary(df[df["source"] == "Montgomery"], source_label="montgomery")
    df_chn = compute_group_summary(df[df["source"] == "Shenzhen"], source_label="shenzhen")

    summary_df = pd.concat([df_all, df_mcu, df_chn], ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved baseline summary table to {SUMMARY_CSV}")

    print("\n" + "=" * 90)
    print("                      OVERALL BASELINE SUMMARY (source='all')")
    print("=" * 90)
    print(df_all.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df


if __name__ == "__main__":
    summarize_baselines()
