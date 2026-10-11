"""Review predicted masks and uncertainty maps on images without ground-truth masks.

Generates:
- results/mask_review_unmasked.png: 24 sampled images showing original image with red outline
  and uncertainty map side-by-side, with stems.
- Prints statistical summary over all 96 unmasked images:
  - Mean area fraction
  - Connected components statistics
  - List of outliers (< 0.15 or > 0.50 area fraction).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, RESULTS_DIR, WORK_DIR
from medcomp.io_utils import load_image

PRED_DIR = WORK_DIR / "pred"
MASK_DIR = PRED_DIR / "mask"
UNC_DIR = PRED_DIR / "unc"


def analyze_unmasked_images(
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, dict[int, int], list[dict]]:
    """Compute mask area fraction and connected component counts for all 96 unmasked images."""
    unmasked = manifest_df[manifest_df["has_mask"] == 0].copy().reset_index(drop=True)
    assert len(unmasked) == 96, f"Expected 96 unmasked images, found {len(unmasked)}"

    records = []
    outliers = []
    cc_counts = {}

    for _, row in unmasked.iterrows():
        stem = str(row["stem"])
        mask_path = MASK_DIR / f"{stem}.png"
        assert mask_path.is_file(), f"Mask not found: {mask_path}"
        mask = load_image(mask_path)

        binary = (mask > 128).astype(np.uint8)
        area_frac = float(np.sum(binary) / binary.size)

        num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
        num_fg_cc = max(0, num_labels - 1)

        cc_counts[num_fg_cc] = cc_counts.get(num_fg_cc, 0) + 1

        rec = {
            "stem": stem,
            "source": row["source"],
            "label": int(row["label"]),
            "area_fraction": area_frac,
            "num_cc": num_fg_cc,
        }
        records.append(rec)

        if area_frac < 0.15 or area_frac > 0.50:
            outliers.append(rec)

    df_stats = pd.DataFrame(records)
    mean_area_frac = float(df_stats["area_fraction"].mean())

    return df_stats, mean_area_frac, cc_counts, outliers


def generate_grid_plot(
    df_stats: pd.DataFrame,
    save_path: Path,
    n_samples: int = 24,
    seed: int = 42,
) -> None:
    """Generate 6x4 pair grid (6 rows, 8 columns) showing image with red mask outline and uncertainty map."""
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(df_stats), size=n_samples, replace=False)
    sample_df = df_stats.iloc[chosen_indices].reset_index(drop=True)

    # 6 rows, 4 samples per row -> 8 subplots per row (Image+Outline, Uncertainty)
    n_rows = 6
    samples_per_row = 4
    n_cols = samples_per_row * 2  # 8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 17))
    plt.subplots_adjust(wspace=0.15, hspace=0.35, top=0.96, bottom=0.03, left=0.03, right=0.97)

    fig.suptitle(
        f"Unmasked Images Mask Quality Review (24 / 96 Seeded Samples)\n"
        f"Pairs: [Left: X-Ray with Predicted Mask Contour (Red)]  |  [Right: MC-Dropout Uncertainty (Std Dev)]",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    for idx, row in sample_df.iterrows():
        r = idx // samples_per_row
        pair_col = (idx % samples_per_row) * 2

        ax_img = axes[r, pair_col]
        ax_unc = axes[r, pair_col + 1]

        stem = row["stem"]
        area_frac = row["area_fraction"]
        num_cc = row["num_cc"]

        # Load original image, mask, and uncertainty
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = MASK_DIR / f"{stem}.png"
        unc_path = UNC_DIR / f"{stem}.png"

        img = load_image(img_path)
        mask = load_image(mask_path)
        unc = load_image(unc_path)

        # Convert image to RGB and draw red contour
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_img, contours, -1, (255, 30, 30), thickness=2)

        ax_img.imshow(rgb_img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title("Image + Mask", fontsize=9, pad=3)
        ax_img.set_xlabel(f"{stem}\nArea: {area_frac:.1%} | CC: {num_cc}", fontsize=9, labelpad=2)

        # Uncertainty map with colormap
        im_unc = ax_unc.imshow(unc, cmap="magma")
        ax_unc.set_xticks([])
        ax_unc.set_yticks([])
        ax_unc.set_title("Uncertainty", fontsize=9, pad=3)
        ax_unc.set_xlabel(f"Max \u03c3: {unc.max()/255.0:.2f}", fontsize=9, labelpad=2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved review grid to {save_path}")


def main() -> None:
    manifest_csv = DATA_DIR / "manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)

    df_stats, mean_area_frac, cc_counts, outliers = analyze_unmasked_images(manifest_df)

    out_png = RESULTS_DIR / "mask_review_unmasked.png"
    generate_grid_plot(df_stats, out_png, n_samples=24, seed=42)

    print("\n" + "=" * 70)
    print("UNMASKED IMAGES PREDICTED MASK EVALUATION (96 IMAGES)")
    print("=" * 70)
    print(f"Total unmasked images analyzed: {len(df_stats)}")
    print(f"Mean predicted-mask area fraction: {mean_area_frac:.4f} ({mean_area_frac*100:.2f}%)")
    print(f"Area fraction min: {df_stats['area_fraction'].min():.4f}, max: {df_stats['area_fraction'].max():.4f}")
    print("\nConnected components distribution across all 96 images:")
    for n_cc, count in sorted(cc_counts.items()):
        pct = (count / len(df_stats)) * 100.0
        print(f"  {n_cc} component(s): {count} images ({pct:.1f}%)")

    print("\nImages with area fraction < 0.15 or > 0.50:")
    if outliers:
        for out in outliers:
            print(f"  - Stem: {out['stem']:<16} | Source: {out['source']} | Area Fraction: {out['area_fraction']:.4f} ({out['area_fraction']*100:.2f}%) | CC: {out['num_cc']}")
    else:
        print("  None! All 96 images fall strictly within [0.15, 0.50].")
    print("=" * 70)


if __name__ == "__main__":
    main()
