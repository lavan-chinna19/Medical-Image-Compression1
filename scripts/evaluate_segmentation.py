"""Evaluate out-of-fold U-Net lung segmentation predictions, margin analysis, and plots."""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import (
    DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RESULTS_DIR,
    WORK_DIR,
)
from medcomp.io_utils import load_image

PRED_DIR = WORK_DIR / "pred"
PROB_DIR = PRED_DIR / "prob"
UNC_DIR = PRED_DIR / "unc"
MASK_DIR = PRED_DIR / "mask"

SEG_RESULTS_CSV = RESULTS_DIR / "segmentation_results.csv"
SEG_SUMMARY_CSV = RESULTS_DIR / "segmentation_summary.csv"
SEG_MARGINS_CSV = RESULTS_DIR / "segmentation_margins.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute Dice, IoU, Lung Recall, and Precision between two boolean masks."""
    p = pred.astype(bool)
    t = target.astype(bool)

    intersection = int(np.sum(p & t))
    p_sum = int(np.sum(p))
    t_sum = int(np.sum(t))

    dice = (2.0 * intersection) / max(1, p_sum + t_sum)
    union = p_sum + t_sum - intersection
    iou = intersection / max(1, union)
    recall = intersection / max(1, t_sum)
    precision = intersection / max(1, p_sum)
    area_ratio = p_sum / max(1, t_sum)

    return {
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "precision": precision,
        "area_ratio": area_ratio,
    }


def dilate_mask(mask: np.ndarray, d_pixels: int) -> np.ndarray:
    """Dilate binary mask by radius d_pixels."""
    if d_pixels <= 0:
        return mask
    k_size = 2 * d_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def run_evaluation() -> None:
    """Perform comprehensive segmentation evaluation and margin analysis."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    masked_df = df[df["has_mask"] == True].copy()

    results: list[dict[str, Any]] = []
    margin_records: list[dict[str, Any]] = []

    print(f"Evaluating out-of-fold predictions for {len(masked_df)} masked images...")

    for _, row in tqdm(masked_df.iterrows(), total=len(masked_df), desc="Evaluating Segmentation"):
        stem = str(row["stem"])
        source = str(row["source"])
        fold = int(row["fold"])
        label = int(row["label"])

        true_mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
        pred_mask_path = MASK_DIR / f"{stem}.png"
        unc_path = UNC_DIR / f"{stem}.png"

        assert true_mask_path.is_file(), f"True mask missing: {true_mask_path}"
        assert pred_mask_path.is_file(), f"Predicted mask missing: {pred_mask_path}"

        true_mask = (load_image(true_mask_path) > 127)
        pred_mask = (load_image(pred_mask_path) > 127)
        unc_map = load_image(unc_path).astype(np.float32) / 255.0

        # Baseline metrics
        m = compute_metrics(pred_mask, true_mask)
        results.append({
            "stem": stem,
            "source": source,
            "fold": fold,
            "label": label,
            "dice": round(m["dice"], 4),
            "iou": round(m["iou"], 4),
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "area_ratio": round(m["area_ratio"], 4),
        })

        # Dilation margins d in [0, 2, 4, 8]
        for d in [0, 2, 4, 8]:
            d_mask = dilate_mask(pred_mask, d)
            dm = compute_metrics(d_mask, true_mask)
            margin_records.append({
                "stem": stem,
                "source": source,
                "margin_type": f"dilation_{d}px",
                "dice": dm["dice"],
                "recall": dm["recall"],
                "area_ratio": dm["area_ratio"],
            })

        # Uncertainty margins t in [0.05, 0.1, 0.2]
        for t in [0.05, 0.1, 0.2]:
            t_mask = pred_mask | (unc_map > t)
            tm = compute_metrics(t_mask, true_mask)
            margin_records.append({
                "stem": stem,
                "source": source,
                "margin_type": f"uncertainty_t{t}",
                "dice": tm["dice"],
                "recall": tm["recall"],
                "area_ratio": tm["area_ratio"],
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(SEG_RESULTS_CSV, index=False)
    print(f"Saved image-level segmentation results to {SEG_RESULTS_CSV}")

    # Summary table: Mean, Std, 5th percentile overall, per source, and per fold
    summary_rows = []
    subsets = (
        [("overall", "all", res_df)]
        + [("source", s, res_df[res_df["source"] == s]) for s in sorted(res_df["source"].unique())]
        + [("fold", f"fold_{f}", res_df[res_df["fold"] == f]) for f in sorted(res_df["fold"].unique())]
    )

    for group_type, group_name, sub in subsets:
        summary_rows.append({
            "group_type": group_type,
            "group_name": group_name,
            "count": len(sub),
            "dice_mean": round(float(sub["dice"].mean()), 4),
            "dice_std": round(float(sub["dice"].std()), 4),
            "dice_p5": round(float(np.percentile(sub["dice"], 5)), 4),
            "iou_mean": round(float(sub["iou"].mean()), 4),
            "iou_std": round(float(sub["iou"].std()), 4),
            "iou_p5": round(float(np.percentile(sub["iou"], 5)), 4),
            "recall_mean": round(float(sub["recall"].mean()), 4),
            "recall_std": round(float(sub["recall"].std()), 4),
            "recall_p5": round(float(np.percentile(sub["recall"], 5)), 4),
            "precision_mean": round(float(sub["precision"].mean()), 4),
            "precision_std": round(float(sub["precision"].std()), 4),
            "precision_p5": round(float(np.percentile(sub["precision"], 5)), 4),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(SEG_SUMMARY_CSV, index=False)
    print(f"Saved segmentation summary table to {SEG_SUMMARY_CSV}")

    # Margin summary
    m_df = pd.DataFrame(margin_records)
    margin_summary_rows = []
    m_subsets = [("all", m_df)] + [(s, m_df[m_df["source"] == s]) for s in sorted(m_df["source"].unique())]

    for src, sub in m_subsets:
        for m_type in sorted(sub["margin_type"].unique()):
            m_sub = sub[sub["margin_type"] == m_type]
            margin_summary_rows.append({
                "source": src,
                "margin_type": m_type,
                "mean_dice": round(float(m_sub["dice"].mean()), 4),
                "mean_recall": round(float(m_sub["recall"].mean()), 4),
                "mean_area_ratio": round(float(m_sub["area_ratio"].mean()), 4),
            })

    margin_sum_df = pd.DataFrame(margin_summary_rows)
    margin_sum_df.to_csv(SEG_MARGINS_CSV, index=False)
    print(f"Saved margin analysis table to {SEG_MARGINS_CSV}")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_dice_histogram(res_df, PLOTS_DIR / "seg_dice_histogram_by_source.png")
    plot_recall_vs_dilation(m_df, PLOTS_DIR / "seg_recall_vs_dilation.png")
    plot_overlays(res_df, PLOTS_DIR / "seg_overlay_samples.png")
    plot_worst_cases(res_df, PLOTS_DIR / "seg_worst_dice_uncertainty.png")


def plot_dice_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Plot Dice distribution histogram by source dataset."""
    plt.figure(figsize=(8, 5.5), dpi=300)
    for src, col in [("Montgomery", "#1f77b4"), ("Shenzhen", "#ff7f0e")]:
        sub = df[df["source"] == src]["dice"]
        plt.hist(sub, bins=30, range=(0.7, 1.0), alpha=0.6, color=col, label=f"{src} (Mean: {sub.mean():.4f}, P5: {np.percentile(sub, 5):.4f})")

    plt.xlabel("Dice Coefficient", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Cases", fontsize=11, fontweight="bold")
    plt.title("Out-of-Fold U-Net Lung Segmentation Dice Distribution", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Dice histogram to {out_path}")


def plot_recall_vs_dilation(m_df: pd.DataFrame, out_path: Path) -> None:
    """Plot Lung Recall and Area Ratio vs Dilation radius."""
    dil_types = ["dilation_0px", "dilation_2px", "dilation_4px", "dilation_8px"]
    radii = [0, 2, 4, 8]

    plt.figure(figsize=(8, 5.5), dpi=300)
    for src, col in [("Montgomery", "#1f77b4"), ("Shenzhen", "#ff7f0e")]:
        recalls = []
        for dt in dil_types:
            val = m_df[(m_df["source"] == src) & (m_df["margin_type"] == dt)]["recall"].mean()
            recalls.append(val * 100.0)
        plt.plot(radii, recalls, marker="o", lw=2.2, color=col, label=f"{src} Lung Recall (%)")

    plt.xlabel("Boundary Dilation Radius (pixels)", fontsize=11, fontweight="bold")
    plt.ylabel("Lung Recall (%)", fontsize=11, fontweight="bold")
    plt.title("Lung Recall vs Boundary Dilation Radius", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved recall vs dilation plot to {out_path}")


def plot_overlays(df: pd.DataFrame, out_path: Path) -> None:
    """Generate 6 random overlay samples per source (green true, red predicted)."""
    fig, axes = plt.subplots(4, 3, figsize=(12, 16), dpi=300)
    rng = np.random.default_rng(42)

    stems = []
    for src in ["Montgomery", "Shenzhen"]:
        sub = df[df["source"] == src]
        chosen = rng.choice(sub["stem"].values, size=min(6, len(sub)), replace=False)
        stems.extend([(src, c) for c in chosen])

    for ax_idx, (src, stem) in enumerate(stems):
        r, c = ax_idx // 3, ax_idx % 3
        ax = axes[r, c]

        img = load_image(PROCESSED_IMAGES_DIR / f"{stem}.png")
        true_mask = (load_image(PROCESSED_MASKS_DIR / f"{stem}.png") > 127).astype(np.uint8)
        pred_mask = (load_image(MASK_DIR / f"{stem}.png") > 127).astype(np.uint8)

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Contours
        cnt_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(rgb, cnt_true, -1, (0, 255, 0), 2)  # Green = Ground Truth
        cv2.drawContours(rgb, cnt_pred, -1, (255, 0, 0), 2)  # Red = Prediction

        d_score = df[df["stem"] == stem]["dice"].iloc[0]
        ax.imshow(rgb)
        ax.set_title(f"{src}: {stem}\nDice = {d_score:.4f}", fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Lung Segmentation Outlines: Green = True Mask, Red = Predicted Mask", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved overlay samples to {out_path}")


def plot_worst_cases(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 6 worst-Dice images alongside their prediction and uncertainty maps."""
    worst = df.sort_values("dice").head(6)

    fig, axes = plt.subplots(6, 4, figsize=(14, 18), dpi=300)

    for row_idx, (_, row) in enumerate(worst.iterrows()):
        stem = str(row["stem"])
        d_val = float(row["dice"])

        img = load_image(PROCESSED_IMAGES_DIR / f"{stem}.png")
        true_mask = load_image(PROCESSED_MASKS_DIR / f"{stem}.png")
        pred_mask = load_image(MASK_DIR / f"{stem}.png")
        unc_map = load_image(UNC_DIR / f"{stem}.png")

        axes[row_idx, 0].imshow(img, cmap="gray")
        axes[row_idx, 0].set_title(f"{stem} (Dice={d_val:.4f})", fontsize=8, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(true_mask, cmap="gray")
        axes[row_idx, 1].set_title("Ground Truth", fontsize=8, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_mask, cmap="gray")
        axes[row_idx, 2].set_title("Predicted Mask", fontsize=8, fontweight="bold")
        axes[row_idx, 2].axis("off")

        im = axes[row_idx, 3].imshow(unc_map, cmap="inferno")
        axes[row_idx, 3].set_title("MC Uncertainty", fontsize=8, fontweight="bold")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved worst-case uncertainty visualizations to {out_path}")


if __name__ == "__main__":
    run_evaluation()
