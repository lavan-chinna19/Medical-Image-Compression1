"""Preprocessing pipeline: resizing images, masks, report generation, previews, and contact sheets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import (
    DATA_DIR,
    DATA_ROOT,
    PROJECT_ROOT,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RESULTS_DIR,
    TARGET_LONG_SIDE,
)
from medcomp.data_utils import (
    load_mask,
    resize_long_side,
    resize_mask_nearest,
    roi_fraction,
)
from medcomp.io_utils import load_image, save_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preprocess")


def process_single_item(row_dict: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Process a single image and optional mask.

    Returns record dict for processed_report.csv.
    """
    stem = row_dict["stem"]
    source = row_dict["source"]
    has_mask = bool(row_dict["has_mask"])

    img_rel = row_dict["image_path"]
    img_path = PROJECT_ROOT / img_rel

    out_img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    out_mask_path = PROCESSED_MASKS_DIR / f"{stem}.png" if has_mask else None

    # Load and process image
    if force or not out_img_path.is_file():
        img = load_image(img_path)
        proc_img = resize_long_side(img, target=TARGET_LONG_SIDE)
        save_image(out_img_path, proc_img)
    else:
        proc_img = load_image(out_img_path)

    proc_h, proc_w = proc_img.shape[:2]

    # Process mask if present
    roi_frac_proc = ""
    mask_binary_ok = ""
    empty_mask = ""
    warning_msg = None

    if has_mask:
        mask_rel = row_dict["mask_path"]
        mask_path = PROJECT_ROOT / mask_rel

        if force or not out_mask_path.is_file():
            mask_arr = load_mask(mask_path)
            orig_h = int(row_dict["orig_h"])
            orig_w = int(row_dict["orig_w"])

            # Check if mask shape matches original image shape
            if mask_arr.shape != (orig_h, orig_w):
                warning_msg = (
                    f"Shape mismatch for {stem}: image {(orig_h, orig_w)} vs mask {mask_arr.shape}. "
                    f"Resizing mask to image shape with nearest neighbor."
                )
                mask_arr = resize_mask_nearest(mask_arr, (orig_h, orig_w))

            # Resize to processed size
            proc_mask_bool = resize_mask_nearest(mask_arr, (proc_h, proc_w))
            proc_mask_uint8 = (proc_mask_bool.astype(np.uint8) * 255)
            save_image(out_mask_path, proc_mask_uint8)
        else:
            proc_mask_raw = load_image(out_mask_path)
            proc_mask_bool = (proc_mask_raw > 127)
            proc_mask_uint8 = proc_mask_bool.astype(np.uint8) * 255

        # Sanity checks on mask
        unique_vals = set(np.unique(proc_mask_uint8))
        mask_binary_ok = bool(unique_vals.issubset({0, 255}))
        empty_mask = bool(np.count_nonzero(proc_mask_bool) == 0)
        roi_frac_proc = round(roi_fraction(proc_mask_bool), 6)

    return {
        "stem": stem,
        "source": source,
        "processed_h": proc_h,
        "processed_w": proc_w,
        "has_mask": has_mask,
        "roi_fraction_processed": roi_frac_proc,
        "mask_binary_ok": mask_binary_ok,
        "empty_mask": empty_mask,
        "warning": warning_msg,
    }


def make_thumbnail(img: np.ndarray, target_size: int = 160) -> np.ndarray:
    """Create a letterboxed square thumbnail with aspect ratio preserved."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off : y_off + nh, x_off : x_off + nw] = resized
    return canvas


def generate_contact_sheet(
    image_stems: List[str],
    output_path: Path,
    rows: int = 4,
    cols: int = 6,
    thumb_size: int = 160,
) -> None:
    """Create a contact sheet grid of 24 thumbnails."""
    grid = np.zeros((rows * thumb_size, cols * thumb_size), dtype=np.uint8)

    for idx, stem in enumerate(image_stems[: rows * cols]):
        r = idx // cols
        c = idx % cols
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        img = load_image(img_path)
        thumb = make_thumbnail(img, target_size=thumb_size)
        grid[r * thumb_size : (r + 1) * thumb_size, c * thumb_size : (c + 1) * thumb_size] = thumb

    save_image(output_path, grid)


def generate_previews_and_contacts(df_report: pd.DataFrame) -> None:
    """Generate ROI outline previews and contact sheets."""
    preview_dir = RESULTS_DIR / "mask_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # 1. Select images for previews:
    # 6 random masked per source + extreme roi_fractions (< 0.10 or > 0.60)
    masked_df = df_report[df_report["has_mask"] == True].copy()
    mcu_masked = masked_df[masked_df["source"] == "Montgomery"]
    chn_masked = masked_df[masked_df["source"] == "Shenzhen"]

    mcu_sample = rng.choice(mcu_masked["stem"].values, size=min(6, len(mcu_masked)), replace=False).tolist()
    chn_sample = rng.choice(chn_masked["stem"].values, size=min(6, len(chn_masked)), replace=False).tolist()

    extremes = []
    for _, row in masked_df.iterrows():
        frac = float(row["roi_fraction_processed"])
        if frac < 0.10 or frac > 0.60:
            extremes.append(row["stem"])

    preview_stems = sorted(set(mcu_sample + chn_sample + extremes))
    print(f"Generating {len(preview_stems)} mask overlay previews in {preview_dir}...")

    for stem in preview_stems:
        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

        img = load_image(img_path)
        mask = load_image(mask_path)

        # Create BGR image
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Red outline in BGR is (0, 0, 255)
        cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2)

        out_preview = preview_dir / f"{stem}_preview.png"
        cv2.imwrite(str(out_preview), bgr)

    # 2. Contact sheets: 24 random thumbnails for Montgomery and Shenzhen
    print("Generating contact sheets for Montgomery and Shenzhen...")
    mcu_all = df_report[df_report["source"] == "Montgomery"]["stem"].values
    chn_all = df_report[df_report["source"] == "Shenzhen"]["stem"].values

    mcu_24 = rng.choice(mcu_all, size=24, replace=False).tolist()
    chn_24 = rng.choice(chn_all, size=24, replace=False).tolist()

    generate_contact_sheet(mcu_24, RESULTS_DIR / "contact_montgomery.png")
    generate_contact_sheet(chn_24, RESULTS_DIR / "contact_shenzhen.png")
    print(f"Contact sheets saved to {RESULTS_DIR}")


def run_preprocessing(workers: int = 4, force: bool = False) -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}. Run build_manifest.py first."

    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(manifest_csv)
    records = df_manifest.to_dict(orient="records")

    report_rows = []
    warnings_list = []

    print(f"Starting preprocessing for {len(records)} images with {workers} workers (force={force})...")

    if workers <= 1:
        for r in tqdm(records, desc="Preprocessing"):
            res = process_single_item(r, force=force)
            if res["warning"]:
                warnings_list.append(res["warning"])
                logger.warning(res["warning"])
            del res["warning"]
            report_rows.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_stem = {executor.submit(process_single_item, r, force): r["stem"] for r in records}
            for fut in tqdm(as_completed(future_to_stem), total=len(records), desc="Preprocessing"):
                res = fut.result()
                if res["warning"]:
                    warnings_list.append(res["warning"])
                    logger.warning(res["warning"])
                del res["warning"]
                report_rows.append(res)

    # Sort report rows by stem
    report_rows.sort(key=lambda x: x["stem"])
    df_report = pd.DataFrame(report_rows)

    # Write results/processed_report.csv
    report_csv = RESULTS_DIR / "processed_report.csv"
    df_report.to_csv(report_csv, index=False)
    print(f"Preprocessing report written to {report_csv}")

    # Write warnings list if any
    warnings_txt = RESULTS_DIR / "preprocessing_warnings.txt"
    if warnings_list:
        warnings_txt.write_text("\n".join(warnings_list) + "\n", encoding="utf-8")
    else:
        warnings_txt.write_text("No shape mismatches or preprocessing warnings.\n", encoding="utf-8")

    # Generate previews and contact sheets
    generate_previews_and_contacts(df_report)

    # Print summary statistics
    masked_report = df_report[df_report["has_mask"] == True]
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Total processed images: {len(df_report)} (expected 800)")
    print(f"Total processed masks:  {len(masked_report)} (expected 704)")
    print(f"All masks binary [0, 255] OK: {masked_report['mask_binary_ok'].all()}")
    print(f"Empty masks detected: {masked_report['empty_mask'].any()}")

    for src_name, group in masked_report.groupby("source"):
        rois = group["roi_fraction_processed"].astype(float)
        print(f"ROI fraction {src_name}: min={rois.min():.4f}, median={rois.median():.4f}, max={rois.max():.4f}")

    return df_report


def main():
    parser = argparse.ArgumentParser(description="Preprocess medical X-ray dataset.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing files")
    args = parser.parse_args()

    run_preprocessing(workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
