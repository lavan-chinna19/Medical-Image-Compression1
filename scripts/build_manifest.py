"""Build dataset manifest CSV, summarize dataset statistics, and check labels."""

import math
from pathlib import Path
import struct
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import (
    CLINICAL_DIR,
    CXR_DIR,
    DATA_DIR,
    DATA_ROOT,
    MASKS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from medcomp.data_utils import (
    load_mask,
    mask_path_for,
    parse_reading,
    roi_fraction,
    source_of,
)


def get_png_dimensions(path: Path) -> tuple[int, int]:
    """Fast extraction of image dimensions (H, W) from PNG header."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a valid PNG: {path}")
        f.read(4)  # Length
        ihdr_type = f.read(4)
        if ihdr_type != b"IHDR":
            raise ValueError(f"Missing IHDR chunk in PNG: {path}")
        w, h = struct.unpack(">II", f.read(8))
        return (h, w)


def build_manifest() -> pd.DataFrame:
    """Build manifest dataframe from CXR_png, masks, and ClinicalReadings."""
    image_files = sorted(CXR_DIR.glob("*.png"))
    assert len(image_files) == 800, f"Expected 800 images in CXR_png, found {len(image_files)}"

    stems = [f.stem for f in image_files]
    assert len(stems) == len(set(stems)), "Image stems must be unique"

    # Verify all masks correspond to an existing image
    all_mask_files = sorted(MASKS_DIR.glob("*.png"))
    assert len(all_mask_files) == 704, f"Expected 704 masks, found {len(all_mask_files)}"
    for mf in all_mask_files:
        stem = mf.stem.replace("_mask", "")
        img_cand = CXR_DIR / f"{stem}.png"
        assert img_cand.is_file(), f"Mask {mf.name} has no corresponding image in CXR_png"

    rows = []
    unparseable_readings = []
    label_mismatches = []
    mask_shape_mismatches = []

    print(f"Building manifest for {len(image_files)} images...")
    for img_path in tqdm(image_files, desc="Processing manifest"):
        stem = img_path.stem
        src = source_of(stem)
        patient_id = stem.rsplit("_", 1)[0]
        label = int(stem.rsplit("_", 1)[-1])

        # Clinical reading
        reading_path = CLINICAL_DIR / f"{stem}.txt"
        assert reading_path.is_file(), f"Missing clinical reading for image: {stem}"
        reading = parse_reading(reading_path)

        if math.isnan(reading["age_years"]) and reading["sex"] is None and not reading["findings_text"]:
            unparseable_readings.append(stem)

        # Cross-check label against findings text
        # Tolerant check: treat as normal if lowercased findings text starts with "normal" (covers typos like "normale")
        findings = reading["findings_text"].strip()
        findings_lower = findings.lower()
        says_normal = findings_lower.startswith("normal")
        if label == 0 and not says_normal:
            label_mismatches.append({
                "stem": stem,
                "source": src,
                "label": label,
                "reading_indicates_normal": False,
                "findings_text": findings,
                "notes": "Label is 0 (normal) but findings text does not start with 'normal'",
            })
        elif label == 1 and says_normal:
            label_mismatches.append({
                "stem": stem,
                "source": src,
                "label": label,
                "reading_indicates_normal": True,
                "findings_text": findings,
                "notes": "Label is 1 (TB) but findings text starts with 'normal'",
            })

        # Image dimensions
        orig_h, orig_w = get_png_dimensions(img_path)

        # Mask info
        mask_path = mask_path_for(stem, DATA_ROOT)
        has_mask = mask_path is not None

        mask_shape_matches_image: Union[bool, str] = ""
        roi_frac: Union[float, str] = ""

        if has_mask:
            assert mask_path is not None
            mask_h, mask_w = get_png_dimensions(mask_path)
            shape_match = (mask_h == orig_h and mask_w == orig_w)
            mask_shape_matches_image = shape_match
            if not shape_match:
                mask_shape_mismatches.append((stem, (orig_h, orig_w), (mask_h, mask_w)))

            mask_arr = load_mask(mask_path)
            roi_frac = round(roi_fraction(mask_arr), 6)

        # Store relative paths to project root for portability
        rel_img_path = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rel_mask_path = str(mask_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if mask_path else ""

        rows.append({
            "source": src,
            "stem": stem,
            "patient_id": patient_id,
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "has_mask": has_mask,
            "label": label,
            "sex": reading["sex"] if reading["sex"] is not None else "",
            "age_years": reading["age_years"] if not math.isnan(reading["age_years"]) else "",
            "findings_text": reading["findings_text"],
            "orig_h": orig_h,
            "orig_w": orig_w,
            "mask_shape_matches_image": mask_shape_matches_image,
            "roi_fraction": roi_frac,
        })

    df = pd.DataFrame(rows)

    # Asserts
    assert len(df) == 800, f"Expected 800 rows, got {len(df)}"
    for src_name, group in df.groupby("source"):
        p_ids = group["patient_id"].tolist()
        assert len(p_ids) == len(set(p_ids)), f"patient_id is not unique in source {src_name}"

    # Write data/manifest.csv
    manifest_csv = DATA_DIR / "manifest.csv"
    df.to_csv(manifest_csv, index=False)
    print(f"Manifest written to {manifest_csv}")

    # Write results/label_mismatches.csv
    mismatches_df = pd.DataFrame(label_mismatches)
    mismatches_csv = RESULTS_DIR / "label_mismatches.csv"
    if not mismatches_df.empty:
        mismatches_df.to_csv(mismatches_csv, index=False)
    else:
        mismatches_csv.write_text("none\n", encoding="utf-8")
    print(f"Label mismatches written to {mismatches_csv} ({len(label_mismatches)} found)")

    # Generate and save results/manifest_summary.txt
    summary_lines = [
        "=== MEDICAL DATASET MANIFEST SUMMARY ===",
        f"Total images: {len(df)}",
        f"Total masks: {df['has_mask'].sum()}",
        "",
        "--- IMAGES AND MASKS PER SOURCE ---",
    ]
    for src_name, group in df.groupby("source"):
        img_cnt = len(group)
        mask_cnt = group["has_mask"].sum()
        summary_lines.append(f"{src_name}: {img_cnt} images, {mask_cnt} masks, {img_cnt - mask_cnt} unmasked")

    summary_lines.extend([
        "",
        "--- HAS_MASK COUNTS ---",
        f"With mask (has_mask=True):  {df['has_mask'].sum()}",
        f"Without mask (has_mask=False): {(~df['has_mask']).sum()}",
        "",
        "--- CLASS COUNTS PER SOURCE AND HAS_MASK GROUP ---",
    ])

    for src_name, group in df.groupby("source"):
        summary_lines.append(f"{src_name} Overall: Class 0 (Normal) = {(group['label'] == 0).sum()}, Class 1 (TB) = {(group['label'] == 1).sum()}")
        for has_m, m_group in group.groupby("has_mask"):
            status_str = "has_mask=True" if has_m else "has_mask=False"
            summary_lines.append(f"  {status_str}: Class 0 = {(m_group['label'] == 0).sum()}, Class 1 = {(m_group['label'] == 1).sum()} (Total = {len(m_group)})")

    summary_lines.extend([
        "",
        "--- OVERALL CLASS COUNTS ---",
        f"Class 0 (Normal): {(df['label'] == 0).sum()} ({((df['label'] == 0).sum() / len(df) * 100):.2f}%)",
        f"Class 1 (TB):     {(df['label'] == 1).sum()} ({((df['label'] == 1).sum() / len(df) * 100):.2f}%)",
        "",
        "--- QUALITY CHECKS & ANOMALIES ---",
        f"Unparseable readings count: {len(unparseable_readings)}",
        f"Unparseable reading stems: {unparseable_readings if unparseable_readings else 'None'}",
        f"Mask shape mismatches count: {len(mask_shape_mismatches)}",
        f"Label vs reading mismatches count: {len(label_mismatches)}",
    ])
    if label_mismatches:
        for lm in label_mismatches:
            summary_lines.append(f"  Stem: {lm['stem']} | Label: {lm['label']} | Findings: '{lm['findings_text']}'")

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = RESULTS_DIR / "manifest_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Manifest summary written to {summary_path}")
    print(summary_text)

    return df


if __name__ == "__main__":
    build_manifest()
