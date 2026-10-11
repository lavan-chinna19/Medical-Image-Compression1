"""Ablation study on ROI fill strategies (none vs mean vs inpaint) on 40 stratified images."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

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
)
from medcomp.io_utils import load_image
from medcomp.roi_codec import roi_encode

FILL_ABLATION_CSV = RESULTS_DIR / "roi_fill_ablation.csv"


def run_fill_ablation() -> pd.DataFrame:
    """Run fill strategy ablation across 40 seeded images at quality 30."""
    manifest_csv = DATA_DIR / "manifest.csv"
    assert manifest_csv.is_file(), f"Manifest not found: {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    df = df[df["has_mask"] == True].copy()

    # Stratified 40 images (source + label, seed=42)
    limit = 40
    df["strat_group"] = df["source"].astype(str) + "_" + df["label"].astype(str)
    sampled_indices = []
    rng = np.random.default_rng(42)
    for _, group in df.groupby("strat_group"):
        n_group = max(1, int(round(limit * len(group) / len(df))))
        chosen = rng.choice(group.index.values, size=min(n_group, len(group)), replace=False)
        sampled_indices.extend(chosen)
    if len(sampled_indices) > limit:
        sampled_indices = sampled_indices[:limit]
    elif len(sampled_indices) < limit:
        remaining = [idx for idx in df.index if idx not in sampled_indices]
        sampled_indices.extend(rng.choice(remaining, size=limit - len(sampled_indices), replace=False))

    df_sample = df.loc[sampled_indices].copy()
    print(f"Sampled {len(df_sample)} images for fill ablation: Montgomery={sum(df_sample['source']=='Montgomery')}, Shenzhen={sum(df_sample['source']=='Shenzhen')}")

    methods = ["dct", "dwt"]
    fills = ["none", "mean", "inpaint"]
    q = 30

    rows: List[Dict[str, Any]] = []

    for _, item in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Fill Ablation"):
        stem = item["stem"]
        source = item["source"]
        label = int(item["label"])

        img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
        mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"

        img = load_image(img_path)
        mask = (load_image(mask_path) > 127)

        for method in methods:
            for fill in fills:
                data, _, parts = roi_encode(img, mask, method=method, quality=q, fill=fill)
                rows.append({
                    "stem": stem,
                    "source": source,
                    "label": label,
                    "method": method,
                    "quality": q,
                    "fill": fill,
                    "total_bytes": len(data),
                    "header_bytes": parts["header"],
                    "mask_bytes": parts["mask"],
                    "roi_bytes": parts["roi"],
                    "background_bytes": parts["background"],
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(FILL_ABLATION_CSV, index=False)
    print(f"Saved {len(df_out)} rows to {FILL_ABLATION_CSV}")

    # Aggregated comparison
    summary = df_out.groupby(["method", "fill"]).agg(
        mean_total_bytes=("total_bytes", "mean"),
        mean_bg_bytes=("background_bytes", "mean"),
    ).reset_index()

    print("\n" + "=" * 65)
    print("                 ROI FILL ABLATION SUMMARY (Q=30)")
    print("=" * 65)
    print(summary.to_string(index=False))
    print("=" * 65)

    overall_fill = df_out.groupby("fill")["background_bytes"].mean()
    best_fill = overall_fill.idxmin()
    print(f"\nBest performing fill strategy (lowest background bytes): '{best_fill}' "
          f"with {overall_fill[best_fill]:.1f} mean bg bytes (vs {overall_fill['none']:.1f} for 'none')\n")

    return df_out


if __name__ == "__main__":
    run_fill_ablation()
