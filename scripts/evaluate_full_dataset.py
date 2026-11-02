"""CLI and Background Runner for Full-Dataset Performance Evaluation.

Evaluates DCT-Based Method, DWT-Based Method, and Proposed Region-Aware Near-Lossless
Methodology across every eligible image in the evaluation dataset.

Usage:
    python scripts/evaluate_full_dataset.py
    python scripts/evaluate_full_dataset.py --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medcomp.config import PROCESSED_IMAGES_DIR, RESULTS_DIR
from medcomp.dataset_eval import (
    discover_evaluation_dataset,
    run_full_dataset_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate compression methods across full dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke testing.")
    parser.add_argument("--q-dct", type=int, default=50, help="DCT quality factor (1-100). Default: 50.")
    parser.add_argument("--q-dwt", type=int, default=50, help="DWT quality factor (1-100). Default: 50.")
    parser.add_argument("--delta-core", type=int, default=2, help="Proposed delta core tolerance. Default: 2.")
    parser.add_argument("--delta-band", type=int, default=4, help="Proposed delta band tolerance. Default: 4.")
    parser.add_argument("--band-px", type=int, default=8, help="Proposed band dilation radius. Default: 8.")
    parser.add_argument("--bg-q", type=int, default=50, help="Proposed background quality factor. Default: 50.")
    args = parser.parse_args()

    print("=" * 70)
    print("MEDCOMP: Full-Dataset Performance Evaluation Runner")
    print("=" * 70)

    eligible_paths, skipped = discover_evaluation_dataset()
    total = len(eligible_paths)
    if args.limit is not None and args.limit > 0:
        total = min(total, args.limit)

    print(f"Dataset Location: {PROCESSED_IMAGES_DIR}")
    print(f"Discovered Eligible Images: {len(eligible_paths)} (Evaluating: {total})")
    if skipped:
        print(f"Skipped / Unreadable Files: {len(skipped)}")
        for s in skipped[:5]:
            print(f"  - {s['file']}: {s['reason']}")

    params = {
        "q_dct": args.q_dct,
        "q_dwt": args.q_dwt,
        "delta_core": args.delta_core,
        "delta_band": args.delta_band,
        "band_px": args.band_px,
        "bg_quality": args.bg_q,
    }

    print("\nCompression Settings:")
    print(f"  - DCT Quality: {params['q_dct']}")
    print(f"  - DWT Quality: {params['q_dwt']}")
    print(f"  - Proposed Core / Band Delta: {params['delta_core']} / {params['delta_band']}")
    print(f"  - Proposed Band Radius: {params['band_px']} px")
    print(f"  - Proposed Background Quality: {params['bg_quality']}")
    print("\nStarting evaluation across all three methodologies...\n")

    def progress_cb(current: int, total_count: int, stage: str, failed_count: int):
        pct = (current / total_count * 100.0) if total_count > 0 else 0.0
        sys.stdout.write(f"\r[{stage}] {current}/{total_count} images ({pct:.1f}%) | Failures: {failed_count}")
        sys.stdout.flush()

    t0 = time.perf_counter()
    eval_run = run_full_dataset_evaluation(
        params=params,
        progress_callback=progress_cb,
        max_images=args.limit,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n\nEvaluation finished in {elapsed:.2f} seconds.")
    print(f"Run ID: {eval_run.run_id}")
    print(f"Successfully evaluated: {eval_run.successfully_evaluated_images}/{eval_run.total_eligible_images} images")
    print(f"Failed images: {eval_run.failed_images_count}")

    print("\n" + "=" * 70)
    print("FULL-DATASET PERFORMANCE EVALUATION RESULTS")
    print("=" * 70)
    df_disp = eval_run.to_display_dataframe()
    print(df_disp.to_string(index=False))
    print("=" * 70)
    print(f"Results persisted to: {RESULTS_DIR / 'full_dataset_evaluation.json'}")
    print(f"Per-image audit trail: {RESULTS_DIR / 'full_dataset_evaluation_per_image.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
