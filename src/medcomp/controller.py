"""Closed-loop rate controller for medical image compression.

Selects the minimum bitstream size that preserves clinical diagnostic fidelity
as measured by the steering classifier (ResNet18).

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Strict Isolation: Selection functions for honest variants (V0 to V4) NEVER import
   or access the judge classifier or ground truth labels. Any attempt to pass judge
   or label columns into honest controller functions raises an explicit AssertionError.
2. Selection Principle:
   - V0: Exhaustive minimum-byte rule.
   - V1: Suffix-stable rule (constraint holds at rung AND all higher rungs).
   - V2: Floor-restricted rule (excludes rungs with mean cohort drift > 0.10).
   - V3: Backoff rule (V0 + k rungs up, capped at top rung).
   - V4: Suffix-stable with 1-rung backoff (V1 + 1 rung up, capped at top rung).
3. Oracle Controls (Clearly marked for theoretical bounding only):
   - O1 / O2: Steered by the judge classifier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


def _assert_no_forbidden_columns(data: Any) -> None:
    """Assert that inputs never contain judge or ground-truth label columns.

    Enforces complete architectural isolation between the rate controller
    and external evaluation benchmarks.
    """
    forbidden_tokens = ["judge", "label"]

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            col_str = str(col).lower()
            for token in forbidden_tokens:
                assert token not in col_str, (
                    f"Forbidden column '{col}' detected in controller inputs! "
                    f"The controller must NEVER read judge or label columns."
                )
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                for k in item.keys():
                    k_str = str(k).lower()
                    for token in forbidden_tokens:
                        assert token not in k_str, (
                            f"Forbidden key '{k}' detected in controller inputs! "
                            f"The controller must NEVER read judge or label columns."
                        )
            elif isinstance(item, pd.Series):
                for k in item.index:
                    k_str = str(k).lower()
                    for token in forbidden_tokens:
                        assert token not in k_str, (
                            f"Forbidden key '{k}' detected in controller inputs! "
                            f"The controller must NEVER read judge or label columns."
                        )
    elif isinstance(data, dict):
        for k in data.keys():
            k_str = str(k).lower()
            for token in forbidden_tokens:
                assert token not in k_str, (
                    f"Forbidden key '{k}' detected in controller inputs! "
                    f"The controller must NEVER read judge or label columns."
                )


def _extract_candidate_val(row: dict[str, Any], candidate_keys: list[str]) -> Any:
    """Extract value matching one of the candidate keys."""
    for key in candidate_keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of {candidate_keys} found in row keys: {list(row.keys())}")


def satisfies_constraint(
    prob: float,
    prob_orig: float,
    tolerance: float,
    threshold: float = 0.5,
) -> bool:
    """Check whether classifier satisfies diagnostic fidelity constraint.

    Constraint:
    1. Same predicted binary class as on the original image:
       (prob >= threshold) == (prob_orig >= threshold)
    2. Absolute probability drift <= tolerance:
       abs(prob - prob_orig) <= tolerance

    Args:
        prob: Predicted probability on reconstructed image.
        prob_orig: Predicted probability on original image.
        tolerance: Allowed probability difference.
        threshold: Classification decision threshold (default: 0.5).

    Returns:
        bool: True if both conditions are satisfied.
    """
    same_class = (prob >= threshold) == (prob_orig >= threshold)
    prob_diff = abs(prob - prob_orig)
    return bool(same_class and (prob_diff <= tolerance + 1e-9))


def _prepare_sorted_records(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    is_oracle: bool = False,
) -> list[dict[str, Any]]:
    """Validate inputs and return sorted records by bytes ascending."""
    if not is_oracle:
        _assert_no_forbidden_columns(rows_for_one_image)

    if isinstance(rows_for_one_image, pd.DataFrame):
        if len(rows_for_one_image) == 0:
            raise ValueError("Input rows DataFrame is empty.")
        records = rows_for_one_image.to_dict(orient="records")
    elif isinstance(rows_for_one_image, (list, tuple)):
        if len(rows_for_one_image) == 0:
            raise ValueError("Input rows list is empty.")
        records = [dict(r) for r in rows_for_one_image]
    else:
        raise ValueError(f"Unsupported input type: {type(rows_for_one_image)}")

    bytes_keys = ["total_bytes", "bytes", "size_bytes"]
    quality_keys = ["quality", "q", "bg_quality"]

    def get_bytes(r: dict[str, Any]) -> int:
        return int(_extract_candidate_val(r, bytes_keys))

    def get_quality(r: dict[str, Any]) -> float:
        for qk in quality_keys:
            if qk in r:
                return float(r[qk])
        return 0.0

    return sorted(records, key=lambda r: (get_bytes(r), get_quality(r)))


# ==============================================================================
# HONEST CONTROLLER VARIANTS (V0 to V4)
# ==============================================================================

def choose_setting_v0(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Variant V0: Exhaustive smallest-bytes rung that satisfies the constraint.

    Selection uses ONLY steering classifier outputs.
    """
    _assert_no_forbidden_columns(rows_for_one_image)
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
    prob_keys = ["prob_steering", "prob", "prob_tb", "prob_rec"]
    orig_keys = ["prob_orig_steering", "prob_original", "prob_orig"]

    for idx, r in enumerate(sorted_records):
        p = float(_extract_candidate_val(r, prob_keys))
        p_orig = float(_extract_candidate_val(r, orig_keys))
        if satisfies_constraint(p, p_orig, tolerance):
            chosen = dict(r)
            chosen["unsatisfied"] = False
            chosen["variant"] = "V0"
            chosen["rung_index"] = idx
            return chosen

    # If none satisfied, return top rung
    chosen = dict(sorted_records[-1])
    chosen["unsatisfied"] = True
    chosen["variant"] = "V0"
    chosen["rung_index"] = len(sorted_records) - 1
    return chosen


def choose_setting_v1(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Variant V1 (Suffix-stable): Smallest-bytes rung such that constraint holds at that rung AND all higher rungs.

    Prevents jumping onto isolated degenerate low-bitrate rungs.
    Selection uses ONLY steering classifier outputs.
    """
    _assert_no_forbidden_columns(rows_for_one_image)
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
    prob_keys = ["prob_steering", "prob", "prob_tb", "prob_rec"]
    orig_keys = ["prob_orig_steering", "prob_original", "prob_orig"]
    n = len(sorted_records)

    sat = [
        satisfies_constraint(
            float(_extract_candidate_val(r, prob_keys)),
            float(_extract_candidate_val(r, orig_keys)),
            tolerance,
        )
        for r in sorted_records
    ]

    # Find smallest index i such that sat[j] is True for all j >= i
    for i in range(n):
        if all(sat[i:]):
            chosen = dict(sorted_records[i])
            chosen["unsatisfied"] = False
            chosen["variant"] = "V1"
            chosen["rung_index"] = i
            return chosen

    chosen = dict(sorted_records[-1])
    chosen["unsatisfied"] = True
    chosen["variant"] = "V1"
    chosen["rung_index"] = n - 1
    return chosen


def choose_setting_v2(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
    allowed_rungs: Optional[Set[Any]] = None,
) -> dict[str, Any]:
    """Variant V2 (Floor): Exhaustive rule restricted to rungs whose cohort mean steering abs prob change <= 0.10.

    Selection uses ONLY steering classifier outputs.
    """
    _assert_no_forbidden_columns(rows_for_one_image)
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
    rung_keys = ["rung", "quality", "q"]

    def get_rung_id(r: dict[str, Any]) -> Any:
        for k in rung_keys:
            if k in r:
                return r[k]
        return None

    if allowed_rungs is not None:
        filtered_records = [r for r in sorted_records if get_rung_id(r) in allowed_rungs]
    else:
        filtered_records = sorted_records

    if not filtered_records:
        chosen = dict(sorted_records[-1])
        chosen["unsatisfied"] = True
        chosen["variant"] = "V2"
        chosen["rung_index"] = len(sorted_records) - 1
        return chosen

    prob_keys = ["prob_steering", "prob", "prob_tb", "prob_rec"]
    orig_keys = ["prob_orig_steering", "prob_original", "prob_orig"]

    for r in filtered_records:
        p = float(_extract_candidate_val(r, prob_keys))
        p_orig = float(_extract_candidate_val(r, orig_keys))
        if satisfies_constraint(p, p_orig, tolerance):
            chosen = dict(r)
            chosen["unsatisfied"] = False
            chosen["variant"] = "V2"
            chosen["rung_index"] = sorted_records.index(r)
            return chosen

    chosen = dict(filtered_records[-1])
    chosen["unsatisfied"] = True
    chosen["variant"] = "V2"
    chosen["rung_index"] = sorted_records.index(filtered_records[-1])
    return chosen


def choose_setting_v3(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
    k: int = 1,
) -> dict[str, Any]:
    """Variant V3 (Backoff): Take V0's chosen rung and move up k rungs, capped at the top rung.

    Selection uses ONLY steering classifier outputs.
    """
    _assert_no_forbidden_columns(rows_for_one_image)
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
    n = len(sorted_records)

    v0_choice = choose_setting_v0(sorted_records, tolerance=tolerance)
    idx = v0_choice["rung_index"]

    if v0_choice["unsatisfied"]:
        chosen = dict(sorted_records[-1])
        chosen["unsatisfied"] = True
        chosen["variant"] = f"V3_k{k}"
        chosen["rung_index"] = n - 1
        return chosen

    new_idx = min(idx + k, n - 1)
    chosen = dict(sorted_records[new_idx])
    chosen["unsatisfied"] = False
    chosen["variant"] = f"V3_k{k}"
    chosen["rung_index"] = new_idx
    return chosen


def choose_setting_v4(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Variant V4: Suffix-stable (V1) plus backoff of 1 rung, capped at the top rung.

    Selection uses ONLY steering classifier outputs.
    """
    _assert_no_forbidden_columns(rows_for_one_image)
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
    n = len(sorted_records)

    v1_choice = choose_setting_v1(sorted_records, tolerance=tolerance)
    idx = v1_choice["rung_index"]

    if v1_choice["unsatisfied"]:
        chosen = dict(sorted_records[-1])
        chosen["unsatisfied"] = True
        chosen["variant"] = "V4"
        chosen["rung_index"] = n - 1
        return chosen

    new_idx = min(idx + 1, n - 1)
    chosen = dict(sorted_records[new_idx])
    chosen["unsatisfied"] = False
    chosen["variant"] = "V4"
    chosen["rung_index"] = new_idx
    return chosen


# ==============================================================================
# ORACLE BOUNDING CONTROLS (O1 and O2)
# ==============================================================================

def choose_setting_oracle_judge(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """THEORETICAL UPPER BOUND ORACLE: Steered directly by the Judge classifier.

    THIS FUNCTION USES THE INDEPENDENT JUDGE CLASSIFIER TO CHOOSE RUNGS.
    FOR THEORETICAL BOUNDING AND BENCHMARKING PURPOSES ONLY.
    NEVER USE AS AN OPERATIONAL CONTROLLER.
    """
    sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=True)
    prob_keys = ["prob_judge"]
    orig_keys = ["prob_orig_judge"]

    for idx, r in enumerate(sorted_records):
        p = float(_extract_candidate_val(r, prob_keys))
        p_orig = float(_extract_candidate_val(r, orig_keys))
        if satisfies_constraint(p, p_orig, tolerance):
            chosen = dict(r)
            chosen["unsatisfied"] = False
            chosen["variant"] = "Oracle_Judge"
            chosen["rung_index"] = idx
            return chosen

    chosen = dict(sorted_records[-1])
    chosen["unsatisfied"] = True
    chosen["variant"] = "Oracle_Judge"
    chosen["rung_index"] = len(sorted_records) - 1
    return chosen


# ==============================================================================
# GENERAL DISPATCHER
# ==============================================================================

def choose_setting(
    rows_for_one_image: Union[pd.DataFrame, Sequence[dict[str, Any]]],
    tolerance: float = 0.05,
    mode: str = "exhaustive",
    variant: str = "V0",
    allowed_rungs: Optional[Set[Any]] = None,
) -> dict[str, Any]:
    """General dispatcher for rate controller selection algorithms.

    Args:
        rows_for_one_image: Input rung rows.
        tolerance: Allowed probability deviation.
        mode: Search algorithm ('exhaustive' or 'bisection').
        variant: Controller variant ('V0', 'V1', 'V2', 'V3_k1', 'V3_k2', 'V4', 'O1', 'O2').
        allowed_rungs: Allowed rungs set for V2 floor variant.

    Returns:
        dict[str, Any]: Selected rung record.
    """
    if variant == "V0":
        if mode == "bisection":
            # Original bisection implementation
            _assert_no_forbidden_columns(rows_for_one_image)
            sorted_records = _prepare_sorted_records(rows_for_one_image, is_oracle=False)
            n_rungs = len(sorted_records)
            prob_keys = ["prob_steering", "prob", "prob_tb", "prob_rec"]
            orig_keys = ["prob_orig_steering", "prob_original", "prob_orig"]
            evaluated_indices: set[int] = set()

            def test_idx(idx: int) -> bool:
                evaluated_indices.add(idx)
                r = sorted_records[idx]
                p = float(_extract_candidate_val(r, prob_keys))
                po = float(_extract_candidate_val(r, orig_keys))
                return satisfies_constraint(p, po, tolerance)

            low = 0
            high = n_rungs - 1
            best_idx: Optional[int] = None

            while low <= high:
                mid = (low + high) // 2
                if test_idx(mid):
                    best_idx = mid
                    high = mid - 1
                else:
                    low = mid + 1

            if best_idx is not None:
                lower_nbr = best_idx - 1
                if lower_nbr >= 0 and lower_nbr not in evaluated_indices:
                    if test_idx(lower_nbr):
                        best_idx = lower_nbr
                chosen = dict(sorted_records[best_idx])
                chosen["unsatisfied"] = False
            else:
                if (n_rungs - 1) not in evaluated_indices:
                    if test_idx(n_rungs - 1):
                        best_idx = n_rungs - 1
                if best_idx is not None:
                    chosen = dict(sorted_records[best_idx])
                    chosen["unsatisfied"] = False
                else:
                    chosen = dict(sorted_records[-1])
                    chosen["unsatisfied"] = True

            chosen["rounds"] = len(evaluated_indices)
            chosen["search_mode"] = "bisection"
            chosen["variant"] = "V0"
            return chosen
        else:
            res = choose_setting_v0(rows_for_one_image, tolerance)
            res["rounds"] = len(rows_for_one_image)
            res["search_mode"] = "exhaustive"
            return res

    elif variant == "V1":
        return choose_setting_v1(rows_for_one_image, tolerance)
    elif variant == "V2":
        return choose_setting_v2(rows_for_one_image, tolerance, allowed_rungs=allowed_rungs)
    elif variant in ("V3", "V3_k1"):
        return choose_setting_v3(rows_for_one_image, tolerance, k=1)
    elif variant == "V3_k2":
        return choose_setting_v3(rows_for_one_image, tolerance, k=2)
    elif variant == "V4":
        return choose_setting_v4(rows_for_one_image, tolerance)
    elif variant in ("O1", "O2", "Oracle_Judge"):
        return choose_setting_oracle_judge(rows_for_one_image, tolerance)
    else:
        raise ValueError(f"Unknown variant '{variant}'.")


def compute_allowed_floor_rungs(
    ladder_df: pd.DataFrame,
    max_mean_delta: float = 0.10,
) -> set[str]:
    """Identify ladder rungs whose cohort-wide mean steering absolute prob delta <= max_mean_delta.

    Used by Variant V2 to eliminate floor rungs that cause widespread collapse.
    """
    _assert_no_forbidden_columns(ladder_df)
    prob_keys = ["prob_steering", "prob", "prob_tb", "prob_rec"]
    orig_keys = ["prob_orig_steering", "prob_original", "prob_orig"]

    df = ladder_df.copy()
    p_col = next(c for c in prob_keys if c in df.columns)
    po_col = next(c for c in orig_keys if c in df.columns)

    df["abs_diff"] = np.abs(df[p_col] - df[po_col])
    rung_means = df.groupby("rung")["abs_diff"].mean()

    allowed = set(rung_means[rung_means <= max_mean_delta].index)
    return allowed


def compare_bisection_vs_exhaustive(
    ladder_df: pd.DataFrame,
    tolerances: Sequence[float] = (0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Compare bisection search vs exhaustive search across all images and families."""
    _assert_no_forbidden_columns(ladder_df)

    results: list[dict[str, Any]] = []
    families = ladder_df["family"].unique() if "family" in ladder_df.columns else ["all"]
    stems = ladder_df["stem"].unique()

    for fam in families:
        fam_df = ladder_df[ladder_df["family"] == fam] if fam != "all" else ladder_df

        for tol in tolerances:
            agreements = 0
            bisection_rounds_list: list[int] = []
            exhaustive_rounds_list: list[int] = []
            total_cases = 0

            for stem in stems:
                img_rows = fam_df[fam_df["stem"] == stem]
                if len(img_rows) == 0:
                    continue

                exh_choice = choose_setting(img_rows, tolerance=tol, mode="exhaustive")
                bis_choice = choose_setting(img_rows, tolerance=tol, mode="bisection")

                is_match = (
                    exh_choice["total_bytes"] == bis_choice["total_bytes"]
                    and exh_choice["unsatisfied"] == bis_choice["unsatisfied"]
                )
                if is_match:
                    agreements += 1

                bisection_rounds_list.append(bis_choice["rounds"])
                exhaustive_rounds_list.append(exh_choice["rounds"])
                total_cases += 1

            results.append({
                "family": fam,
                "tolerance": tol,
                "n_images": total_cases,
                "agreement_rate": float(agreements / max(1, total_cases)),
                "mean_bisection_rounds": float(np.mean(bisection_rounds_list)) if bisection_rounds_list else 0.0,
                "mean_exhaustive_rounds": float(np.mean(exhaustive_rounds_list)) if exhaustive_rounds_list else 0.0,
            })

    return pd.DataFrame(results)
