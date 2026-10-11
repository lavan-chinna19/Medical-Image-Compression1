"""Unit tests for closed-loop rate controller and selection variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medcomp.controller import (
    choose_setting,
    choose_setting_oracle_judge,
    choose_setting_v0,
    choose_setting_v1,
    choose_setting_v2,
    choose_setting_v3,
    choose_setting_v4,
    satisfies_constraint,
)


def test_controller_never_reads_judge_or_label_columns():
    """Assert on function inputs that passing judge or label columns raises AssertionError."""
    # Test 1: DataFrame with judge column
    df_with_judge = pd.DataFrame([
        {"prob_steering": 0.1, "prob_orig_steering": 0.1, "total_bytes": 1000, "prob_judge": 0.2}
    ])
    with pytest.raises(AssertionError, match="judge"):
        choose_setting(df_with_judge, tolerance=0.05)

    # Test 2: DataFrame with label column
    df_with_label = pd.DataFrame([
        {"prob_steering": 0.1, "prob_orig_steering": 0.1, "total_bytes": 1000, "label": 1}
    ])
    with pytest.raises(AssertionError, match="label"):
        choose_setting(df_with_label, tolerance=0.05)

    # Test 3: List of dicts with judge key
    dict_with_judge = [
        {"prob_steering": 0.1, "prob_orig_steering": 0.1, "total_bytes": 1000, "judge_score": 0.3}
    ]
    with pytest.raises(AssertionError, match="judge"):
        choose_setting(dict_with_judge, tolerance=0.05)

    # Test 4: List of dicts with true_label key
    dict_with_label = [
        {"prob_steering": 0.1, "prob_orig_steering": 0.1, "total_bytes": 1000, "true_label": 0}
    ]
    with pytest.raises(AssertionError, match="label"):
        choose_setting(dict_with_label, tolerance=0.05)


def test_variants_assert_no_judge_or_label_inputs():
    """Verify that all honest selection functions (V0 to V4) strictly assert on inputs."""
    bad_df = pd.DataFrame([
        {"prob_steering": 0.1, "prob_orig_steering": 0.1, "total_bytes": 1000, "prob_judge": 0.1}
    ])
    for func in [choose_setting_v0, choose_setting_v1, choose_setting_v2, choose_setting_v3, choose_setting_v4]:
        with pytest.raises(AssertionError, match="judge"):
            func(bad_df, tolerance=0.05)


def test_choose_setting_smallest_bytes():
    """Verify choose_setting returns the smallest-bytes satisfying rung on synthetic data."""
    rows = pd.DataFrame([
        {"rung": "q10", "total_bytes": 1200, "prob_steering": 0.12, "prob_orig_steering": 0.10, "quality": 10},
        {"rung": "q20", "total_bytes": 2400, "prob_steering": 0.11, "prob_orig_steering": 0.10, "quality": 20},
        {"rung": "q50", "total_bytes": 5000, "prob_steering": 0.105, "prob_orig_steering": 0.10, "quality": 50},
        {"rung": "q90", "total_bytes": 9000, "prob_steering": 0.101, "prob_orig_steering": 0.10, "quality": 90},
    ])

    # All satisfy tolerance 0.05; smallest bytes is q10 (1200 bytes)
    chosen = choose_setting(rows, tolerance=0.05, mode="exhaustive")
    assert chosen["rung"] == "q10"
    assert chosen["total_bytes"] == 1200
    assert chosen["unsatisfied"] is False


def test_choose_setting_unsatisfied_flag():
    """Verify choose_setting returns highest-quality rung and sets unsatisfied=True when nothing qualifies."""
    rows = pd.DataFrame([
        {"rung": "q10", "total_bytes": 1000, "prob_steering": 0.70, "prob_orig_steering": 0.20, "quality": 10},
        {"rung": "q20", "total_bytes": 2000, "prob_steering": 0.65, "prob_orig_steering": 0.20, "quality": 20},
        {"rung": "q90", "total_bytes": 8000, "prob_steering": 0.55, "prob_orig_steering": 0.20, "quality": 90},
    ])

    chosen = choose_setting(rows, tolerance=0.05, mode="exhaustive")
    assert chosen["rung"] == "q90"
    assert chosen["total_bytes"] == 8000
    assert chosen["unsatisfied"] is True


def test_choose_setting_respects_tolerance():
    """Verify choose_setting correctly adapts choices across different tolerances."""
    rows = pd.DataFrame([
        {"rung": "q10", "total_bytes": 1000, "prob_steering": 0.18, "prob_orig_steering": 0.10, "quality": 10},
        {"rung": "q30", "total_bytes": 2500, "prob_steering": 0.14, "prob_orig_steering": 0.10, "quality": 30},
        {"rung": "q70", "total_bytes": 6000, "prob_steering": 0.11, "prob_orig_steering": 0.10, "quality": 70},
    ])

    c10 = choose_setting(rows, tolerance=0.10, mode="exhaustive")
    assert c10["rung"] == "q10"
    assert c10["total_bytes"] == 1000
    assert c10["unsatisfied"] is False

    c05 = choose_setting(rows, tolerance=0.05, mode="exhaustive")
    assert c05["rung"] == "q30"
    assert c05["total_bytes"] == 2500
    assert c05["unsatisfied"] is False

    c02 = choose_setting(rows, tolerance=0.02, mode="exhaustive")
    assert c02["rung"] == "q70"
    assert c02["total_bytes"] == 6000
    assert c02["unsatisfied"] is False


def test_bisection_equals_exhaustive_monotonic():
    """Verify bisection search equals exhaustive search on monotonic synthetic data."""
    for split_point in range(8):
        rows = []
        for i in range(8):
            diff = 0.20 if i < split_point else 0.01
            rows.append({
                "rung": f"rung_{i}",
                "quality": 10 * (i + 1),
                "total_bytes": 500 * (i + 1),
                "prob_steering": 0.20 + diff,
                "prob_orig_steering": 0.20,
            })
        df = pd.DataFrame(rows)

        for tol in [0.02, 0.05, 0.10]:
            exh = choose_setting(df, tolerance=tol, mode="exhaustive")
            bis = choose_setting(df, tolerance=tol, mode="bisection")

            assert exh["rung"] == bis["rung"], f"Mismatch at split {split_point}, tol {tol}"
            assert exh["total_bytes"] == bis["total_bytes"]
            assert exh["unsatisfied"] == bis["unsatisfied"]


def test_bisection_rounds_efficiency():
    """Verify bisection search operates in ~log2(N) rounds."""
    n = 16
    rows = []
    for i in range(n):
        rows.append({
            "rung": f"r_{i}",
            "quality": i * 5,
            "total_bytes": (i + 1) * 200,
            "prob_steering": 0.30 + (0.15 if i < 10 else 0.02),
            "prob_orig_steering": 0.30,
        })
    df = pd.DataFrame(rows)

    bis = choose_setting(df, tolerance=0.05, mode="bisection")
    assert bis["rounds"] <= 7
    assert bis["rounds"] < n


def test_v1_suffix_stable_synthetic():
    """Verify Variant V1 (suffix-stable) ignores isolated low-rung satisfaction and picks the stable suffix."""
    rows = pd.DataFrame([
        {"rung": "q5", "total_bytes": 500, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 5},    # Satisfies (isolated)
        {"rung": "q10", "total_bytes": 1000, "prob_steering": 0.35, "prob_orig_steering": 0.20, "quality": 10}, # FAILS
        {"rung": "q20", "total_bytes": 2000, "prob_steering": 0.22, "prob_orig_steering": 0.20, "quality": 20}, # Satisfies
        {"rung": "q50", "total_bytes": 5000, "prob_steering": 0.21, "prob_orig_steering": 0.20, "quality": 50}, # Satisfies
        {"rung": "q90", "total_bytes": 9000, "prob_steering": 0.205, "prob_orig_steering": 0.20, "quality": 90},# Satisfies
    ])

    # V0 picks q5 because it satisfies and has smallest bytes
    v0_choice = choose_setting_v0(rows, tolerance=0.05)
    assert v0_choice["rung"] == "q5"

    # V1 must reject q5 because q10 fails, and pick q20 because suffix [q20, q50, q90] is stable
    v1_choice = choose_setting_v1(rows, tolerance=0.05)
    assert v1_choice["rung"] == "q20"
    assert v1_choice["unsatisfied"] is False


def test_v3_backoff_capped_at_top():
    """Verify Variant V3 (backoff) moves up k rungs and is strictly capped at the top rung."""
    rows = pd.DataFrame([
        {"rung": "q10", "total_bytes": 1000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 10},
        {"rung": "q20", "total_bytes": 2000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 20},
        {"rung": "q50", "total_bytes": 5000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 50},
        {"rung": "q90", "total_bytes": 9000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 90},
    ])

    # V0 selects index 0 (q10)
    # k=1 -> index 1 (q20)
    c_k1 = choose_setting_v3(rows, tolerance=0.05, k=1)
    assert c_k1["rung"] == "q20"

    # k=2 -> index 2 (q50)
    c_k2 = choose_setting_v3(rows, tolerance=0.05, k=2)
    assert c_k2["rung"] == "q50"

    # If V0 had chosen q50 (index 2), backoff k=2 -> capped at index 3 (q90)
    rows_top = rows.copy()
    rows_top.loc[0:1, "prob_steering"] = 0.80  # fail q10, q20
    c_capped = choose_setting_v3(rows_top, tolerance=0.05, k=2)
    assert c_capped["rung"] == "q90"
    assert c_capped["rung_index"] == 3


def test_v2_floor_respected():
    """Verify Variant V2 (floor) restricts selection to allowed rungs."""
    rows = pd.DataFrame([
        {"rung": "q5", "total_bytes": 500, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 5},
        {"rung": "q10", "total_bytes": 1000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 10},
        {"rung": "q20", "total_bytes": 2000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 20},
        {"rung": "q50", "total_bytes": 5000, "prob_steering": 0.20, "prob_orig_steering": 0.20, "quality": 50},
    ])

    # If floor only allows q20 and q50
    allowed = {"q20", "q50"}
    c_v2 = choose_setting_v2(rows, tolerance=0.05, allowed_rungs=allowed)
    assert c_v2["rung"] == "q20"
    assert c_v2["total_bytes"] == 2000


def test_oracle_marked_and_functional():
    """Verify Oracle function clearly indicates judge steering and functions correctly."""
    # Docstring must contain ORACLE warning
    assert "ORACLE" in choose_setting_oracle_judge.__doc__
    assert "JUDGE" in choose_setting_oracle_judge.__doc__

    rows_with_judge = pd.DataFrame([
        {"rung": "q10", "total_bytes": 1000, "prob_judge": 0.35, "prob_orig_judge": 0.20, "quality": 10}, # judge diff 0.15 (fails tol 0.05)
        {"rung": "q30", "total_bytes": 3000, "prob_judge": 0.22, "prob_orig_judge": 0.20, "quality": 30}, # judge diff 0.02 (satisfies tol 0.05)
        {"rung": "q70", "total_bytes": 7000, "prob_judge": 0.20, "prob_orig_judge": 0.20, "quality": 70},
    ])

    oracle_choice = choose_setting_oracle_judge(rows_with_judge, tolerance=0.05)
    assert oracle_choice["rung"] == "q30"
    assert oracle_choice["variant"] == "Oracle_Judge"
