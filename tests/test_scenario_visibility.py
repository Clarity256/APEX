"""Tests for scenario/visibility.py — visibility matrix and feasibility checks."""

from __future__ import annotations

import numpy as np

from leo_alloc.scenario.visibility import check_feasibility, compute_visibility, ensure_coverage


def test_compute_visibility_shape() -> None:
    el = np.random.default_rng(0).uniform(-90, 90, (2, 5, 10))
    v = compute_visibility(el, threshold_deg=20.0)
    assert v.shape == (2, 5, 10)


def test_compute_visibility_binary() -> None:
    el = np.random.default_rng(1).uniform(-90, 90, (3, 4, 8))
    v = compute_visibility(el, threshold_deg=20.0)
    unique = set(np.unique(v))
    assert unique.issubset({0.0, 1.0})


def test_compute_visibility_dtype() -> None:
    el = np.ones((2, 3, 4)) * 30.0
    v = compute_visibility(el, threshold_deg=20.0)
    assert v.dtype == np.float64


def test_above_threshold_is_one() -> None:
    el = np.array([[[30.0]]])
    v = compute_visibility(el, threshold_deg=20.0)
    assert v[0, 0, 0] == 1.0


def test_below_threshold_is_zero() -> None:
    el = np.array([[[10.0]]])
    v = compute_visibility(el, threshold_deg=20.0)
    assert v[0, 0, 0] == 0.0


def test_exactly_at_threshold_is_one() -> None:
    el = np.array([[[20.0]]])
    v = compute_visibility(el, threshold_deg=20.0)
    assert v[0, 0, 0] == 1.0


# ---------------------------------------------------------------------------
# ensure_coverage
# ---------------------------------------------------------------------------


def test_ensure_coverage_always_feasible() -> None:
    rng = np.random.default_rng(7)
    el = rng.uniform(-90, 90, (3, 5, 8))
    v = compute_visibility(el, threshold_deg=20.0)
    v_cov = ensure_coverage(v, el)
    # Every (cell, slot) must have at least one visible satellite
    assert np.all(v_cov.sum(axis=0) >= 1)


def test_ensure_coverage_preserves_existing_ones() -> None:
    el = np.ones((2, 3, 4)) * 30.0
    v = compute_visibility(el, threshold_deg=20.0)
    v_cov = ensure_coverage(v, el)
    np.testing.assert_array_equal(v, v_cov)


def test_ensure_coverage_fills_unserved_cell_slot() -> None:
    # Satellite 0 below threshold; satellite 1 above only at k=1
    el = np.array([[[5.0, 5.0]], [[5.0, 25.0]]])  # [S=2, C=1, K=2]
    v = compute_visibility(el, threshold_deg=20.0)
    # At k=0, no satellite is visible — ensure_coverage should fix that
    assert v[:, 0, 0].sum() == 0
    v_cov = ensure_coverage(v, el)
    assert v_cov[:, 0, 0].sum() >= 1


# ---------------------------------------------------------------------------
# check_feasibility — DP min-handover test
# ---------------------------------------------------------------------------


def test_always_visible_zero_budget_is_feasible() -> None:
    v = np.ones((3, 4, 5))
    handover_budget = np.zeros(4)
    feasible, min_ho = check_feasibility(v, handover_budget)
    assert feasible
    assert np.all(min_ho == 0)


def test_forced_handover_feasible_with_budget() -> None:
    # Sat 0 visible at k=0, sat 1 visible at k=1 (C=1)
    v = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # [S=2, C=1, K=2]
    handover_budget = np.array([1.0])
    feasible, min_ho = check_feasibility(v, handover_budget)
    assert feasible
    assert min_ho[0] == 1


def test_forced_handover_infeasible_without_budget() -> None:
    v = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # [S=2, C=1, K=2]
    handover_budget = np.array([0.0])
    feasible, min_ho = check_feasibility(v, handover_budget)
    assert not feasible


def test_min_handovers_correct_for_two_segment_pass() -> None:
    # Sat 0 visible at k=0,1; Sat 1 visible at k=2,3 (C=1)
    v = np.array([[[1.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]])  # [2, 1, 4]
    handover_budget = np.array([1.0])
    feasible, min_ho = check_feasibility(v, handover_budget)
    assert feasible
    assert min_ho[0] == 1


def test_multiple_cells_independent_budgets() -> None:
    # C=2: cell 0 needs 1 handover (budget=1 ✓), cell 1 needs 2 (budget=1 ✗)
    v = np.zeros((2, 2, 4))
    v[0, 0, :2] = 1.0
    v[1, 0, 2:] = 1.0  # cell 0: needs 1 handover
    v[0, 1, :] = 1.0
    v[1, 1, 1:3] = 1.0  # cell 1: can stay on sat 0 (0 handovers)
    handover_budget = np.array([1.0, 0.0])
    feasible, min_ho = check_feasibility(v, handover_budget)
    assert feasible
    assert min_ho[0] == 1
    assert min_ho[1] == 0


def test_infeasible_when_no_satellite_visible_at_slot() -> None:
    v = np.zeros((2, 1, 3))
    v[:, 0, 0] = 1.0
    v[:, 0, 2] = 1.0
    # k=1: nothing visible for cell 0 → infeasible
    handover_budget = np.array([5.0])
    feasible, _ = check_feasibility(v, handover_budget)
    assert not feasible
