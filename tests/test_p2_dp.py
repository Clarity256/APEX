"""Correctness tests for the P2 dynamic-programming solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers import P2DPSolver, P2MILPSolver, P2Result


@dataclass(frozen=True)
class TinyScenario:
    """Minimal ScenarioInstance-compatible test fixture."""

    S: int
    C: int
    K: int
    M: int
    g: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]
    N_PRB: NDArray[np.float64]
    P_max: NDArray[np.float64]
    H: NDArray[np.float64]
    W_PRB: float
    N0: float
    T_f: float
    eps: float
    lambda_h: float


def _scenario(
    v: NDArray[np.float64],
    handover_budget: NDArray[np.float64],
    g: NDArray[np.float64] | None = None,
) -> TinyScenario:
    satellite_count, cell_count, slow_slot_count = v.shape
    fast_slot_count = 2
    if g is None:
        g = np.ones((satellite_count, cell_count, slow_slot_count), dtype=np.float64) * 1e-12
    return TinyScenario(
        S=satellite_count,
        C=cell_count,
        K=slow_slot_count,
        M=fast_slot_count,
        g=g,
        v=v,
        a=np.ones((cell_count, slow_slot_count, fast_slot_count), dtype=np.float64) * 1e4,
        N_PRB=np.ones(satellite_count, dtype=np.float64) * 100.0,
        P_max=np.ones(satellite_count, dtype=np.float64) * 100.0,
        H=handover_budget,
        W_PRB=180e3,
        N0=1e-15,
        T_f=0.01,
        eps=1e-4,
        lambda_h=0.1,
    )


def _assert_assignment_feasible(result: P2Result, scenario: TinyScenario) -> None:
    assert result.x.shape == (scenario.S, scenario.C, scenario.K)
    assert result.h.shape == (scenario.C, scenario.K)
    assert result.xi.shape == (scenario.C, scenario.K)
    assert not result.status.startswith("infeasible")
    np.testing.assert_allclose(result.x.sum(axis=0), 1.0, atol=1e-8)
    assert np.all(result.x <= scenario.v + 1e-8)
    np.testing.assert_allclose(result.h[:, 0], 0.0, atol=1e-8)
    assert np.all(result.handover_per_cell <= scenario.H + 1e-8)


def test_dp_single_cell_prefers_stable_best_satellite() -> None:
    """C=1 with a stable best satellite should choose it with no handovers."""
    v = np.ones((2, 1, 3), dtype=np.float64)
    g = np.array([[[2e-12, 2e-12, 2e-12]], [[1e-13, 1e-13, 1e-13]]], dtype=np.float64)
    scenario = _scenario(v=v, handover_budget=np.array([2.0]), g=g)

    result = P2DPSolver(scenario).solve()

    _assert_assignment_feasible(result, scenario)
    np.testing.assert_allclose(result.x[0, 0, :], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.h[0, :], 0.0, atol=1e-8)


def test_dp_budget_zero_forces_constant_association() -> None:
    """H_c=0 should prevent switching even if channel quality changes."""
    v = np.ones((2, 1, 4), dtype=np.float64)
    g = np.array(
        [[[2e-12, 2e-12, 1e-13, 1e-13]], [[1e-13, 1e-13, 2e-12, 2e-12]]],
        dtype=np.float64,
    )
    scenario = _scenario(v=v, handover_budget=np.array([0.0]), g=g)

    result = P2DPSolver(scenario).solve()

    _assert_assignment_feasible(result, scenario)
    selected = np.argmax(result.x[:, 0, :], axis=0)
    assert np.unique(selected).size == 1
    np.testing.assert_allclose(result.handover_per_cell[0], 0.0, atol=1e-8)


def test_dp_visibility_break_consumes_budget() -> None:
    """A forced visibility break should consume exactly one handover."""
    v = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]], dtype=np.float64)
    scenario = _scenario(v=v, handover_budget=np.array([1.0]))

    result = P2DPSolver(scenario).solve()

    _assert_assignment_feasible(result, scenario)
    np.testing.assert_allclose(result.x[0, 0, 0], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.x[1, 0, 1:], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.handover_per_cell[0], 1.0, atol=1e-8)


def test_dp_reports_infeasible_when_budget_exhausted() -> None:
    """A forced visibility break with H_c=0 should report infeasible."""
    v = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]], dtype=np.float64)
    scenario = _scenario(v=v, handover_budget=np.array([0.0]))

    result = P2DPSolver(scenario).solve()

    assert result.status.startswith("infeasible")
    assert float("-inf") == result.U


def test_dp_matches_full_milp_objective_on_small_instance() -> None:
    """DP should match the full MILP optimum for the current separable surrogate."""
    v = np.ones((2, 2, 4), dtype=np.float64)
    g = np.array(
        [
            [[2e-12, 2e-12, 1e-13, 1e-13], [2e-12, 1e-13, 1e-13, 1e-13]],
            [[1e-13, 1e-13, 2e-12, 2e-12], [1e-13, 2e-12, 2e-12, 2e-12]],
        ],
        dtype=np.float64,
    )
    scenario = _scenario(v=v, handover_budget=np.array([1.0, 1.0]), g=g)

    milp = P2MILPSolver(scenario, time_limit=10.0).solve()
    dp = P2DPSolver(scenario).solve()

    _assert_assignment_feasible(dp, scenario)
    np.testing.assert_allclose(dp.U, milp.U, atol=1e-8)
    np.testing.assert_allclose(dp.handover_per_cell, milp.handover_per_cell, atol=1e-8)
