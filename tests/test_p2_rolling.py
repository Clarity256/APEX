"""Correctness tests for the P2 rolling-window solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from leo_alloc.solvers import P2MILPSolver, P2Result, P2RollingSolver


@dataclass(frozen=True)
class RollingScenario:
    """Minimal ScenarioInstance-compatible rolling test fixture."""

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
) -> RollingScenario:
    satellite_count, cell_count, slow_slot_count = v.shape
    fast_slot_count = 2
    if g is None:
        g = np.ones((satellite_count, cell_count, slow_slot_count), dtype=np.float64) * 1e-12
    return RollingScenario(
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


def _assert_rolling_feasible(result: P2Result, scenario: RollingScenario) -> None:
    assert result.x.shape == (scenario.S, scenario.C, scenario.K)
    assert result.h.shape == (scenario.C, scenario.K)
    assert result.xi.shape == (scenario.C, scenario.K)
    assert not result.status.startswith("infeasible")
    np.testing.assert_allclose(result.x.sum(axis=0), 1.0, atol=1e-8)
    assert np.all(result.x <= scenario.v + 1e-8)
    assert np.all(result.handover_per_cell <= scenario.H + 1e-8)


def test_rolling_forced_boundary_switch_consumes_remaining_budget() -> None:
    """A forced switch at an overlap boundary should be counted in the committed result."""
    v = np.array([[[1.0, 1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0, 1.0]]])
    scenario = _scenario(v=v.astype(np.float64), handover_budget=np.array([1.0]))

    result = P2RollingSolver(scenario, window=3, step=2, time_limit=10.0).solve()

    _assert_rolling_feasible(result, scenario)
    np.testing.assert_allclose(result.x[0, 0, :2], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.x[1, 0, 2:], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.h[0], np.array([0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(result.handover_per_cell[0], 1.0, atol=1e-8)


def test_rolling_full_window_matches_full_milp() -> None:
    """Using one window over the full horizon should reproduce the L1 MILP result."""
    v = np.ones((2, 1, 4), dtype=np.float64)
    g = np.array([[[2e-12, 2e-12, 1e-13, 1e-13]], [[1e-13, 1e-13, 2e-12, 2e-12]]])
    scenario = _scenario(v=v, handover_budget=np.array([1.0]), g=g.astype(np.float64))

    full = P2MILPSolver(scenario, time_limit=10.0).solve()
    rolling = P2RollingSolver(scenario, window=4, step=4, time_limit=10.0).solve()

    _assert_rolling_feasible(rolling, scenario)
    np.testing.assert_allclose(rolling.x, full.x, atol=1e-8)
    np.testing.assert_allclose(rolling.h, full.h, atol=1e-8)
    np.testing.assert_allclose(rolling.xi, full.xi, atol=1e-8)
    np.testing.assert_allclose(rolling.U, full.U, atol=1e-8)


def test_rolling_reports_infeasible_when_remaining_budget_is_exhausted() -> None:
    """Rolling must report infeasible when a later visibility break needs unavailable budget."""
    v = np.array([[[1.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=np.float64)
    scenario = _scenario(v=v, handover_budget=np.array([0.0]))

    result = P2RollingSolver(scenario, window=2, step=2, time_limit=10.0).solve()

    assert result.status.startswith("infeasible")
    assert float("-inf") == result.U


def test_rolling_rejects_step_larger_than_window() -> None:
    """The solver should reject nonsensical rolling configurations."""
    v = np.ones((2, 1, 3), dtype=np.float64)
    scenario = _scenario(v=v, handover_budget=np.array([1.0]))

    with pytest.raises(ValueError, match="step must be less than or equal to window"):
        P2RollingSolver(scenario, window=2, step=3)
