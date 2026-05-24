"""Correctness tests for the P2 MILP association solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers import P2MILPSolver, P2Result


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
    seed: int = 0
    scenario_id: str = "tiny"


def _base_scenario(
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
    np.testing.assert_allclose(result.x.sum(axis=0), 1.0, atol=1e-8)
    assert np.all(result.x <= scenario.v + 1e-8)
    np.testing.assert_allclose(result.h[:, 0], 0.0, atol=1e-8)
    assert np.all(result.handover_per_cell <= scenario.H + 1e-8)


def test_single_cell_no_handover_when_best_satellite_is_stable() -> None:
    """C=1 with stable best visibility should choose one satellite and h=0."""
    v = np.ones((2, 1, 3), dtype=np.float64)
    g = np.array([[[2e-12, 2e-12, 2e-12]], [[1e-13, 1e-13, 1e-13]]], dtype=np.float64)
    scenario = _base_scenario(v=v, handover_budget=np.array([2.0], dtype=np.float64), g=g)

    result = P2MILPSolver(scenario, time_limit=10.0).solve()

    assert not result.status.startswith("infeasible")
    _assert_assignment_feasible(result, scenario)
    np.testing.assert_allclose(result.x[0, 0, :], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.h[0, :], 0.0, atol=1e-8)


def test_handover_budget_zero_forces_constant_association() -> None:
    """H_c=0 should prevent switching even if the preferred satellite changes."""
    v = np.ones((2, 1, 4), dtype=np.float64)
    g = np.array(
        [[[2e-12, 2e-12, 1e-13, 1e-13]], [[1e-13, 1e-13, 2e-12, 2e-12]]],
        dtype=np.float64,
    )
    scenario = _base_scenario(v=v, handover_budget=np.array([0.0], dtype=np.float64), g=g)

    result = P2MILPSolver(scenario, time_limit=10.0).solve()

    assert not result.status.startswith("infeasible")
    _assert_assignment_feasible(result, scenario)
    selected = np.argmax(result.x[:, 0, :], axis=0)
    assert np.unique(selected).size == 1
    np.testing.assert_allclose(result.handover_per_cell, 0.0, atol=1e-8)


def test_visibility_forced_switch_consumes_budget() -> None:
    """A visibility break should be feasible when the cell has enough budget."""
    v = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]], dtype=np.float64)
    scenario = _base_scenario(v=v, handover_budget=np.array([1.0], dtype=np.float64))

    result = P2MILPSolver(scenario, time_limit=10.0).solve()

    assert not result.status.startswith("infeasible")
    _assert_assignment_feasible(result, scenario)
    np.testing.assert_allclose(result.x[0, 0, 0], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.x[1, 0, 1:], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.handover_per_cell[0], 1.0, atol=1e-8)


def test_invisibility_reports_infeasible_when_budget_exhausted() -> None:
    """Budget exhaustion plus visibility break must not silently violate H_c."""
    v = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]], dtype=np.float64)
    scenario = _base_scenario(v=v, handover_budget=np.array([0.0], dtype=np.float64))

    result = P2MILPSolver(scenario, time_limit=10.0).solve()

    assert result.status.startswith("infeasible")
    assert float("-inf") == result.U
    np.testing.assert_allclose(result.handover_per_cell, 0.0, atol=1e-8)


def test_multiple_cells_respect_independent_budgets() -> None:
    """Per-cell hard budgets should be enforced independently."""
    v = np.ones((2, 2, 3), dtype=np.float64)
    g = np.array(
        [
            [[2e-12, 2e-12, 2e-12], [2e-12, 1e-13, 1e-13]],
            [[1e-13, 1e-13, 1e-13], [1e-13, 2e-12, 2e-12]],
        ],
        dtype=np.float64,
    )
    scenario = _base_scenario(
        v=v,
        handover_budget=np.array([0.0, 1.0], dtype=np.float64),
        g=g,
    )

    result = P2MILPSolver(scenario, time_limit=10.0).solve()

    assert not result.status.startswith("infeasible")
    _assert_assignment_feasible(result, scenario)
    np.testing.assert_allclose(result.handover_per_cell[0], 0.0, atol=1e-8)
    assert result.handover_per_cell[1] <= 1.0 + 1e-8
