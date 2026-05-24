"""Tests for the fast P1 dual-decomposition approximation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from leo_alloc.solvers import P1CVXSolver, P1DualSolver, P1Result


@pytest.fixture
def dual_sys_params() -> dict[str, NDArray[np.float64] | float]:
    """Return a compact P1 parameter set shared by L1 and L2 tests."""
    return {
        "N_PRB": np.array([100.0, 100.0], dtype=np.float64),
        "P_max": np.array([100.0, 100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }


def _single_sat_params() -> dict[str, NDArray[np.float64] | float]:
    return {
        "N_PRB": np.array([100.0], dtype=np.float64),
        "P_max": np.array([100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }


def _rate_bits(
    n: float,
    p: float,
    g: float,
    sys_params: dict[str, NDArray[np.float64] | float],
) -> float:
    w_prb = float(sys_params["W_PRB"])
    n0 = float(sys_params["N0"])
    t_f = float(sys_params["T_f"])
    return t_f * w_prb * n * np.log2(1.0 + p * g / (w_prb * n * n0))


def _assert_feasible(
    result: P1Result,
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    sys_params: dict[str, NDArray[np.float64] | float],
) -> None:
    n_prb = np.asarray(sys_params["N_PRB"], dtype=np.float64)
    p_max = np.asarray(sys_params["P_max"], dtype=np.float64)
    assert np.all(result.n >= -1e-8)
    assert np.all(result.p >= -1e-8)
    assert np.all(result.z >= -1e-8)
    assert np.all(result.n.sum(axis=1) <= n_prb[:, None] + 1e-6)
    assert np.all(result.p.sum(axis=1) <= p_max[:, None] + 1e-6)
    np.testing.assert_allclose(result.n * (1.0 - x[:, :, None]), 0.0, atol=1e-8)
    np.testing.assert_allclose(result.p * (1.0 - x[:, :, None]), 0.0, atol=1e-8)
    assert np.all(np.cumsum(result.z, axis=1) <= np.cumsum(a, axis=1) + 1e-6)
    assert np.all((result.xi >= -1e-8) & (result.xi <= 1.0 + 1e-8))


def test_dual_single_cell_matches_analytical_solution() -> None:
    """S=C=1 should use all resources and match the scalar rate expression."""
    sys_params = _single_sat_params()
    solver = P1DualSolver(S=1, C=1, M=1, sys_params=sys_params)
    x = np.array([[1.0]], dtype=np.float64)
    a = np.array([[1e6]], dtype=np.float64)
    g = np.array([[1e-12]], dtype=np.float64)

    result = solver.solve(x, a, g)

    assert result.status.startswith("dual_")
    np.testing.assert_allclose(result.n[0, 0, 0], 100.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.p[0, 0, 0], 100.0, rtol=1e-10, atol=1e-10)
    expected_z = _rate_bits(100.0, 100.0, 1e-12, sys_params)
    np.testing.assert_allclose(result.z[0, 0], expected_z, rtol=1e-10)
    np.testing.assert_allclose(result.xi[0], expected_z / a[0, 0], rtol=1e-10)


def test_dual_respects_association_mask_and_constraints(
    dual_sys_params: dict[str, NDArray[np.float64] | float],
) -> None:
    """L2 allocations must be feasible and keep x=0 pairs at zero."""
    solver = P1DualSolver(S=2, C=3, M=4, sys_params=dual_sys_params)
    x = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    a = np.full((3, 4), 5e5, dtype=np.float64)
    g = np.full((2, 3), 1e-12, dtype=np.float64)

    result = solver.solve(x, a, g)

    _assert_feasible(result, x, a, dual_sys_params)


def test_dual_zero_demand_cell(
    dual_sys_params: dict[str, NDArray[np.float64] | float],
) -> None:
    """Zero-demand cells should keep xi=1 without served demand."""
    solver = P1DualSolver(S=2, C=2, M=3, sys_params=dual_sys_params)
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    a = np.array([[0.0, 0.0, 0.0], [5e5, 4e5, 6e5]], dtype=np.float64)
    g = np.full((2, 2), 1e-12, dtype=np.float64)

    result = solver.solve(x, a, g)

    assert result.zero_demand.tolist() == [True, False]
    np.testing.assert_allclose(result.xi[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(result.z[0, :], 0.0, atol=1e-12)
    _assert_feasible(result, x, a, dual_sys_params)


def test_dual_matches_cvx_on_high_demand_toy_instances(
    dual_sys_params: dict[str, NDArray[np.float64] | float],
) -> None:
    """L2 should stay close to L1 for overloaded toy instances."""
    gaps: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = np.zeros((2, 5), dtype=np.float64)
        x[rng.integers(0, 2, size=5), np.arange(5)] = 1.0
        a = rng.uniform(2e5, 1e6, size=(5, 5))
        g = rng.uniform(5e-13, 2e-12, size=(2, 5))

        cvx_result = P1CVXSolver(2, 5, 5, dual_sys_params, solver="ECOS").solve(x, a, g)
        dual_result = P1DualSolver(2, 5, 5, dual_sys_params).solve(x, a, g)

        _assert_feasible(dual_result, x, a, dual_sys_params)
        gaps.append((cvx_result.U - dual_result.U) / abs(cvx_result.U))
        assert dual_result.solve_time < cvx_result.solve_time

    assert float(np.median(gaps)) <= 0.01
    assert float(np.quantile(gaps, 0.95)) <= 0.02


def test_dual_invalid_inputs_raise_value_error(
    dual_sys_params: dict[str, NDArray[np.float64] | float],
) -> None:
    """L2 should reject invalid inputs before numerical allocation."""
    solver = P1DualSolver(S=2, C=2, M=2, sys_params=dual_sys_params)
    x = np.array([[1.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    a = np.ones((2, 2), dtype=np.float64)
    g = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="x must be binary"):
        solver.solve(x, a, g)
