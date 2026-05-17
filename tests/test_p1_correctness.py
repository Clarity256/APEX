"""Correctness tests for the P1 convex optimization kernel."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from leo_alloc.solvers import P1CVXSolver


@pytest.fixture
def sys_params_default() -> dict[str, NDArray[np.float64] | float]:
    """Return a small, numerically stable P1 parameter set."""
    return {
        "N_PRB": np.array([100.0], dtype=np.float64),
        "P_max": np.array([100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }


def _analytical_rate_bits(
    n: float,
    p: float,
    g: float,
    sys_params: dict[str, NDArray[np.float64] | float],
) -> float:
    """Compute the scalar Shannon-style served bits expression."""
    w_prb = float(sys_params["W_PRB"])
    n0 = float(sys_params["N0"])
    t_f = float(sys_params["T_f"])
    return t_f * w_prb * n * np.log2(1.0 + p * g / (w_prb * n * n0))


def test_single_sat_single_cell_single_slot(
    sys_params_default: dict[str, NDArray[np.float64] | float],
) -> None:
    """S=C=M=1 should allocate all resources to the only cell."""
    solver = P1CVXSolver(S=1, C=1, M=1, sys_params=sys_params_default)
    x = np.array([[1.0]], dtype=np.float64)
    a = np.array([[1e6]], dtype=np.float64)
    g = np.array([[1e-12]], dtype=np.float64)

    result = solver.solve(x, a, g)

    assert result.status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(result.n[0, 0, 0], 100.0, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(result.p[0, 0, 0], 100.0, rtol=1e-3, atol=1e-3)

    rate_bits = _analytical_rate_bits(100.0, 100.0, 1e-12, sys_params_default)
    np.testing.assert_allclose(result.z[0, 0], rate_bits, rtol=5e-3, atol=1e-2)
    np.testing.assert_allclose(result.xi[0], rate_bits / a[0, 0], rtol=5e-3)


def test_symmetric_two_cells() -> None:
    """Two identical cells should receive symmetric allocations."""
    sys_params = {
        "N_PRB": np.array([100.0], dtype=np.float64),
        "P_max": np.array([100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }
    solver = P1CVXSolver(S=1, C=2, M=1, sys_params=sys_params)
    x = np.array([[1.0, 1.0]], dtype=np.float64)
    a = np.array([[1e6], [1e6]], dtype=np.float64)
    g = np.array([[1e-12, 1e-12]], dtype=np.float64)

    result = solver.solve(x, a, g)

    np.testing.assert_allclose(result.n[0, 0, 0], result.n[0, 1, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(result.p[0, 0, 0], result.p[0, 1, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(result.xi[0], result.xi[1], rtol=1e-3, atol=1e-6)


def test_association_mask_forces_unassigned_pair_to_zero() -> None:
    """Pairs with x=0 must have zero PRB and power allocation."""
    sys_params = {
        "N_PRB": np.array([100.0, 100.0], dtype=np.float64),
        "P_max": np.array([100.0, 100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }
    solver = P1CVXSolver(S=2, C=2, M=2, sys_params=sys_params)
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    a = np.full((2, 2), 5e5, dtype=np.float64)
    g = np.full((2, 2), 1e-12, dtype=np.float64)

    result = solver.solve(x, a, g)

    np.testing.assert_allclose(result.n[0, 1, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(result.p[0, 1, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(result.n[1, 0, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(result.p[1, 0, :], 0.0, atol=1e-5)


def test_zero_demand_cell_gets_unit_satisfaction() -> None:
    """A zero-demand cell should be marked and assigned xi=1."""
    sys_params = {
        "N_PRB": np.array([100.0], dtype=np.float64),
        "P_max": np.array([100.0], dtype=np.float64),
        "W_PRB": 180e3,
        "N0": 1e-15,
        "T_f": 0.01,
        "eps": 1e-4,
    }
    solver = P1CVXSolver(S=1, C=2, M=2, sys_params=sys_params)
    x = np.array([[1.0, 1.0]], dtype=np.float64)
    a = np.array([[0.0, 0.0], [5e5, 5e5]], dtype=np.float64)
    g = np.array([[1e-12, 1e-12]], dtype=np.float64)

    result = solver.solve(x, a, g)

    assert result.zero_demand.tolist() == [True, False]
    np.testing.assert_allclose(result.xi[0], 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.z[0, :], 0.0, atol=1e-5)


def test_large_demand_scale_is_stable(
    sys_params_default: dict[str, NDArray[np.float64] | float],
) -> None:
    """Large bit-scale demand should solve without invalid outputs."""
    solver = P1CVXSolver(S=1, C=1, M=2, sys_params=sys_params_default)
    x = np.array([[1.0]], dtype=np.float64)
    a = np.array([[1e9, 1e9]], dtype=np.float64)
    g = np.array([[1e-12]], dtype=np.float64)

    result = solver.solve(x, a, g)

    assert np.isfinite(result.U)
    assert np.all(np.isfinite(result.z))
    assert np.all((result.xi >= -1e-8) & (result.xi <= 1.0 + 1e-8))


@pytest.mark.parametrize(
    ("x", "a", "g", "match"),
    [
        (
            np.ones((1, 2), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            "x must have shape",
        ),
        (
            np.array([[0.5]], dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            "x must be binary",
        ),
        (
            np.array([[1.0]], dtype=np.float64),
            np.array([[-1.0]], dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            "a must be non-negative",
        ),
        (
            np.array([[1.0]], dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            np.array([[-1.0]], dtype=np.float64),
            "g must be non-negative",
        ),
    ],
)
def test_invalid_inputs_raise_value_error(
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    g: NDArray[np.float64],
    match: str,
    sys_params_default: dict[str, NDArray[np.float64] | float],
) -> None:
    """Invalid P1 inputs should fail before reaching CVXPY."""
    solver = P1CVXSolver(S=1, C=1, M=1, sys_params=sys_params_default)

    with pytest.raises(ValueError, match=match):
        solver.solve(x, a, g)
