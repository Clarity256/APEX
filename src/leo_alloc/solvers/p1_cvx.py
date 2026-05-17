"""CVXPY implementation of the P1 fast-slot convex allocation kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cvxpy as _cp
import numpy as np
from cvxpy.error import SolverError
from numpy.typing import NDArray

cp: Any = _cp
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_ACCEPTED_STATUSES = {"optimal", "optimal_inaccurate"}


@dataclass
class P1Result:
    """Container for one P1 convex optimization result."""

    n: FloatArray
    p: FloatArray
    z: FloatArray
    xi: FloatArray
    U: float
    status: str
    solve_time: float
    zero_demand: BoolArray


class P1CVXSolver:
    """Solve P1 with CVXPY using the perspective-function convexification."""

    def __init__(
        self,
        S: int,  # noqa: N803
        C: int,  # noqa: N803
        M: int,  # noqa: N803
        sys_params: Mapping[str, object],
        solver: str = "MOSEK",
        use_dpp: bool = True,
    ) -> None:
        """
        Initialize the reusable CVXPY model for P1.

        Parameters
        ----------
        S, C, M : int
            Number of satellites, cells, and fast slots.
        sys_params : mapping
            Must contain N_PRB, P_max, W_PRB, N0, T_f, and eps.
        solver : str, default="MOSEK"
            Preferred CVXPY solver. CLARABEL and ECOS are fallback solvers.
        use_dpp : bool, default=True
            Whether to ask CVXPY to use DPP parameter caching.
        """
        self.S = _positive_int(S, "S")
        self.C = _positive_int(C, "C")
        self.M = _positive_int(M, "M")
        self.N_PRB = _read_positive_vector(sys_params, "N_PRB", self.S)
        self.P_max = _read_positive_vector(sys_params, "P_max", self.S)
        self.W_PRB = _read_positive_scalar(sys_params, "W_PRB")
        self.N0 = _read_positive_scalar(sys_params, "N0")
        self.T_f = _read_positive_scalar(sys_params, "T_f")
        self.eps = _read_positive_scalar(sys_params, "eps")
        self.solver = solver
        self.use_dpp = use_dpp
        self.bit_scale = 1e6
        self._pair_count = self.S * self.C
        self._satellite_pair_matrix = _satellite_pair_matrix(self.S, self.C)
        self._cell_pair_matrix = _cell_pair_matrix(self.S, self.C)
        self._build_problem()

    def solve(self, x: FloatArray, a: FloatArray, g: FloatArray) -> P1Result:
        """
        Solve the fast-slot PRB-power allocation problem.

        Parameters
        ----------
        x : ndarray of shape (S, C)
            Binary association matrix from the slow-slot decision.
        a : ndarray of shape (C, M)
            Per-fast-slot demand arrivals in bits.
        g : ndarray of shape (S, C)
            Large-scale channel gain in linear scale, not dB.

        Returns
        -------
        P1Result
            Optimal PRB, power, served-demand, and satisfaction-rate values.

        Notes
        -----
        Uses n * log(1 + alpha * p / n) = -rel_entr(n, n + alpha * p).
        """
        x_checked, a_checked, g_checked = self._validate_inputs(x, a, g)
        self._set_parameter_values(x_checked, a_checked, g_checked)
        status, solve_time = self._solve_with_fallbacks()
        return self._extract_result(a_checked, status, solve_time)

    def _build_problem(self) -> None:
        self.n = cp.Variable((self._pair_count, self.M), nonneg=True)
        self.p = cp.Variable((self._pair_count, self.M), nonneg=True)
        self.z = cp.Variable((self.C, self.M), nonneg=True)
        self.xi = cp.Variable(self.C, nonneg=True)
        self._x_param = cp.Parameter((self._pair_count, self.M), nonneg=True)
        self._a_param = cp.Parameter((self.C, self.M), nonneg=True)
        self._alpha_param = cp.Parameter((self._pair_count, self.M), nonneg=True)

        n_limit = np.repeat(self.N_PRB, self.C)[:, None]
        p_limit = np.repeat(self.P_max, self.C)[:, None]
        rate = self._rate_scale() * (
            -cp.rel_entr(self.n, self.n + cp.multiply(self._alpha_param, self.p))
        )
        constraints = [
            self._satellite_pair_matrix @ self.n <= self.N_PRB[:, None],
            self._satellite_pair_matrix @ self.p <= self.P_max[:, None],
            self.n <= cp.multiply(n_limit, self._x_param),
            self.p <= cp.multiply(p_limit, self._x_param),
            self.z <= self._cell_pair_matrix @ rate,
            cp.cumsum(self.z, axis=1) <= cp.cumsum(self._a_param, axis=1),
            cp.multiply(cp.sum(self._a_param, axis=1), self.xi) <= cp.sum(self.z, axis=1),
            self.xi <= 1.0,
        ]
        objective = cp.Maximize(cp.sum(cp.log(self.eps + self.xi)))
        self.prob = cp.Problem(objective, constraints)
        self._validate_problem_graph()

    def _rate_scale(self) -> float:
        return float(self.T_f * self.W_PRB / (self.bit_scale * np.log(2.0)))

    def _validate_problem_graph(self) -> None:
        is_dcp = bool(self.prob.is_dcp(dpp=self.use_dpp))
        if not is_dcp:
            mode = "DCP/DPP" if self.use_dpp else "DCP"
            raise ValueError(f"P1 CVXPY problem is not {mode} compliant")

    def _validate_inputs(
        self,
        x: FloatArray,
        a: FloatArray,
        g: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        x_checked = _as_float_array(x, "x", (self.S, self.C))
        a_checked = _as_float_array(a, "a", (self.C, self.M))
        g_checked = _as_float_array(g, "g", (self.S, self.C))
        if not np.all(np.isclose(x_checked, 0.0) | np.isclose(x_checked, 1.0)):
            raise ValueError("x must be binary")
        if not np.allclose(x_checked.sum(axis=0), 1.0):
            raise ValueError("Each cell must have exactly one satellite association")
        if np.any(a_checked < 0.0):
            raise ValueError("a must be non-negative")
        if np.any(g_checked < 0.0):
            raise ValueError("g must be non-negative")
        return np.round(x_checked), a_checked, g_checked

    def _set_parameter_values(self, x: FloatArray, a: FloatArray, g: FloatArray) -> None:
        alpha = g / (self.W_PRB * self.N0)
        self._x_param.value = np.repeat(x.reshape(self._pair_count, 1), self.M, axis=1)
        self._a_param.value = a / self.bit_scale
        self._alpha_param.value = np.repeat(alpha.reshape(self._pair_count, 1), self.M, axis=1)

    def _solve_with_fallbacks(self) -> tuple[str, float]:
        start_time = perf_counter()
        last_status = "not_run"
        last_error: SolverError | None = None
        for solver_name in self._candidate_solvers():
            try:
                self.prob.solve(
                    solver=solver_name,
                    warm_start=True,
                    ignore_dpp=not self.use_dpp,
                )
            except SolverError as exc:
                last_error = exc
                continue
            last_status = str(self.prob.status)
            if last_status in _ACCEPTED_STATUSES:
                return last_status, perf_counter() - start_time
        if last_error is not None:
            message = f"P1Solver failed after fallback attempts: {last_error}"
            raise RuntimeError(message) from last_error
        raise RuntimeError(f"P1Solver failed with status {last_status}")

    def _candidate_solvers(self) -> list[str]:
        installed = {str(name) for name in cp.installed_solvers()}
        ordered = [self.solver, "CLARABEL", "ECOS"]
        candidates: list[str] = []
        for solver_name in ordered:
            if solver_name in installed and solver_name not in candidates:
                candidates.append(solver_name)
        if not candidates:
            raise RuntimeError("No supported CVXPY solver is installed")
        return candidates

    def _extract_result(self, a: FloatArray, status: str, solve_time: float) -> P1Result:
        n = _clean_nonnegative(_value_array(self.n.value, "n")).reshape(self.S, self.C, self.M)
        p = _clean_nonnegative(_value_array(self.p.value, "p")).reshape(self.S, self.C, self.M)
        z = _clean_nonnegative(_value_array(self.z.value, "z")) * self.bit_scale
        xi = np.clip(_clean_nonnegative(_value_array(self.xi.value, "xi")), 0.0, 1.0)
        zero_demand = np.isclose(a.sum(axis=1), 0.0)
        xi[zero_demand] = 1.0
        z[zero_demand, :] = 0.0
        return P1Result(
            n=n,
            p=p,
            z=z,
            xi=xi,
            U=float(np.sum(np.log(self.eps + xi))),
            status=status,
            solve_time=solve_time,
            zero_demand=zero_demand,
        )


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_positive_vector(
    sys_params: Mapping[str, object],
    key: str,
    expected_length: int,
) -> FloatArray:
    if key not in sys_params:
        raise ValueError(f"sys_params must include {key}")
    value = np.asarray(sys_params[key], dtype=np.float64)
    if value.shape != (expected_length,):
        raise ValueError(f"{key} must have shape ({expected_length},)")
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"{key} must contain positive finite values")
    return value


def _read_positive_scalar(sys_params: Mapping[str, object], key: str) -> float:
    if key not in sys_params:
        raise ValueError(f"sys_params must include {key}")
    value = float(np.asarray(sys_params[key], dtype=np.float64))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite scalar")
    return value


def _as_float_array(value: FloatArray, name: str, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _value_array(value: Any, name: str) -> FloatArray:
    if value is None:
        raise RuntimeError(f"CVXPY did not return a value for {name}")
    return np.asarray(value, dtype=np.float64)


def _clean_nonnegative(value: FloatArray) -> FloatArray:
    return np.maximum(value, 0.0)


def _satellite_pair_matrix(satellite_count: int, cell_count: int) -> FloatArray:
    return np.repeat(np.eye(satellite_count, dtype=np.float64), cell_count, axis=1)


def _cell_pair_matrix(satellite_count: int, cell_count: int) -> FloatArray:
    matrix = np.zeros((cell_count, satellite_count * cell_count), dtype=np.float64)
    for s in range(satellite_count):
        for c in range(cell_count):
            matrix[c, s * cell_count + c] = 1.0
    return matrix
