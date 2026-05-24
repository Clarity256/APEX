"""Fast NumPy approximation for the P1 dual-decomposition layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers.p1_cvx import P1Result

FloatArray = NDArray[np.float64]


class P1DualSolver:
    """Solve P1 with a fast dual-weighted water-filling approximation."""

    def __init__(
        self,
        S: int,  # noqa: N803
        C: int,  # noqa: N803
        M: int,  # noqa: N803
        sys_params: Mapping[str, object],
        tol: float = 1e-3,
        max_iter: int = 200,
        step_init: float = 1.0,
    ) -> None:
        """
        Initialize the fast P1 approximation.

        Parameters
        ----------
        S, C, M : int
            Number of satellites, cells, and fast slots.
        sys_params : mapping
            Must contain N_PRB, P_max, W_PRB, N0, T_f, and eps.
        tol : float, default=1e-3
            Fixed-point tolerance on satisfaction-rate updates.
        max_iter : int, default=200
            Maximum number of dual-weight updates.
        step_init : float, default=1.0
            Initial damping step for dual-weight feedback.
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
        self.tol = _positive_float(tol, "tol")
        self.max_iter = _positive_int(max_iter, "max_iter")
        self.step_init = _positive_float(step_init, "step_init")

    def solve(self, x: FloatArray, a: FloatArray, g: FloatArray) -> P1Result:
        """
        Solve the fast-slot PRB-power allocation problem approximately.

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
            Feasible approximate PRB, power, served-demand, and satisfaction values.
        """
        start_time = perf_counter()
        x_checked, a_checked, g_checked = self._validate_inputs(x, a, g)
        state = _initial_state(a_checked)
        best_result = self._result_from_state(x_checked, a_checked, g_checked, state, "dual_init")
        status = "dual_max_iter"
        for iteration in range(self.max_iter):
            n, p = self._water_filling(x_checked, g_checked, state)
            capacity = self._compute_capacity(n, p, g_checked)
            z, need = _compute_served_demand(capacity, a_checked)
            xi = _compute_xi(z, a_checked)
            candidate = _DualState(xi=xi, need=need, demand_total=state.demand_total)
            result = self._build_result(n, p, z, xi, start_time, "dual_running")
            if result.U > best_result.U:
                best_result = result
            if float(np.max(np.abs(candidate.xi - state.xi))) <= self.tol:
                status = f"dual_converged_{iteration + 1}"
                best_result.status = status
                best_result.solve_time = perf_counter() - start_time
                return best_result
            state = _mix_state(state, candidate, self._step(iteration))
        best_result.status = status
        best_result.solve_time = perf_counter() - start_time
        return best_result

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

    def _result_from_state(
        self,
        x: FloatArray,
        a: FloatArray,
        g: FloatArray,
        state: _DualState,
        status: str,
    ) -> P1Result:
        n, p = self._water_filling(x, g, state)
        capacity = self._compute_capacity(n, p, g)
        z, _ = _compute_served_demand(capacity, a)
        xi = _compute_xi(z, a)
        return self._build_result(n, p, z, xi, perf_counter(), status)

    def _water_filling(
        self,
        x: FloatArray,
        g: FloatArray,
        state: _DualState,
    ) -> tuple[FloatArray, FloatArray]:
        spectral_efficiency = self._spectral_efficiency(g)
        cell_weight = self._cell_weight(state.xi, state.demand_total)
        slot_weight = self._slot_weight(cell_weight, state.need)
        score = x[:, :, None] * spectral_efficiency[:, :, None] * slot_weight[None, :, :]
        share = _normalize_by_satellite_slot(score)
        n = self.N_PRB[:, None, None] * share
        p = self.P_max[:, None, None] * share
        return n, p

    def _spectral_efficiency(self, g: FloatArray) -> FloatArray:
        power_per_prb = self.P_max / self.N_PRB
        snr = g * power_per_prb[:, None] / (self.W_PRB * self.N0)
        return np.asarray(self.T_f * self.W_PRB * np.log2(1.0 + snr), dtype=np.float64)

    def _cell_weight(self, xi: FloatArray, demand_total: FloatArray) -> FloatArray:
        positive = demand_total > 0.0
        weight = np.zeros(self.C, dtype=np.float64)
        if not np.any(positive):
            return weight
        demand_scale = float(np.mean(demand_total[positive]))
        weight[positive] = demand_scale / demand_total[positive]
        weight[positive] /= self.eps + np.clip(xi[positive], 0.0, 1.0)
        return weight

    def _slot_weight(self, cell_weight: FloatArray, need: FloatArray) -> FloatArray:
        scale = np.maximum(np.mean(need, axis=1, keepdims=True), 1.0)
        return np.asarray(cell_weight[:, None] * need / scale, dtype=np.float64)

    def _compute_capacity(self, n: FloatArray, p: FloatArray, g: FloatArray) -> FloatArray:
        safe_n = np.where(n > 0.0, n, 1.0)
        snr = p * g[:, :, None] / (self.W_PRB * safe_n * self.N0)
        rate = self.T_f * self.W_PRB * n * np.log2(1.0 + np.maximum(snr, 0.0))
        return np.asarray(rate.sum(axis=0), dtype=np.float64)

    def _build_result(
        self,
        n: FloatArray,
        p: FloatArray,
        z: FloatArray,
        xi: FloatArray,
        start_time: float,
        status: str,
    ) -> P1Result:
        zero_demand = np.isclose(z.sum(axis=1), 0.0) & np.isclose(xi, 1.0)
        return P1Result(
            n=_clean_nonnegative(n),
            p=_clean_nonnegative(p),
            z=_clean_nonnegative(z),
            xi=np.clip(xi, 0.0, 1.0),
            U=float(np.sum(np.log(self.eps + np.clip(xi, 0.0, 1.0)))),
            status=status,
            solve_time=perf_counter() - start_time,
            zero_demand=zero_demand,
        )

    def _step(self, iteration: int) -> float:
        return float(min(1.0, self.step_init / np.sqrt(iteration + 1.0)))


@dataclass
class _DualState:
    """State carried by dual-weight fixed-point iterations."""

    xi: FloatArray
    need: FloatArray
    demand_total: FloatArray


def _initial_state(a: FloatArray) -> _DualState:
    demand_total = a.sum(axis=1)
    xi = np.where(demand_total > 0.0, 0.5, 1.0)
    need = _initial_need(a)
    return _DualState(xi=xi.astype(np.float64), need=need, demand_total=demand_total)


def _initial_need(a: FloatArray) -> FloatArray:
    backlog = np.zeros(a.shape[0], dtype=np.float64)
    need = np.zeros_like(a)
    for m in range(a.shape[1]):
        need[:, m] = backlog + a[:, m]
        backlog += a[:, m]
    return need


def _mix_state(current: _DualState, candidate: _DualState, step: float) -> _DualState:
    xi = (1.0 - step) * current.xi + step * candidate.xi
    need = (1.0 - step) * current.need + step * candidate.need
    return _DualState(
        xi=np.asarray(xi, dtype=np.float64),
        need=np.maximum(need, 0.0),
        demand_total=current.demand_total,
    )


def _compute_served_demand(capacity: FloatArray, a: FloatArray) -> tuple[FloatArray, FloatArray]:
    backlog = np.zeros(a.shape[0], dtype=np.float64)
    z = np.zeros_like(a)
    need = np.zeros_like(a)
    for m in range(a.shape[1]):
        available = backlog + a[:, m]
        need[:, m] = available
        z[:, m] = np.minimum(capacity[:, m], available)
        backlog = available - z[:, m]
    return z, need


def _compute_xi(z: FloatArray, a: FloatArray) -> FloatArray:
    demand_total = a.sum(axis=1)
    xi = np.ones(a.shape[0], dtype=np.float64)
    positive = demand_total > 0.0
    xi[positive] = np.minimum(z[positive].sum(axis=1) / demand_total[positive], 1.0)
    return xi


def _normalize_by_satellite_slot(score: FloatArray) -> FloatArray:
    denom = score.sum(axis=1, keepdims=True)
    share = np.divide(score, denom, out=np.zeros_like(score), where=denom > 0.0)
    return np.asarray(share, dtype=np.float64)


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return checked


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


def _clean_nonnegative(value: FloatArray) -> FloatArray:
    return np.maximum(value, 0.0)
