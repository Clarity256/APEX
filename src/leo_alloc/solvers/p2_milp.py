"""MILP solver for P2 slow-slot satellite-cell association."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _ScenarioLike(Protocol):
    """Protocol for ScenarioInstance-compatible objects consumed by P2."""

    S: int
    C: int
    K: int
    M: int
    g: FloatArray
    v: FloatArray
    a: FloatArray
    N_PRB: FloatArray
    P_max: FloatArray
    H: FloatArray
    W_PRB: float
    N0: float
    T_f: float
    eps: float
    lambda_h: float


@dataclass
class P2Result:
    """Container for one P2 MILP association result."""

    x: FloatArray
    h: FloatArray
    xi: FloatArray
    U: float
    handover_per_cell: FloatArray
    solve_time: float
    mip_gap: float
    status: str


@dataclass(frozen=True)
class _P2Data:
    """Validated P2 arrays and constants extracted from a scenario object."""

    S: int
    C: int
    K: int
    M: int
    g: FloatArray
    v: FloatArray
    a: FloatArray
    N_PRB: FloatArray
    P_max: FloatArray
    H: FloatArray
    W_PRB: float
    N0: float
    T_f: float
    eps: float
    lambda_h: float


@dataclass(frozen=True)
class _Index:
    """Flat MILP variable index helper."""

    S: int
    C: int
    K: int

    @property
    def x_size(self) -> int:
        return self.S * self.C * self.K

    @property
    def h_offset(self) -> int:
        return self.x_size

    @property
    def h_size(self) -> int:
        return self.C * self.K

    @property
    def xi_offset(self) -> int:
        return self.h_offset + self.h_size

    @property
    def xi_size(self) -> int:
        return self.C * self.K

    @property
    def total_size(self) -> int:
        return self.x_size + self.h_size + self.xi_size

    def x(self, s: int, c: int, k: int) -> int:
        """Return the flat index for x[s,c,k]."""
        return (s * self.C + c) * self.K + k

    def h(self, c: int, k: int) -> int:
        """Return the flat index for h[c,k]."""
        return self.h_offset + c * self.K + k

    def xi(self, c: int, k: int) -> int:
        """Return the flat index for xi[c,k]."""
        return self.xi_offset + c * self.K + k


class P2MILPSolver:
    """Solve the offline P2 association problem with a linear MILP surrogate."""

    def __init__(
        self,
        scenario: _ScenarioLike,
        time_limit: float = 3600.0,
        mip_gap: float = 0.01,
    ) -> None:
        """
        Initialize a P2 MILP solver.

        Parameters
        ----------
        scenario : object
            Object with ScenarioInstance-compatible attributes from the architecture doc.
        time_limit : float, default=3600.0
            Solver wall-clock limit in seconds.
        mip_gap : float, default=0.01
            Relative MIP gap target passed to HiGHS.
        """
        self.data = _scenario_data(scenario)
        self.time_limit = _positive_float(time_limit, "time_limit")
        self.mip_gap = _nonnegative_float(mip_gap, "mip_gap")
        self.index = _Index(self.data.S, self.data.C, self.data.K)
        self.capacity_proxy = _capacity_proxy(self.data)

    def solve(self) -> P2Result:
        """
        Solve the full-window P2 association MILP.

        Returns
        -------
        P2Result
            Association, handover indicators, proxy satisfaction rates, and status.
        """
        start_time = perf_counter()
        constraints = self._constraints()
        result = milp(
            c=self._objective_vector(),
            integrality=self._integrality(),
            bounds=self._bounds(),
            constraints=constraints,
            options={"time_limit": self.time_limit, "mip_rel_gap": self.mip_gap},
        )
        solve_time = perf_counter() - start_time
        if result.x is None or not bool(result.success):
            return self._infeasible_result(str(result.message), solve_time, result)
        return self._extract_result(np.asarray(result.x, dtype=np.float64), solve_time, result)

    def _objective_vector(self) -> FloatArray:
        objective = np.zeros(self.index.total_size, dtype=np.float64)
        for c in range(self.data.C):
            for k in range(self.data.K):
                objective[self.index.xi(c, k)] = -1.0
                objective[self.index.h(c, k)] = self.data.lambda_h
        return objective

    def _integrality(self) -> IntArray:
        integrality = np.zeros(self.index.total_size, dtype=np.int64)
        integrality[: self.index.x_size + self.index.h_size] = 1
        return integrality

    def _bounds(self) -> Bounds:
        lower = np.zeros(self.index.total_size, dtype=np.float64)
        upper = np.ones(self.index.total_size, dtype=np.float64)
        for s in range(self.data.S):
            for c in range(self.data.C):
                for k in range(self.data.K):
                    upper[self.index.x(s, c, k)] = self.data.v[s, c, k]
        for c in range(self.data.C):
            upper[self.index.h(c, 0)] = 0.0
        return Bounds(lower, upper)

    def _constraints(self) -> list[LinearConstraint]:
        return [
            self._association_constraints(),
            self._handover_constraints(),
            self._budget_constraints(),
            self._capacity_constraints(),
        ]

    def _association_constraints(self) -> LinearConstraint:
        rows = self.data.C * self.data.K
        matrix = lil_matrix((rows, self.index.total_size), dtype=np.float64)
        row = 0
        for c in range(self.data.C):
            for k in range(self.data.K):
                for s in range(self.data.S):
                    matrix[row, self.index.x(s, c, k)] = 1.0
                row += 1
        rhs = np.ones(rows, dtype=np.float64)
        return LinearConstraint(matrix.tocsr(), rhs, rhs)

    def _handover_constraints(self) -> LinearConstraint:
        rows = 2 * self.data.S * self.data.C * max(self.data.K - 1, 0)
        matrix = lil_matrix((rows, self.index.total_size), dtype=np.float64)
        row = 0
        for s in range(self.data.S):
            for c in range(self.data.C):
                for k in range(1, self.data.K):
                    matrix[row, self.index.x(s, c, k)] = 1.0
                    matrix[row, self.index.x(s, c, k - 1)] = -1.0
                    matrix[row, self.index.h(c, k)] = -1.0
                    row += 1
                    matrix[row, self.index.x(s, c, k)] = -1.0
                    matrix[row, self.index.x(s, c, k - 1)] = 1.0
                    matrix[row, self.index.h(c, k)] = -1.0
                    row += 1
        lower = np.full(rows, -np.inf, dtype=np.float64)
        upper = np.zeros(rows, dtype=np.float64)
        return LinearConstraint(matrix.tocsr(), lower, upper)

    def _budget_constraints(self) -> LinearConstraint:
        matrix = lil_matrix((self.data.C, self.index.total_size), dtype=np.float64)
        for c in range(self.data.C):
            for k in range(1, self.data.K):
                matrix[c, self.index.h(c, k)] = 1.0
        lower = np.full(self.data.C, -np.inf, dtype=np.float64)
        return LinearConstraint(matrix.tocsr(), lower, self.data.H)

    def _capacity_constraints(self) -> LinearConstraint:
        rows = self.data.C * self.data.K
        matrix = lil_matrix((rows, self.index.total_size), dtype=np.float64)
        row = 0
        for c in range(self.data.C):
            for k in range(self.data.K):
                matrix[row, self.index.xi(c, k)] = 1.0
                for s in range(self.data.S):
                    matrix[row, self.index.x(s, c, k)] = -self.capacity_proxy[s, c, k]
                row += 1
        lower = np.full(rows, -np.inf, dtype=np.float64)
        upper = np.zeros(rows, dtype=np.float64)
        return LinearConstraint(matrix.tocsr(), lower, upper)

    def _extract_result(self, solution: FloatArray, solve_time: float, raw_result: Any) -> P2Result:
        x = np.zeros((self.data.S, self.data.C, self.data.K), dtype=np.float64)
        h = np.zeros((self.data.C, self.data.K), dtype=np.float64)
        xi = np.zeros((self.data.C, self.data.K), dtype=np.float64)
        for s in range(self.data.S):
            for c in range(self.data.C):
                for k in range(self.data.K):
                    x[s, c, k] = np.round(solution[self.index.x(s, c, k)])
        for c in range(self.data.C):
            for k in range(self.data.K):
                h[c, k] = np.round(solution[self.index.h(c, k)])
                xi[c, k] = np.clip(solution[self.index.xi(c, k)], 0.0, 1.0)
        return P2Result(
            x=x,
            h=h,
            xi=xi,
            U=_p2_objective(xi, h, self.data.eps, self.data.lambda_h),
            handover_per_cell=h.sum(axis=1),
            solve_time=solve_time,
            mip_gap=_result_mip_gap(raw_result),
            status=str(raw_result.message),
        )

    def _infeasible_result(self, status: str, solve_time: float, raw_result: Any) -> P2Result:
        x = np.zeros((self.data.S, self.data.C, self.data.K), dtype=np.float64)
        h = np.zeros((self.data.C, self.data.K), dtype=np.float64)
        xi = np.zeros((self.data.C, self.data.K), dtype=np.float64)
        return P2Result(
            x=x,
            h=h,
            xi=xi,
            U=float("-inf"),
            handover_per_cell=np.zeros(self.data.C, dtype=np.float64),
            solve_time=solve_time,
            mip_gap=_result_mip_gap(raw_result),
            status=f"infeasible: {status}",
        )


def _scenario_data(scenario: _ScenarioLike) -> _P2Data:
    satellite_count = _positive_int(int(scenario.S), "S")
    cell_count = _positive_int(int(scenario.C), "C")
    slow_slot_count = _positive_int(int(scenario.K), "K")
    fast_slot_count = _positive_int(int(scenario.M), "M")
    return _P2Data(
        S=satellite_count,
        C=cell_count,
        K=slow_slot_count,
        M=fast_slot_count,
        g=_as_float_array(scenario.g, "g", (satellite_count, cell_count, slow_slot_count)),
        v=_binary_array(scenario.v, "v", (satellite_count, cell_count, slow_slot_count)),
        a=_as_float_array(scenario.a, "a", (cell_count, slow_slot_count, fast_slot_count)),
        N_PRB=_read_positive_vector(scenario.N_PRB, "N_PRB", satellite_count),
        P_max=_read_positive_vector(scenario.P_max, "P_max", satellite_count),
        H=_read_nonnegative_vector(scenario.H, "H", cell_count),
        W_PRB=_positive_float(float(scenario.W_PRB), "W_PRB"),
        N0=_positive_float(float(scenario.N0), "N0"),
        T_f=_positive_float(float(scenario.T_f), "T_f"),
        eps=_positive_float(float(scenario.eps), "eps"),
        lambda_h=_nonnegative_float(float(scenario.lambda_h), "lambda_h"),
    )


def _capacity_proxy(data: _P2Data) -> FloatArray:
    demand = data.a.sum(axis=2)
    expected_load = _expected_satellite_load(data.v, demand)
    power_per_prb = data.P_max / data.N_PRB
    snr = data.g * power_per_prb[:, None, None] / (data.W_PRB * data.N0)
    exclusive_bits = data.T_f * data.M * data.W_PRB * data.N_PRB[:, None, None]
    exclusive_bits = exclusive_bits * np.log2(1.0 + np.maximum(snr, 0.0))
    proxy = np.ones((data.S, data.C, data.K), dtype=np.float64)
    positive = demand > 0.0
    proxy[:, positive] = np.minimum(exclusive_bits[:, positive] / expected_load[:, positive], 1.0)
    return np.asarray(np.clip(proxy * data.v, 0.0, 1.0), dtype=np.float64)


def _expected_satellite_load(v: FloatArray, demand: FloatArray) -> FloatArray:
    visible_demand = np.einsum("sck,ck->sk", v, demand)
    visible_count = np.maximum(v.sum(axis=0), 1.0)
    load = visible_demand[:, None, :] / visible_count[None, :, :]
    fallback = np.maximum(demand.mean(), 1.0)
    return np.asarray(np.maximum(load, fallback), dtype=np.float64)


def _p2_objective(xi: FloatArray, h: FloatArray, eps: float, lambda_h: float) -> float:
    return float(np.sum(np.log(eps + np.clip(xi, 0.0, 1.0))) - lambda_h * np.sum(h))


def _result_mip_gap(raw_result: Any) -> float:
    mip_gap = getattr(raw_result, "mip_gap", np.nan)
    return float(mip_gap) if mip_gap is not None else float("nan")


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return value


def _as_float_array(value: object, name: str, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if name in {"a", "g"} and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _binary_array(value: object, name: str, shape: tuple[int, ...]) -> FloatArray:
    array = _as_float_array(value, name, shape)
    if not np.all(np.isclose(array, 0.0) | np.isclose(array, 1.0)):
        raise ValueError(f"{name} must be binary")
    return np.round(array)


def _read_positive_vector(value: object, name: str, expected_length: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},)")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return array


def _read_nonnegative_vector(value: object, name: str, expected_length: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},)")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain non-negative finite values")
    return array
