"""Rolling-window decomposition solver for P2 association optimization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers.p2_milp import P2MILPSolver, P2Result

FloatArray = NDArray[np.float64]
BudgetPolicy = Literal["remaining", "proportional"]


class _ScenarioLike(Protocol):
    """Protocol for ScenarioInstance-compatible objects consumed by P2 rolling."""

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
class _WindowScenario:
    """Scenario-compatible view for one rolling MILP window."""

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


class P2RollingSolver:
    """Solve P2 by repeatedly optimizing and committing rolling MILP windows."""

    def __init__(
        self,
        scenario: _ScenarioLike,
        window: int = 5,
        step: int = 3,
        time_limit: float = 3600.0,
        mip_gap: float = 0.01,
        budget_policy: BudgetPolicy = "remaining",
    ) -> None:
        """
        Initialize a rolling-window P2 solver.

        Parameters
        ----------
        scenario : object
            ScenarioInstance-compatible object.
        window : int, default=5
            Number of slow slots included in each MILP subproblem.
        step : int, default=3
            Number of leading slow slots committed from each subproblem.
        time_limit : float, default=3600.0
            Per-window MILP wall-clock limit in seconds.
        mip_gap : float, default=0.01
            Relative MIP gap target for each subproblem.
        budget_policy : {"remaining", "proportional"}, default="remaining"
            Window handover budget allocation rule. ``remaining`` exposes all
            currently unused budget to each window; committed switches still keep
            the final trajectory within the hard budget. ``proportional`` uses a
            rounded horizon-length share of the remaining budget.
        """
        self.scenario = scenario
        self.window = _positive_int(window, "window")
        self.step = _positive_int(step, "step")
        if self.step > self.window:
            raise ValueError("step must be less than or equal to window")
        self.time_limit = _positive_float(time_limit, "time_limit")
        self.mip_gap = _nonnegative_float(mip_gap, "mip_gap")
        if budget_policy not in {"remaining", "proportional"}:
            raise ValueError("budget_policy must be 'remaining' or 'proportional'")
        self.budget_policy: BudgetPolicy = budget_policy

    def solve(self) -> P2Result:
        """
        Run the rolling-window decomposition and return a full-horizon result.

        Returns
        -------
        P2Result
            Full association trajectory, committed handovers, proxy satisfaction,
            and aggregate solver diagnostics.
        """
        start_time = perf_counter()
        x_all = np.zeros((self.scenario.S, self.scenario.C, self.scenario.K), dtype=np.float64)
        h_all = np.zeros((self.scenario.C, self.scenario.K), dtype=np.float64)
        xi_all = np.zeros((self.scenario.C, self.scenario.K), dtype=np.float64)
        remaining_budget = np.asarray(self.scenario.H, dtype=np.float64).copy()
        x_previous: FloatArray | None = None
        mip_gaps: list[float] = []
        windows_solved = 0
        start = 0

        while start < self.scenario.K:
            end = min(start + self.window, self.scenario.K)
            commit_end = min(start + self.step, self.scenario.K)
            window_budget = self._window_budget(start, end, remaining_budget)
            sub_result = P2MILPSolver(
                self._window_scenario(start, end, window_budget),
                time_limit=self.time_limit,
                mip_gap=self.mip_gap,
                initial_x=x_previous,
            ).solve()
            windows_solved += 1
            mip_gaps.append(sub_result.mip_gap)
            if sub_result.status.startswith("infeasible"):
                return self._infeasible_result(
                    status=f"infeasible at window [{start}, {end}): {sub_result.status}",
                    solve_time=perf_counter() - start_time,
                    x=x_all,
                    h=h_all,
                    xi=xi_all,
                    mip_gaps=mip_gaps,
                )

            commit_len = commit_end - start
            x_all[:, :, start:commit_end] = sub_result.x[:, :, :commit_len]
            h_all[:, start:commit_end] = sub_result.h[:, :commit_len]
            xi_all[:, start:commit_end] = sub_result.xi[:, :commit_len]

            used_budget = sub_result.h[:, :commit_len].sum(axis=1)
            if np.any(used_budget > remaining_budget + 1e-7):
                return self._infeasible_result(
                    status=f"rolling budget violation at window [{start}, {end})",
                    solve_time=perf_counter() - start_time,
                    x=x_all,
                    h=h_all,
                    xi=xi_all,
                    mip_gaps=mip_gaps,
                )
            remaining_budget = np.maximum(remaining_budget - used_budget, 0.0)
            x_previous = x_all[:, :, commit_end - 1].copy()
            start = commit_end

        solve_time = perf_counter() - start_time
        return P2Result(
            x=x_all,
            h=h_all,
            xi=xi_all,
            U=_p2_objective(xi_all, h_all, self.scenario.eps, self.scenario.lambda_h),
            handover_per_cell=h_all.sum(axis=1),
            solve_time=solve_time,
            mip_gap=_aggregate_mip_gap(mip_gaps),
            status=f"rolling_optimal windows={windows_solved}",
        )

    def _window_scenario(
        self,
        start: int,
        end: int,
        handover_budget: FloatArray,
    ) -> _WindowScenario:
        return _WindowScenario(
            S=self.scenario.S,
            C=self.scenario.C,
            K=end - start,
            M=self.scenario.M,
            g=np.asarray(self.scenario.g[:, :, start:end], dtype=np.float64),
            v=np.asarray(self.scenario.v[:, :, start:end], dtype=np.float64),
            a=np.asarray(self.scenario.a[:, start:end, :], dtype=np.float64),
            N_PRB=np.asarray(self.scenario.N_PRB, dtype=np.float64),
            P_max=np.asarray(self.scenario.P_max, dtype=np.float64),
            H=handover_budget.copy(),
            W_PRB=float(self.scenario.W_PRB),
            N0=float(self.scenario.N0),
            T_f=float(self.scenario.T_f),
            eps=float(self.scenario.eps),
            lambda_h=float(self.scenario.lambda_h),
        )

    def _window_budget(
        self,
        start: int,
        end: int,
        remaining_budget: FloatArray,
    ) -> FloatArray:
        if self.budget_policy == "remaining":
            return remaining_budget.copy()
        remaining_horizon = max(self.scenario.K - start, 1)
        window_fraction = (end - start) / remaining_horizon
        proportional_budget = np.ceil(remaining_budget * window_fraction - 1e-12)
        return np.minimum(remaining_budget, np.maximum(proportional_budget, 0.0))

    def _infeasible_result(
        self,
        status: str,
        solve_time: float,
        x: FloatArray,
        h: FloatArray,
        xi: FloatArray,
        mip_gaps: list[float],
    ) -> P2Result:
        return P2Result(
            x=x,
            h=h,
            xi=xi,
            U=float("-inf"),
            handover_per_cell=h.sum(axis=1),
            solve_time=solve_time,
            mip_gap=_aggregate_mip_gap(mip_gaps),
            status=status,
        )


def _p2_objective(xi: FloatArray, h: FloatArray, eps: float, lambda_h: float) -> float:
    return float(np.sum(np.log(eps + np.clip(xi, 0.0, 1.0))) - lambda_h * np.sum(h))


def _aggregate_mip_gap(mip_gaps: list[float]) -> float:
    finite_gaps = np.array([gap for gap in mip_gaps if np.isfinite(gap)], dtype=np.float64)
    if finite_gaps.size == 0:
        return float("nan")
    return float(np.max(finite_gaps))


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
