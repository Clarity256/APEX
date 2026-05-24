"""Dynamic-programming solver for P2 hard-budget association."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers.p2_milp import P2Result, _capacity_proxy, _p2_objective, _scenario_data

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _ScenarioLike(Protocol):
    """Protocol for ScenarioInstance-compatible objects consumed by P2 DP."""

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


class P2DPSolver:
    """Solve P2 exactly for the current separable linear surrogate via per-cell DP."""

    def __init__(self, scenario: _ScenarioLike) -> None:
        """
        Initialize a P2 dynamic-programming solver.

        The current P2 surrogate has no coupling across cells after the capacity
        proxy is precomputed, and the handover budget is per cell.  This makes
        the association problem separable by cell.
        """
        self.data = _scenario_data(scenario)
        self.capacity_proxy = _capacity_proxy(self.data)

    def solve(self) -> P2Result:
        """
        Solve the full-horizon P2 association problem with per-cell DP.

        Returns
        -------
        P2Result
            Association, handover indicators, proxy satisfaction rates, and status.
        """
        start_time = perf_counter()
        x = np.zeros((self.data.S, self.data.C, self.data.K), dtype=np.float64)
        h = np.zeros((self.data.C, self.data.K), dtype=np.float64)
        xi = np.zeros((self.data.C, self.data.K), dtype=np.float64)

        for cell in range(self.data.C):
            selected = self._solve_cell(cell)
            if selected is None:
                solve_time = perf_counter() - start_time
                return self._infeasible_result(
                    status=f"infeasible: no budget-feasible path for cell {cell}",
                    solve_time=solve_time,
                )
            x[selected, cell, np.arange(self.data.K)] = 1.0
            xi[cell, :] = self.capacity_proxy[selected, cell, np.arange(self.data.K)]
            h[cell, 1:] = (selected[1:] != selected[:-1]).astype(np.float64)

        solve_time = perf_counter() - start_time
        return P2Result(
            x=x,
            h=h,
            xi=xi,
            U=_p2_objective(xi, h, self.data.eps, self.data.lambda_h),
            handover_per_cell=h.sum(axis=1),
            solve_time=solve_time,
            mip_gap=0.0,
            status="dp_optimal",
        )

    def _solve_cell(self, cell: int) -> IntArray | None:
        budget = int(np.floor(self.data.H[cell] + 1e-9))
        if budget < 0:
            return None

        dp = np.full((self.data.K, self.data.S, budget + 1), -np.inf, dtype=np.float64)
        parent_sat = np.full((self.data.K, self.data.S, budget + 1), -1, dtype=np.int64)
        parent_used = np.full((self.data.K, self.data.S, budget + 1), -1, dtype=np.int64)

        visible_initial = self.data.v[:, cell, 0] > 0.5
        for sat in np.where(visible_initial)[0]:
            dp[0, sat, 0] = self.capacity_proxy[sat, cell, 0]

        for slot in range(1, self.data.K):
            visible = self.data.v[:, cell, slot] > 0.5
            for sat in np.where(visible)[0]:
                reward = self.capacity_proxy[sat, cell, slot]
                for prev_sat in range(self.data.S):
                    switch = 0 if sat == prev_sat else 1
                    for used_before in range(budget + 1 - switch):
                        prev_value = dp[slot - 1, prev_sat, used_before]
                        if not np.isfinite(prev_value):
                            continue
                        used_after = used_before + switch
                        value = prev_value + reward - self.data.lambda_h * switch
                        if value > dp[slot, sat, used_after]:
                            dp[slot, sat, used_after] = value
                            parent_sat[slot, sat, used_after] = prev_sat
                            parent_used[slot, sat, used_after] = used_before

        flat_best = int(np.argmax(dp[-1]))
        if not np.isfinite(dp[-1].reshape(-1)[flat_best]):
            return None
        best_sat, best_used = np.unravel_index(flat_best, dp[-1].shape)
        return _backtrack(parent_sat, parent_used, int(best_sat), int(best_used))

    def _infeasible_result(self, status: str, solve_time: float) -> P2Result:
        return P2Result(
            x=np.zeros((self.data.S, self.data.C, self.data.K), dtype=np.float64),
            h=np.zeros((self.data.C, self.data.K), dtype=np.float64),
            xi=np.zeros((self.data.C, self.data.K), dtype=np.float64),
            U=float("-inf"),
            handover_per_cell=np.zeros(self.data.C, dtype=np.float64),
            solve_time=solve_time,
            mip_gap=0.0,
            status=status,
        )


def _backtrack(
    parent_sat: IntArray,
    parent_used: IntArray,
    best_sat: int,
    best_used: int,
) -> IntArray:
    selected = np.zeros(parent_sat.shape[0], dtype=np.int64)
    sat = best_sat
    used = best_used
    for slot in range(parent_sat.shape[0] - 1, -1, -1):
        selected[slot] = sat
        if slot == 0:
            break
        prev_sat = parent_sat[slot, sat, used]
        prev_used = parent_used[slot, sat, used]
        sat = int(prev_sat)
        used = int(prev_used)
    return selected
