"""Visibility matrix computation and per-cell feasibility checks.

Visibility is 1 when a satellite's elevation angle exceeds the threshold,
0 otherwise.  ensure_coverage() guarantees every (cell, slot) pair has at
least one visible satellite, which is required for P2 feasibility.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from leo_alloc.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


def compute_visibility(
    elevations_deg: FloatArray,
    threshold_deg: float = 20.0,
) -> FloatArray:
    """Threshold elevation angles into a binary visibility matrix.

    Parameters
    ----------
    elevations_deg : ndarray of shape (S, C, K)
        Elevation angle from each cell to each satellite at each slow slot.
    threshold_deg : float
        Minimum elevation (inclusive) for a satellite to be considered visible.

    Returns
    -------
    ndarray of shape (S, C, K), dtype float64, values in {0.0, 1.0}
    """
    return np.where(elevations_deg >= threshold_deg, 1.0, 0.0).astype(np.float64)


def ensure_coverage(
    v: FloatArray,
    elevations_deg: FloatArray,
) -> FloatArray:
    """Guarantee at least one visible satellite per (cell, slot).

    For any (cell, slot) with no visible satellite, the satellite with the
    highest elevation (best available link) is forced visible.  This keeps
    channel gains physically consistent while ensuring P2 feasibility.

    Parameters
    ----------
    v : ndarray of shape (S, C, K)
        Current binary visibility matrix.
    elevations_deg : ndarray of shape (S, C, K)
        Elevation angles used to select the best satellite when filling gaps.

    Returns
    -------
    ndarray of shape (S, C, K) with every (cell, slot) covered.
    """
    v_out = v.copy()
    covered = v_out.sum(axis=0) >= 1.0  # [C, K]
    if covered.all():
        return v_out
    gap_cells, gap_slots = np.where(~covered)
    best_sats = np.argmax(elevations_deg[:, gap_cells, gap_slots], axis=0)
    v_out[best_sats, gap_cells, gap_slots] = 1.0
    n_gaps = int(gap_cells.size)
    logger.debug("ensure_coverage: filled %d (cell, slot) gap(s)", n_gaps)
    return v_out


def _min_handovers_dp(v_cell: FloatArray, satellite_count: int, slow_slot_count: int) -> float:
    """DP minimum handovers for one cell, or inf if infeasible.

    Returns inf when no visible satellite exists at any slow slot, making the
    scenario infeasible regardless of the handover budget.

    Parameters
    ----------
    v_cell : ndarray of shape (S, K)
        Binary visibility for a single cell across all slow slots.

    Returns
    -------
    float
        Minimum handovers required, or math.inf if a slot has no coverage.
    """
    # dp[s] = min handovers to reach current slot on satellite s (float for inf)
    dp = np.where(v_cell[:, 0] > 0.5, 0.0, np.inf)
    handover_cost = 1.0 - np.eye(satellite_count, dtype=np.float64)

    for slot in range(1, slow_slot_count):
        cost = dp[:, None] + handover_cost  # [S_from, S_to], may contain inf
        best_incoming = np.min(cost, axis=0)  # [S_to]
        dp = np.where(v_cell[:, slot] > 0.5, best_incoming, np.inf)

    return float(np.min(dp))


def check_feasibility(
    v: FloatArray,
    handover_budget: FloatArray,
) -> tuple[bool, FloatArray]:
    """Check whether the handover budget H[c] is achievable for each cell.

    Uses dynamic programming over slow slots to compute the minimum number of
    handovers each cell requires.  A cell is infeasible if no satellite is
    visible at some slow slot (regardless of budget).

    Parameters
    ----------
    v : ndarray of shape (S, C, K)
        Binary visibility matrix.
    handover_budget : ndarray of shape (C,)
        Handover budget per cell.

    Returns
    -------
    all_feasible : bool
        True iff every cell satisfies min_handovers[c] <= H[c].
    min_handovers : ndarray of shape (C,)
        Minimum number of handovers needed per cell.
    """
    satellite_count, cell_count, slow_slot_count = v.shape
    min_handovers = np.zeros(cell_count, dtype=np.float64)
    for cell in range(cell_count):
        min_handovers[cell] = _min_handovers_dp(
            v[:, cell, :],
            satellite_count,
            slow_slot_count,
        )

    # inf means a coverage gap — infeasible regardless of budget
    all_feasible = bool(
        np.all(np.isfinite(min_handovers)) and np.all(min_handovers <= handover_budget)
    )
    if not all_feasible:
        violated = np.where(min_handovers > handover_budget)[0]
        logger.debug(
            "Feasibility check failed for %d cell(s): %s",
            len(violated),
            violated.tolist(),
        )
    return all_feasible, min_handovers
