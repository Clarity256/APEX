"""Demand arrival process generation for LEO scenario instances.

Two cell types implement the mismatch scenario from doc/05_experiment_design.md:
  Type A (high-load): demand_base_high_bps base rate
  Type B (low-load):  demand_base_low_bps base rate

A sinusoidal intra-day modulation is applied across slow slots, and per-fast-slot
arrivals are drawn from a Poisson distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from leo_alloc.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


def _base_rates_bps(
    cell_count: int,
    mismatch_ratio: float,
    demand_base_high_bps: float,
    demand_base_low_bps: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return per-cell base demand rate in bits/s, shape (C,)."""
    high_count = int(cell_count * mismatch_ratio)
    rates = np.full(cell_count, demand_base_low_bps, dtype=np.float64)
    if high_count > 0:
        high_cells = rng.choice(cell_count, size=high_count, replace=False)
        rates[high_cells] = demand_base_high_bps
    return rates


def _slot_modulation(slot: int, slow_slot_count: int) -> float:
    """Sinusoidal slow-slot demand modulation factor (always positive)."""
    return float(1.0 + 0.3 * np.sin(2.0 * np.pi * slot / max(slow_slot_count, 1)))


def generate_demand(
    cell_count: int,
    slow_slot_count: int,
    fast_slot_count: int,
    mismatch_ratio: float,
    demand_base_high_bps: float,
    demand_base_low_bps: float,
    t_fast_slot_s: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate Poisson demand arrivals with sinusoidal slow-slot modulation.

    Parameters
    ----------
    cell_count, slow_slot_count, fast_slot_count : int
        Number of cells, slow slots, and fast slots.
    mismatch_ratio : float
        Fraction of cells that are high-load (first int(C * mismatch_ratio) cells).
        Must be in [0, 1].
    demand_base_high_bps, demand_base_low_bps : float
        Base demand rates (bits/s) for high- and low-load cells.
    t_fast_slot_s : float
        Fast-slot duration in seconds; scales the Poisson mean to bits/fast-slot.
    rng : Generator
        NumPy random generator (explicit, no global state).

    Returns
    -------
    ndarray of shape (C, K, M), dtype float64
        Demand arrivals in bits.  Each entry a[c, k, m] is a non-negative integer
        drawn from Poisson(lambda_{c,k} * T_f).
    """
    if not (0.0 <= mismatch_ratio <= 1.0):
        raise ValueError(f"mismatch_ratio must be in [0, 1], got {mismatch_ratio}")

    base_rates = _base_rates_bps(
        cell_count,
        mismatch_ratio,
        demand_base_high_bps,
        demand_base_low_bps,
        rng,
    )
    demand = np.zeros((cell_count, slow_slot_count, fast_slot_count), dtype=np.float64)

    for slot in range(slow_slot_count):
        factor = _slot_modulation(slot, slow_slot_count)
        lam = base_rates * factor * t_fast_slot_s
        demand[:, slot, :] = rng.poisson(
            lam[:, None],
            size=(cell_count, fast_slot_count),
        ).astype(np.float64)

    logger.debug(
        "Demand generated: shape=%s mean=%.2e max=%.2e",
        demand.shape,
        float(np.mean(demand)),
        float(np.max(demand)),
    )
    return demand
