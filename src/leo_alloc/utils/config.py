"""Typed configuration objects shared across modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemParams:
    """Physical and optimization constants for one experiment family."""

    freq_Hz: float = 2e9
    bandwidth_Hz: float = 30e6
    W_PRB_Hz: float = 180e3
    N0_W_per_Hz: float = 4e-21
    P_sat_max_W: float = 1000.0
    G_tx_dBi: float = 30.0
    G_rx_dBi: float = 0.0
    T_fast_slot_s: float = 0.01
    eps: float = 1e-4
    lambda_h: float = 0.1


@dataclass(frozen=True)
class ScenarioConfig:
    """Discrete scenario dimensions and demand generation settings."""

    S: int
    C: int
    K: int
    M: int
    H_per_cell: int
    demand_base_high_mbps: float = 50.0
    demand_base_low_mbps: float = 10.0
    mismatch_ratio: float = 0.3
    seed: int = 0
