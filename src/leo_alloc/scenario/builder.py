"""Build ScenarioInstance objects from config and system parameters.

Geometry is synthetic and deterministic, but visibility is still derived from
elevation angles.  The builder never marks a below-threshold link visible by
default; it resamples geometry and raises if the requested handover budget is
not achievable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from leo_alloc.scenario.channel import generate_channel_gains
from leo_alloc.scenario.demand import generate_demand
from leo_alloc.scenario.instance import ScenarioInstance
from leo_alloc.scenario.orbit import generate_cell_positions, generate_synthetic_geometry
from leo_alloc.scenario.visibility import check_feasibility, compute_visibility, ensure_coverage
from leo_alloc.utils.config import ScenarioConfig, SystemParams
from leo_alloc.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


def _build_geometry(
    cfg: ScenarioConfig,
    sys_params: SystemParams,
    rng: np.random.Generator,
    elevation_threshold_deg: float,
    shadow_std_db: float,
    atmospheric_loss_db: float,
    pointing_loss_db: float,
    sim_duration_s: float,
    repair_coverage: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return (g, v, distances_m) for one geometry realisation."""
    cell_lats, cell_lons = generate_cell_positions(cfg.C, rng)
    distances_m, elevations_deg = generate_synthetic_geometry(
        cfg.S,
        cfg.C,
        cfg.K,
        cell_lats,
        cell_lons,
        rng,
        sim_duration_s=sim_duration_s,
        handover_budget=cfg.H_per_cell,
    )
    v_raw = compute_visibility(elevations_deg, threshold_deg=elevation_threshold_deg)
    v = ensure_coverage(v_raw, elevations_deg) if repair_coverage else v_raw
    g = generate_channel_gains(
        distances_m,
        sys_params.freq_Hz,
        sys_params.G_tx_dBi,
        sys_params.G_rx_dBi,
        rng,
        atmospheric_loss_db=atmospheric_loss_db,
        pointing_loss_db=pointing_loss_db,
        shadow_std_db=shadow_std_db,
    )
    return g, v, distances_m


def _assemble(
    cfg: ScenarioConfig,
    sys_params: SystemParams,
    g: FloatArray,
    v: FloatArray,
    a: FloatArray,
    scenario_id: str,
) -> ScenarioInstance:
    """Pack validated arrays into a frozen ScenarioInstance."""
    n_prb = int(sys_params.bandwidth_Hz / sys_params.W_PRB_Hz)
    return ScenarioInstance(
        S=cfg.S,
        C=cfg.C,
        K=cfg.K,
        M=cfg.M,
        g=g,
        v=v,
        a=a,
        N_PRB=np.full(cfg.S, n_prb, dtype=np.float64),
        P_max=np.full(cfg.S, sys_params.P_sat_max_W, dtype=np.float64),
        H=np.full(cfg.C, cfg.H_per_cell, dtype=np.float64),
        W_PRB=sys_params.W_PRB_Hz,
        N0=sys_params.N0_W_per_Hz,
        T_f=sys_params.T_fast_slot_s,
        eps=sys_params.eps,
        lambda_h=sys_params.lambda_h,
        seed=cfg.seed,
        scenario_id=scenario_id,
    )


def build_scenario(
    cfg: ScenarioConfig,
    sys_params: SystemParams,
    scenario_id: str = "",
    elevation_threshold_deg: float = 20.0,
    shadow_std_db: float = 2.0,
    atmospheric_loss_db: float = 0.5,
    pointing_loss_db: float = 3.0,
    sim_duration_s: float = 3600.0,
    max_attempts: int = 16,
    repair_coverage: bool = False,
) -> ScenarioInstance:
    """Build a fully validated ScenarioInstance from config.

    Parameters
    ----------
    cfg : ScenarioConfig
        Discrete scenario dimensions and demand settings.
    sys_params : SystemParams
        Physical and optimisation constants.
    scenario_id : str
        Human-readable identifier.  Auto-generated from cfg if empty.
    elevation_threshold_deg : float
        Minimum elevation (degrees) for a satellite to be visible.
    shadow_std_db : float
        Log-normal shadow fading standard deviation in dB.
    atmospheric_loss_db, pointing_loss_db : float
        Atmospheric and pointing losses in dB.
    sim_duration_s : float
        Simulated orbital time window in seconds (governs satellite movement).

    Returns
    -------
    ScenarioInstance
        Frozen, fully feasible scenario with all arrays set.

    Raises
    ------
    RuntimeError
        If the handover budget H_per_cell is too tight for the generated
        visibility pattern (after ensure_coverage).  Try a different seed.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    handover_budget = np.full(cfg.C, cfg.H_per_cell, dtype=np.float64)
    best_min_handovers: FloatArray | None = None
    best_attempt = -1
    for attempt in range(max_attempts):
        geometry_rng = np.random.default_rng(cfg.seed + 1009 * attempt)
        g, v, _ = _build_geometry(
            cfg,
            sys_params,
            geometry_rng,
            elevation_threshold_deg,
            shadow_std_db,
            atmospheric_loss_db,
            pointing_loss_db,
            sim_duration_s,
            repair_coverage,
        )
        feasible, min_handovers = check_feasibility(v, handover_budget)
        best_min_handovers = min_handovers
        best_attempt = attempt
        if feasible:
            break
    else:
        assert best_min_handovers is not None
        worst = int(np.argmax(best_min_handovers - handover_budget))
        required = best_min_handovers[worst]
        required_text = "coverage gap" if np.isinf(required) else f"{int(required)} handovers"
        raise RuntimeError(
            f"Scenario seed={cfg.seed} infeasible after {max_attempts} attempt(s): "
            f"cell {worst} needs {required_text} but "
            f"budget={cfg.H_per_cell}. Increase H_per_cell, increase S, or use a "
            f"different seed."
        )

    demand_rng = np.random.default_rng(cfg.seed + 7919)
    a = generate_demand(
        cfg.C,
        cfg.K,
        cfg.M,
        mismatch_ratio=cfg.mismatch_ratio,
        demand_base_high_bps=cfg.demand_base_high_mbps * 1e6,
        demand_base_low_bps=cfg.demand_base_low_mbps * 1e6,
        t_fast_slot_s=sys_params.T_fast_slot_s,
        rng=demand_rng,
    )

    sid = scenario_id or f"S{cfg.S}C{cfg.C}K{cfg.K}M{cfg.M}_seed{cfg.seed}"
    logger.info(
        "Built scenario '%s': feasible, min_ho_max=%d, attempt=%d",
        sid,
        int(np.max(best_min_handovers)) if best_min_handovers is not None else -1,
        best_attempt,
    )
    return _assemble(cfg, sys_params, g, v, a, sid)
