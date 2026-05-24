"""Tests for scenario/builder.py — end-to-end ScenarioInstance construction."""

from __future__ import annotations

import numpy as np
import pytest

from leo_alloc.scenario.builder import build_scenario
from leo_alloc.scenario.instance import ScenarioInstance
from leo_alloc.scenario.visibility import check_feasibility
from leo_alloc.utils.config import ScenarioConfig, SystemParams


@pytest.fixture
def toy_cfg() -> ScenarioConfig:
    return ScenarioConfig(S=2, C=3, K=5, M=4, H_per_cell=5, seed=0)


@pytest.fixture
def sys_params() -> SystemParams:
    return SystemParams()


def test_build_returns_scenario_instance(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    scenario = build_scenario(toy_cfg, sys_params)
    assert isinstance(scenario, ScenarioInstance)


def test_build_dimensions(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert s.S == toy_cfg.S
    assert s.C == toy_cfg.C
    assert s.K == toy_cfg.K
    assert s.M == toy_cfg.M


def test_build_array_shapes(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert s.g.shape == (toy_cfg.S, toy_cfg.C, toy_cfg.K)
    assert s.v.shape == (toy_cfg.S, toy_cfg.C, toy_cfg.K)
    assert s.a.shape == (toy_cfg.C, toy_cfg.K, toy_cfg.M)
    assert s.N_PRB.shape == (toy_cfg.S,)
    assert s.P_max.shape == (toy_cfg.S,)
    assert s.H.shape == (toy_cfg.C,)


def test_build_visibility_feasible(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    feasible, _ = check_feasibility(s.v, s.H)
    assert feasible


def test_build_every_cell_slot_has_coverage(
    toy_cfg: ScenarioConfig,
    sys_params: SystemParams,
) -> None:
    s = build_scenario(toy_cfg, sys_params)
    # At least one satellite visible per (cell, slot)
    assert np.all(s.v.sum(axis=0) >= 1)


def test_build_channel_gains_positive(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert np.all(s.g > 0.0)


def test_build_demand_nonnegative(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert np.all(s.a >= 0.0)


def test_build_nprb_derived_from_bandwidth(
    toy_cfg: ScenarioConfig,
    sys_params: SystemParams,
) -> None:
    s = build_scenario(toy_cfg, sys_params)
    expected = int(sys_params.bandwidth_Hz / sys_params.W_PRB_Hz)
    np.testing.assert_array_equal(s.N_PRB, np.full(toy_cfg.S, expected, dtype=np.float64))


def test_build_pmax_from_system_params(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    np.testing.assert_array_equal(s.P_max, np.full(toy_cfg.S, sys_params.P_sat_max_W))


def test_build_handover_budget_from_config(
    toy_cfg: ScenarioConfig,
    sys_params: SystemParams,
) -> None:
    s = build_scenario(toy_cfg, sys_params)
    np.testing.assert_array_equal(s.H, np.full(toy_cfg.C, toy_cfg.H_per_cell, dtype=np.float64))


def test_build_reproducible(sys_params: SystemParams) -> None:
    cfg = ScenarioConfig(S=2, C=3, K=5, M=4, H_per_cell=5, seed=42)
    s1 = build_scenario(cfg, sys_params)
    s2 = build_scenario(cfg, sys_params)
    np.testing.assert_array_equal(s1.g, s2.g)
    np.testing.assert_array_equal(s1.v, s2.v)
    np.testing.assert_array_equal(s1.a, s2.a)


def test_build_different_seeds_differ(sys_params: SystemParams) -> None:
    s1 = build_scenario(ScenarioConfig(S=2, C=3, K=5, M=4, H_per_cell=5, seed=0), sys_params)
    s2 = build_scenario(ScenarioConfig(S=2, C=3, K=5, M=4, H_per_cell=5, seed=1), sys_params)
    assert not np.array_equal(s1.g, s2.g)


def test_build_scenario_id_set(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert len(s.scenario_id) > 0


def test_build_custom_scenario_id(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params, scenario_id="my_test")
    assert s.scenario_id == "my_test"


def test_build_seed_stored(toy_cfg: ScenarioConfig, sys_params: SystemParams) -> None:
    s = build_scenario(toy_cfg, sys_params)
    assert s.seed == toy_cfg.seed


def test_build_raises_when_raw_visibility_is_not_feasible(sys_params: SystemParams) -> None:
    cfg = ScenarioConfig(S=2, C=3, K=5, M=4, H_per_cell=5, seed=0)

    with pytest.raises(RuntimeError, match="infeasible"):
        build_scenario(cfg, sys_params, elevation_threshold_deg=89.9, max_attempts=1)


@pytest.mark.parametrize("s_count,c_count,k_count", [(2, 3, 5), (3, 5, 8)])
def test_build_various_scales(
    s_count: int, c_count: int, k_count: int, sys_params: SystemParams
) -> None:
    cfg = ScenarioConfig(S=s_count, C=c_count, K=k_count, M=4, H_per_cell=k_count, seed=7)
    scenario = build_scenario(cfg, sys_params)
    assert scenario.g.shape == (s_count, c_count, k_count)
    assert np.all(scenario.v.sum(axis=0) >= 1)
