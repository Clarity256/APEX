"""Scenario generation and physical-layer inputs."""

from leo_alloc.scenario.builder import build_scenario
from leo_alloc.scenario.channel import generate_channel_gains
from leo_alloc.scenario.demand import generate_demand
from leo_alloc.scenario.instance import ScenarioInstance
from leo_alloc.scenario.orbit import (
    generate_cell_positions,
    generate_synthetic_geometry,
    orbital_period_s,
)
from leo_alloc.scenario.visibility import check_feasibility, compute_visibility, ensure_coverage

__all__ = [
    "ScenarioInstance",
    "build_scenario",
    "generate_channel_gains",
    "generate_demand",
    "generate_cell_positions",
    "generate_synthetic_geometry",
    "orbital_period_s",
    "compute_visibility",
    "ensure_coverage",
    "check_feasibility",
]
