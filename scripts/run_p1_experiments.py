"""Run a reproducible synthetic P1 convex-kernel experiment."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from leo_alloc.solvers import P1CVXSolver, P1DualSolver, P1Result
from leo_alloc.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
LOCAL_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_SYSTEM_CONFIG = CONFIG_DIR / "system_params.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "p1_runs"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

FloatArray = NDArray[np.float64]
logger = get_logger(__name__)


@dataclass(frozen=True)
class P1ExperimentArtifacts:
    """Files produced by one P1 experiment run."""

    arrays_path: str
    summary_path: str
    plot_path: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["toy", "medium", "stress"], default="toy")
    parser.add_argument("--scenario-config", type=Path, default=None)
    parser.add_argument("--system-config", type=Path, default=DEFAULT_SYSTEM_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", choices=["cvx", "dual"], default="cvx")
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the summary PNG.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cast(dict[str, Any], data)


def scenario_config_path(scale: str, override: Path | None) -> Path:
    """Return the selected scenario configuration path."""
    if override is not None:
        return override
    return CONFIG_DIR / f"scenario_{scale}.yaml"


def solver_params(
    system_config: Mapping[str, Any],
    satellite_count: int,
) -> dict[str, FloatArray | float]:
    """Convert system YAML values into the P1CVXSolver parameter contract."""
    bandwidth_hz = float(system_config["bandwidth_Hz"])
    w_prb_hz = float(system_config["W_PRB_Hz"])
    n_prb = np.full(satellite_count, np.floor(bandwidth_hz / w_prb_hz), dtype=np.float64)
    p_max = np.full(satellite_count, float(system_config["P_sat_max_W"]), dtype=np.float64)
    return {
        "N_PRB": n_prb,
        "P_max": p_max,
        "W_PRB": w_prb_hz,
        "N0": float(system_config["N0_W_per_Hz"]),
        "T_f": float(system_config["T_fast_slot_s"]),
        "eps": float(system_config["eps"]),
    }


def generate_p1_inputs(
    scenario_config: Mapping[str, Any],
    system_config: Mapping[str, Any],
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate deterministic association, demand arrival, and channel gain arrays."""
    satellite_count = int(scenario_config["S"])
    cell_count = int(scenario_config["C"])
    fast_slot_count = int(scenario_config["M"])
    rng = np.random.default_rng(seed)
    x = generate_association(satellite_count, cell_count, rng)
    a = generate_demand_arrival(scenario_config, system_config, rng)
    g = generate_channel_gain(satellite_count, cell_count, rng)
    if x.shape != (satellite_count, cell_count) or a.shape != (cell_count, fast_slot_count):
        raise RuntimeError("Generated P1 inputs do not match scenario dimensions")
    return x, a, g


def generate_association(
    satellite_count: int,
    cell_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate a valid fixed satellite-cell association matrix."""
    x = np.zeros((satellite_count, cell_count), dtype=np.float64)
    selected_satellites = rng.integers(0, satellite_count, size=cell_count)
    x[selected_satellites, np.arange(cell_count)] = 1.0
    return x


def generate_demand_arrival(
    scenario_config: Mapping[str, Any],
    system_config: Mapping[str, Any],
    rng: np.random.Generator,
) -> FloatArray:
    """Generate per-fast-slot demand arrivals in bits."""
    cell_count = int(scenario_config["C"])
    fast_slot_count = int(scenario_config["M"])
    high_mbps = float(scenario_config["demand_base_high_mbps"])
    low_mbps = float(scenario_config["demand_base_low_mbps"])
    fast_slot_s = float(system_config["T_fast_slot_s"])
    high_cells = rng.random(cell_count) < 0.5
    base_mbps = np.where(high_cells, high_mbps, low_mbps)
    fluctuation = rng.lognormal(mean=0.0, sigma=0.25, size=(cell_count, fast_slot_count))
    return base_mbps[:, None] * 1e6 * fast_slot_s * fluctuation


def generate_channel_gain(
    satellite_count: int,
    cell_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate large-scale channel gains in linear scale."""
    shadowing = rng.lognormal(mean=0.0, sigma=0.5, size=(satellite_count, cell_count))
    distance_factor = rng.uniform(0.5, 1.5, size=(satellite_count, cell_count))
    return 1e-12 * shadowing / distance_factor**2


def write_arrays(
    path: Path,
    x: FloatArray,
    a: FloatArray,
    g: FloatArray,
    result: P1Result,
) -> None:
    """Write raw P1 inputs and outputs to an NPZ file."""
    np.savez_compressed(
        path,
        x=x,
        a=a,
        g=g,
        n=result.n,
        p=result.p,
        z=result.z,
        xi=result.xi,
        zero_demand=result.zero_demand,
    )


def summary_dict(
    method: str,
    scale: str,
    seed: int,
    result: P1Result,
    arrays_path: Path,
    plot_path: Path | None,
) -> dict[str, Any]:
    """Build a JSON-serializable experiment summary."""
    return {
        "method": method,
        "scale": scale,
        "seed": seed,
        "status": result.status,
        "objective": result.U,
        "solve_time_s": result.solve_time,
        "xi_min": float(np.min(result.xi)),
        "xi_mean": float(np.mean(result.xi)),
        "xi_max": float(np.max(result.xi)),
        "zero_demand_count": int(np.count_nonzero(result.zero_demand)),
        "arrays_path": str(arrays_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
    }


def write_plot(path: Path, a: FloatArray, result: P1Result) -> None:
    """Write a compact PNG diagnostic for P1 served demand and satisfaction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_index = np.arange(result.xi.size)
    demand_total_mbit = a.sum(axis=1) / 1e6
    served_total_mbit = result.z.sum(axis=1) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(cell_index, demand_total_mbit, label="demand")
    axes[0].bar(cell_index, served_total_mbit, label="served", alpha=0.75)
    axes[0].set_xlabel("cell")
    axes[0].set_ylabel("Mbit")
    axes[0].legend()
    axes[1].bar(cell_index, result.xi)
    axes[1].set_xlabel("cell")
    axes[1].set_ylabel("satisfaction rate")
    axes[1].set_ylim(0.0, 1.05)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> P1ExperimentArtifacts:
    """Run one synthetic P1 experiment and save artifacts."""
    scenario_path = scenario_config_path(args.scale, args.scenario_config)
    scenario_config = load_yaml(scenario_path)
    system_config = load_yaml(args.system_config)
    seed = int(scenario_config.get("seed", 0) if args.seed is None else args.seed)
    satellite_count = int(scenario_config["S"])
    cell_count = int(scenario_config["C"])
    fast_slot_count = int(scenario_config["M"])
    x, a, g = generate_p1_inputs(scenario_config, system_config, seed)
    params = solver_params(system_config, satellite_count)
    if args.method == "dual":
        result = P1DualSolver(satellite_count, cell_count, fast_slot_count, params).solve(x, a, g)
    else:
        result = P1CVXSolver(
            satellite_count,
            cell_count,
            fast_slot_count,
            params,
            solver=args.solver,
        ).solve(x, a, g)
    return write_artifacts(args, x, a, g, result, seed)


def write_artifacts(
    args: argparse.Namespace,
    x: FloatArray,
    a: FloatArray,
    g: FloatArray,
    result: P1Result,
    seed: int,
) -> P1ExperimentArtifacts:
    """Persist arrays, JSON summary, and optional plot for a completed run."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"p1_{args.method}_{args.scale}_seed{seed}_{stamp}"
    arrays_path = args.out_dir / f"{stem}.npz"
    summary_path = args.out_dir / f"{stem}.json"
    plot_path = None if args.no_plot else args.out_dir / f"{stem}.png"
    write_arrays(arrays_path, x, a, g, result)
    if plot_path is not None:
        write_plot(plot_path, a, result)
    summary = summary_dict(args.method, args.scale, seed, result, arrays_path, plot_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return P1ExperimentArtifacts(
        str(arrays_path),
        str(summary_path),
        str(plot_path) if plot_path else None,
    )


def main() -> None:
    """Run the command-line experiment."""
    artifacts = run_experiment(parse_args())
    logger.info("P1 experiment artifacts: %s", json.dumps(asdict(artifacts), sort_keys=True))


if __name__ == "__main__":
    main()
