"""Run a reproducible synthetic P2 MILP association experiment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers import P2MILPSolver, P2Result, P2RollingSolver
from leo_alloc.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "p2_runs"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

FloatArray = NDArray[np.float64]
logger = get_logger(__name__)


@dataclass(frozen=True)
class SyntheticP2Scenario:
    """ScenarioInstance-compatible synthetic P2 experiment container."""

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
    seed: int
    scenario_id: str


@dataclass(frozen=True)
class P2ExperimentArtifacts:
    """Files produced by one P2 experiment run."""

    arrays_path: str
    summary_path: str
    plot_path: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["toy", "medium"], default="toy")
    parser.add_argument("--method", choices=["milp", "rolling"], default="milp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG plot.")
    return parser.parse_args()


def generate_scenario(scale: str, seed: int) -> SyntheticP2Scenario:
    """Generate a deterministic P2 scenario with visibility and demand pressure."""
    if scale == "toy":
        satellite_count, cell_count, slow_slot_count, fast_slot_count = 3, 5, 8, 4
        handover_budget = 2
    else:
        satellite_count, cell_count, slow_slot_count, fast_slot_count = 4, 10, 12, 4
        handover_budget = 3
    rng = np.random.default_rng(seed)
    v = generate_visibility(satellite_count, cell_count, slow_slot_count, rng)
    a = generate_demand(cell_count, slow_slot_count, fast_slot_count, rng)
    g = generate_channel_gain(v, rng)
    return SyntheticP2Scenario(
        S=satellite_count,
        C=cell_count,
        K=slow_slot_count,
        M=fast_slot_count,
        g=g,
        v=v,
        a=a,
        N_PRB=np.full(satellite_count, 100.0, dtype=np.float64),
        P_max=np.full(satellite_count, 100.0, dtype=np.float64),
        H=np.full(cell_count, float(handover_budget), dtype=np.float64),
        W_PRB=180e3,
        N0=1e-15,
        T_f=0.01,
        eps=1e-4,
        lambda_h=0.1,
        seed=seed,
        scenario_id=f"p2_{scale}_seed{seed}",
    )


def generate_visibility(
    satellite_count: int,
    cell_count: int,
    slow_slot_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate rotating visibility windows with guaranteed feasibility."""
    v = np.zeros((satellite_count, cell_count, slow_slot_count), dtype=np.float64)
    for c in range(cell_count):
        primary = c % satellite_count
        secondary = (primary + 1 + c // satellite_count) % satellite_count
        tertiary = (primary + 2) % satellite_count
        split = slow_slot_count // 2
        v[primary, c, : split + 1] = 1.0
        v[secondary, c, max(split - 1, 0) :] = 1.0
        if c % 3 == 0:
            v[tertiary, c, slow_slot_count // 3 : 2 * slow_slot_count // 3 + 1] = 1.0
        random_extra = rng.random((satellite_count, slow_slot_count)) < 0.08
        v[:, c, :] = np.maximum(v[:, c, :], random_extra.astype(np.float64))
    return v


def generate_demand(
    cell_count: int,
    slow_slot_count: int,
    fast_slot_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate bursty per-fast-slot demand arrivals in bits."""
    cell_pressure = np.linspace(1.0, 0.35, cell_count)
    rng.shuffle(cell_pressure)
    slow_profile = 0.85 + 0.35 * np.sin(np.linspace(0.0, 2.0 * np.pi, slow_slot_count))
    base = 6e3 + cell_pressure[:, None] * 4.5e4 * slow_profile[None, :]
    burst = rng.lognormal(mean=0.0, sigma=0.35, size=(cell_count, slow_slot_count, fast_slot_count))
    return base[:, :, None] * burst


def generate_channel_gain(v: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Generate channel gains where visibility windows create meaningful tradeoffs."""
    satellite_count, cell_count, slow_slot_count = v.shape
    trend = np.linspace(1.2, 0.7, slow_slot_count)
    g = rng.lognormal(
        mean=np.log(6e-13),
        sigma=0.45,
        size=(satellite_count, cell_count, slow_slot_count),
    )
    for c in range(cell_count):
        for s in range(satellite_count):
            g[s, c, :] *= trend if (s + c) % 2 == 0 else trend[::-1]
    return np.where(v > 0.0, g, 0.0).astype(np.float64)


def selected_satellites(x: FloatArray) -> FloatArray:
    """Return selected satellite index per cell and slow slot."""
    return np.argmax(x, axis=0).astype(np.float64)


def write_arrays(path: Path, scenario: SyntheticP2Scenario, result: P2Result) -> None:
    """Write raw P2 scenario and result arrays to an NPZ file."""
    np.savez_compressed(
        path,
        g=scenario.g,
        v=scenario.v,
        a=scenario.a,
        H=scenario.H,
        x=result.x,
        h=result.h,
        xi=result.xi,
        selected_satellite=selected_satellites(result.x),
        handover_per_cell=result.handover_per_cell,
    )


def summary_dict(
    method: str,
    scale: str,
    scenario: SyntheticP2Scenario,
    result: P2Result,
    arrays_path: Path,
    plot_path: Path | None,
    window: int | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable P2 experiment summary."""
    summary = {
        "method": method,
        "scale": scale,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "status": result.status,
        "objective": result.U,
        "solve_time_s": result.solve_time,
        "mip_gap": result.mip_gap,
        "handover_total": float(np.sum(result.handover_per_cell)),
        "handover_max_per_cell": float(np.max(result.handover_per_cell)),
        "budget_max_per_cell": float(np.max(scenario.H)),
        "xi_min": float(np.min(result.xi)),
        "xi_mean": float(np.mean(result.xi)),
        "xi_max": float(np.max(result.xi)),
        "arrays_path": str(arrays_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
    }
    if window is not None:
        summary["window"] = window
    if step is not None:
        summary["step"] = step
    return summary


def write_plot(path: Path, scenario: SyntheticP2Scenario, result: P2Result) -> None:
    """Write P2 association, handover, and satisfaction diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = selected_satellites(result.x)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    image = axes[0].imshow(selected, aspect="auto", interpolation="nearest", cmap="tab10")
    axes[0].set_xlabel("slow slot")
    axes[0].set_ylabel("cell")
    axes[0].set_title("selected satellite")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    cell_index = np.arange(scenario.C)
    axes[1].bar(cell_index, scenario.H, label="budget", alpha=0.45)
    axes[1].bar(cell_index, result.handover_per_cell, label="used", alpha=0.85)
    axes[1].set_xlabel("cell")
    axes[1].set_ylabel("handover count")
    axes[1].set_title("hard handover budgets")
    axes[1].legend()
    axes[2].plot(np.arange(scenario.K), result.xi.mean(axis=0), marker="o", label="mean")
    axes[2].plot(np.arange(scenario.K), result.xi.min(axis=0), marker="s", label="min")
    axes[2].set_xlabel("slow slot")
    axes[2].set_ylabel("proxy satisfaction")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("proxy xi over time")
    axes[2].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> P2ExperimentArtifacts:
    """Run one P2 experiment and persist artifacts."""
    scenario = generate_scenario(args.scale, args.seed)
    if args.method == "rolling":
        result = P2RollingSolver(
            scenario,
            window=args.window,
            step=args.step,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        ).solve()
    else:
        result = P2MILPSolver(
            scenario,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        ).solve()
    return write_artifacts(args, scenario, result)


def write_artifacts(
    args: argparse.Namespace,
    scenario: SyntheticP2Scenario,
    result: P2Result,
) -> P2ExperimentArtifacts:
    """Persist arrays, summary JSON, and optional plot."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    method_prefix = "" if args.method == "milp" else f"{args.method}_"
    stem = f"p2_{method_prefix}{args.scale}_seed{args.seed}_{stamp}"
    arrays_path = args.out_dir / f"{stem}.npz"
    summary_path = args.out_dir / f"{stem}.json"
    plot_path = None if args.no_plot else args.out_dir / f"{stem}.png"
    write_arrays(arrays_path, scenario, result)
    if plot_path is not None:
        write_plot(plot_path, scenario, result)
    summary = summary_dict(
        args.method,
        args.scale,
        scenario,
        result,
        arrays_path,
        plot_path,
        window=args.window if args.method == "rolling" else None,
        step=args.step if args.method == "rolling" else None,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return P2ExperimentArtifacts(
        str(arrays_path),
        str(summary_path),
        str(plot_path) if plot_path else None,
    )


def main() -> None:
    """Run the P2 command-line experiment."""
    artifacts = run_experiment(parse_args())
    logger.info("P2 experiment artifacts: %s", json.dumps(asdict(artifacts), sort_keys=True))


if __name__ == "__main__":
    main()
