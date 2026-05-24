"""Benchmark P1 CVX and dual solvers on an overloaded medium scenario."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from leo_alloc.solvers import P1CVXSolver, P1DualSolver
from leo_alloc.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "p1_benchmarks"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

FloatArray = NDArray[np.float64]
logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed overloaded medium benchmark dimensions and physical constants."""

    S: int = 4
    C: int = 10
    M: int = 20
    N_PRB: float = 100.0
    P_max: float = 100.0
    W_PRB: float = 180e3
    N0: float = 1e-15
    T_f: float = 0.01
    eps: float = 1e-4
    demand_low_bits: float = 1.0e3
    demand_high_bits: float = 2.0e4
    channel_low: float = 8e-14
    channel_high: float = 1.6e-12


@dataclass(frozen=True)
class BenchmarkRow:
    """Per-instance L1-vs-L2 benchmark metrics."""

    seed: int
    cvx_status: str
    dual_status: str
    cvx_objective: float
    dual_objective: float
    relative_gap: float
    cvx_time_s: float
    dual_time_s: float
    speedup: float
    cvx_xi_min: float
    dual_xi_min: float
    cvx_xi_mean: float
    dual_xi_mean: float


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Files produced by one benchmark run."""

    rows_csv: str
    summary_json: str
    summary_md: str
    plot_png: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--use-dpp", action="store_true", help="Enable CVXPY DPP caching.")
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG plot.")
    return parser.parse_args()


def sys_params(config: BenchmarkConfig) -> dict[str, FloatArray | float]:
    """Return the P1 solver parameter dictionary for the benchmark."""
    return {
        "N_PRB": np.full(config.S, config.N_PRB, dtype=np.float64),
        "P_max": np.full(config.S, config.P_max, dtype=np.float64),
        "W_PRB": config.W_PRB,
        "N0": config.N0,
        "T_f": config.T_f,
        "eps": config.eps,
    }


def generate_overloaded_instance(
    config: BenchmarkConfig,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate one medium overloaded P1 instance with demand-channel mismatch."""
    rng = np.random.default_rng(seed)
    x = np.zeros((config.S, config.C), dtype=np.float64)
    associated_satellites = rng.integers(0, config.S, size=config.C)
    x[associated_satellites, np.arange(config.C)] = 1.0

    pressure = np.linspace(1.0, 0.0, config.C)
    rng.shuffle(pressure)
    base_demand = config.demand_low_bits + pressure * (
        config.demand_high_bits - config.demand_low_bits
    )
    bursts = rng.lognormal(mean=0.0, sigma=0.35, size=(config.C, config.M))
    a = base_demand[:, None] * bursts

    g = rng.lognormal(mean=np.log(6e-13), sigma=0.45, size=(config.S, config.C))
    assigned_gain = config.channel_low + (1.0 - pressure) * (
        config.channel_high - config.channel_low
    )
    assigned_gain *= rng.lognormal(mean=0.0, sigma=0.18, size=config.C)
    g[associated_satellites, np.arange(config.C)] = assigned_gain
    return x, a.astype(np.float64), g.astype(np.float64)


def relative_gap(cvx_objective: float, dual_objective: float) -> float:
    """Compute the relative objective gap against the CVX objective."""
    return float((cvx_objective - dual_objective) / max(abs(cvx_objective), 1e-12))


def benchmark_one(
    config: BenchmarkConfig,
    seed: int,
    solver: str,
    use_dpp: bool,
) -> BenchmarkRow:
    """Run one L1-vs-L2 benchmark instance."""
    x, a, g = generate_overloaded_instance(config, seed)
    params = sys_params(config)
    cvx_result = P1CVXSolver(
        config.S,
        config.C,
        config.M,
        params,
        solver=solver,
        use_dpp=use_dpp,
    ).solve(x, a, g)
    dual_result = P1DualSolver(config.S, config.C, config.M, params).solve(x, a, g)
    return BenchmarkRow(
        seed=seed,
        cvx_status=cvx_result.status,
        dual_status=dual_result.status,
        cvx_objective=cvx_result.U,
        dual_objective=dual_result.U,
        relative_gap=relative_gap(cvx_result.U, dual_result.U),
        cvx_time_s=cvx_result.solve_time,
        dual_time_s=dual_result.solve_time,
        speedup=cvx_result.solve_time / max(dual_result.solve_time, 1e-12),
        cvx_xi_min=float(np.min(cvx_result.xi)),
        dual_xi_min=float(np.min(dual_result.xi)),
        cvx_xi_mean=float(np.mean(cvx_result.xi)),
        dual_xi_mean=float(np.mean(dual_result.xi)),
    )


def run_benchmark(
    instances: int,
    seed: int,
    solver: str,
    use_dpp: bool,
) -> list[BenchmarkRow]:
    """Run a deterministic batch of overloaded medium benchmark instances."""
    if instances <= 0:
        raise ValueError("instances must be positive")
    config = BenchmarkConfig()
    rows: list[BenchmarkRow] = []
    for offset in range(instances):
        instance_seed = seed + offset
        row = benchmark_one(config, instance_seed, solver, use_dpp)
        rows.append(row)
        logger.info(
            "seed=%s gap=%.4f speedup=%.1fx cvx=%.3fs dual=%.4fs",
            instance_seed,
            row.relative_gap,
            row.speedup,
            row.cvx_time_s,
            row.dual_time_s,
        )
    return rows


def summarize(
    rows: list[BenchmarkRow],
    config: BenchmarkConfig,
    solver: str,
    use_dpp: bool,
) -> dict[str, object]:
    """Return aggregate benchmark metrics."""
    gaps = np.array([row.relative_gap for row in rows], dtype=np.float64)
    speedups = np.array([row.speedup for row in rows], dtype=np.float64)
    cvx_times = np.array([row.cvx_time_s for row in rows], dtype=np.float64)
    dual_times = np.array([row.dual_time_s for row in rows], dtype=np.float64)
    return {
        "scenario": "medium_overloaded",
        "solver": solver,
        "use_dpp": use_dpp,
        "config": asdict(config),
        "instances": len(rows),
        "gap_median": float(np.median(gaps)),
        "gap_p95": float(np.quantile(gaps, 0.95)),
        "gap_max": float(np.max(gaps)),
        "speedup_median": float(np.median(speedups)),
        "speedup_p05": float(np.quantile(speedups, 0.05)),
        "cvx_time_median_s": float(np.median(cvx_times)),
        "dual_time_median_s": float(np.median(dual_times)),
        "cvx_xi_mean": float(np.mean([row.cvx_xi_mean for row in rows])),
        "dual_xi_mean": float(np.mean([row.dual_xi_mean for row in rows])),
        "cvx_xi_min_mean": float(np.mean([row.cvx_xi_min for row in rows])),
        "dual_xi_min_mean": float(np.mean([row.dual_xi_min for row in rows])),
    }


def write_rows_csv(path: Path, rows: Iterable[BenchmarkRow]) -> None:
    """Write per-instance benchmark rows to CSV."""
    rows_list = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows_list[0]).keys()))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    """Write a compact Markdown summary table."""
    lines = [
        "# P1 Medium Overloaded Benchmark",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "instances",
        "gap_median",
        "gap_p95",
        "gap_max",
        "speedup_median",
        "speedup_p05",
        "cvx_time_median_s",
        "dual_time_median_s",
        "cvx_xi_mean",
        "dual_xi_mean",
        "cvx_xi_min_mean",
        "dual_xi_min_mean",
    ]:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.6g} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[BenchmarkRow], summary: dict[str, object]) -> None:
    """Write a PNG with gap, speedup, and satisfaction comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = np.array([row.seed for row in rows], dtype=np.int64)
    gaps = np.array([row.relative_gap for row in rows], dtype=np.float64)
    speedups = np.array([row.speedup for row in rows], dtype=np.float64)
    cvx_xi = np.array([row.cvx_xi_mean for row in rows], dtype=np.float64)
    dual_xi = np.array([row.dual_xi_mean for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(seeds, gaps * 100.0, marker="o", linewidth=1.0)
    axes[0].axhline(5.0, color="tab:red", linestyle="--", linewidth=1.0, label="5% target")
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("relative gap (%)")
    axes[0].set_title(f"p95={float(summary['gap_p95']) * 100.0:.2f}%")
    axes[0].legend()
    axes[1].bar(seeds, speedups)
    axes[1].axhline(10.0, color="tab:red", linestyle="--", linewidth=1.0, label="10x target")
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("speedup")
    axes[1].set_title(f"median={float(summary['speedup_median']):.1f}x")
    axes[1].legend()
    axes[2].plot(seeds, cvx_xi, marker="o", label="CVX")
    axes[2].plot(seeds, dual_xi, marker="s", label="Dual")
    axes[2].set_xlabel("seed")
    axes[2].set_ylabel("mean satisfaction")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("L1 vs L2 xi")
    axes[2].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_artifacts(
    rows: list[BenchmarkRow],
    summary: dict[str, object],
    out_dir: Path,
    no_plot: bool,
) -> BenchmarkArtifacts:
    """Persist benchmark CSV, JSON, Markdown, and optional PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"p1_benchmark_medium_overloaded_n{len(rows)}_{stamp}"
    rows_csv = out_dir / f"{stem}.csv"
    summary_json = out_dir / f"{stem}.json"
    summary_md = out_dir / f"{stem}.md"
    plot_png = None if no_plot else out_dir / f"{stem}.png"
    write_rows_csv(rows_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_md(summary_md, summary)
    if plot_png is not None:
        write_plot(plot_png, rows, summary)
    return BenchmarkArtifacts(
        rows_csv=str(rows_csv),
        summary_json=str(summary_json),
        summary_md=str(summary_md),
        plot_png=str(plot_png) if plot_png is not None else None,
    )


def main() -> None:
    """Run the benchmark from the command line."""
    args = parse_args()
    config = BenchmarkConfig()
    rows = run_benchmark(args.instances, args.seed, args.solver, args.use_dpp)
    summary = summarize(rows, config, args.solver, args.use_dpp)
    artifacts = write_artifacts(rows, summary, args.out_dir, args.no_plot)
    logger.info("P1 benchmark artifacts: %s", json.dumps(asdict(artifacts), sort_keys=True))


if __name__ == "__main__":
    main()
