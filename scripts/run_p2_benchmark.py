"""Benchmark P2 full MILP and rolling-window solvers on generated scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from leo_alloc.scenario import build_scenario
from leo_alloc.solvers import P2MILPSolver, P2RollingSolver
from leo_alloc.utils.config import ScenarioConfig, SystemParams
from leo_alloc.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "p2_benchmarks"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

logger = get_logger(__name__)


@dataclass(frozen=True)
class P2BenchmarkConfig:
    """P2 benchmark dimensions and rolling parameters."""

    scale: str
    S: int
    C: int
    K: int
    M: int
    H_per_cell: int
    window: int
    step: int
    demand_base_high_mbps: float = 50.0
    demand_base_low_mbps: float = 10.0
    mismatch_ratio: float = 0.3


@dataclass(frozen=True)
class P2BenchmarkRow:
    """Per-seed P2 L1-vs-L2 benchmark metrics."""

    seed: int
    milp_status: str
    rolling_status: str
    milp_utility: float
    rolling_utility: float
    utility_gap_per_cell_slot: float
    milp_time_s: float
    rolling_time_s: float
    speedup: float
    milp_handover_total: float
    rolling_handover_total: float
    milp_handover_max: float
    rolling_handover_max: float
    milp_xi_mean: float
    rolling_xi_mean: float


@dataclass(frozen=True)
class P2BenchmarkArtifacts:
    """Files produced by one P2 benchmark run."""

    rows_csv: str
    summary_json: str
    summary_md: str
    plot_png: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["toy", "medium", "stress"], default="toy")
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG plot.")
    return parser.parse_args()


def benchmark_config(scale: str, window: int | None, step: int | None) -> P2BenchmarkConfig:
    """Return benchmark dimensions for a named scale."""
    if scale == "toy":
        default = P2BenchmarkConfig(
            scale=scale,
            S=2,
            C=5,
            K=10,
            M=10,
            H_per_cell=2,
            window=5,
            step=3,
        )
    elif scale == "medium":
        default = P2BenchmarkConfig(
            scale=scale,
            S=4,
            C=12,
            K=30,
            M=10,
            H_per_cell=6,
            window=8,
            step=4,
        )
    else:
        default = P2BenchmarkConfig(
            scale=scale,
            S=8,
            C=30,
            K=50,
            M=10,
            H_per_cell=12,
            window=10,
            step=5,
        )
    return P2BenchmarkConfig(
        **{
            **asdict(default),
            "window": default.window if window is None else window,
            "step": default.step if step is None else step,
        }
    )


def scenario_config(config: P2BenchmarkConfig, seed: int) -> ScenarioConfig:
    """Convert benchmark config into a scenario config."""
    return ScenarioConfig(
        S=config.S,
        C=config.C,
        K=config.K,
        M=config.M,
        H_per_cell=config.H_per_cell,
        demand_base_high_mbps=config.demand_base_high_mbps,
        demand_base_low_mbps=config.demand_base_low_mbps,
        mismatch_ratio=config.mismatch_ratio,
        seed=seed,
    )


def utility_gap_per_cell_slot(
    milp_utility: float,
    rolling_utility: float,
    cell_count: int,
    slow_slot_count: int,
) -> float:
    """Compute utility loss normalized by the number of cell-slot decisions."""
    return float((milp_utility - rolling_utility) / max(cell_count * slow_slot_count, 1))


def benchmark_one(
    config: P2BenchmarkConfig,
    seed: int,
    time_limit: float,
    mip_gap: float,
) -> P2BenchmarkRow:
    """Run one full-MILP-vs-rolling benchmark instance."""
    scenario = build_scenario(
        scenario_config(config, seed),
        SystemParams(),
        scenario_id=f"p2_{config.scale}_seed{seed}",
    )
    milp = P2MILPSolver(scenario, time_limit=time_limit, mip_gap=mip_gap).solve()
    rolling = P2RollingSolver(
        scenario,
        window=config.window,
        step=config.step,
        time_limit=time_limit,
        mip_gap=mip_gap,
    ).solve()
    return P2BenchmarkRow(
        seed=seed,
        milp_status=milp.status,
        rolling_status=rolling.status,
        milp_utility=milp.U,
        rolling_utility=rolling.U,
        utility_gap_per_cell_slot=utility_gap_per_cell_slot(
            milp.U,
            rolling.U,
            config.C,
            config.K,
        ),
        milp_time_s=milp.solve_time,
        rolling_time_s=rolling.solve_time,
        speedup=milp.solve_time / max(rolling.solve_time, 1e-12),
        milp_handover_total=float(np.sum(milp.handover_per_cell)),
        rolling_handover_total=float(np.sum(rolling.handover_per_cell)),
        milp_handover_max=float(np.max(milp.handover_per_cell)),
        rolling_handover_max=float(np.max(rolling.handover_per_cell)),
        milp_xi_mean=float(np.mean(milp.xi)),
        rolling_xi_mean=float(np.mean(rolling.xi)),
    )


def run_benchmark(
    config: P2BenchmarkConfig,
    instances: int,
    seed: int,
    time_limit: float,
    mip_gap: float,
) -> list[P2BenchmarkRow]:
    """Run a deterministic batch of P2 benchmark instances."""
    if instances <= 0:
        raise ValueError("instances must be positive")
    rows: list[P2BenchmarkRow] = []
    for offset in range(instances):
        row = benchmark_one(config, seed + offset, time_limit, mip_gap)
        rows.append(row)
        logger.info(
            "seed=%s gap=%.4f speedup=%.2fx milp=%.4fs rolling=%.4fs",
            row.seed,
            row.utility_gap_per_cell_slot,
            row.speedup,
            row.milp_time_s,
            row.rolling_time_s,
        )
    return rows


def summarize(rows: list[P2BenchmarkRow], config: P2BenchmarkConfig) -> dict[str, Any]:
    """Return aggregate benchmark metrics."""
    gaps = np.array([row.utility_gap_per_cell_slot for row in rows], dtype=np.float64)
    speedups = np.array([row.speedup for row in rows], dtype=np.float64)
    milp_times = np.array([row.milp_time_s for row in rows], dtype=np.float64)
    rolling_times = np.array([row.rolling_time_s for row in rows], dtype=np.float64)
    return {
        "scenario": f"p2_{config.scale}",
        "config": asdict(config),
        "instances": len(rows),
        "gap_median": float(np.median(gaps)),
        "gap_p95": float(np.quantile(gaps, 0.95)),
        "gap_max": float(np.max(gaps)),
        "speedup_median": float(np.median(speedups)),
        "speedup_p05": float(np.quantile(speedups, 0.05)),
        "milp_time_median_s": float(np.median(milp_times)),
        "rolling_time_median_s": float(np.median(rolling_times)),
        "milp_handover_total_mean": float(np.mean([row.milp_handover_total for row in rows])),
        "rolling_handover_total_mean": float(np.mean([row.rolling_handover_total for row in rows])),
        "milp_xi_mean": float(np.mean([row.milp_xi_mean for row in rows])),
        "rolling_xi_mean": float(np.mean([row.rolling_xi_mean for row in rows])),
    }


def write_csv(path: Path, rows: list[P2BenchmarkRow]) -> None:
    """Write per-instance rows to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact Markdown summary table."""
    metric_keys = [
        "instances",
        "gap_median",
        "gap_p95",
        "gap_max",
        "speedup_median",
        "speedup_p05",
        "milp_time_median_s",
        "rolling_time_median_s",
        "milp_handover_total_mean",
        "rolling_handover_total_mean",
        "milp_xi_mean",
        "rolling_xi_mean",
    ]
    lines = ["# P2 Full MILP vs Rolling Benchmark", "", "| Metric | Value |", "|---|---:|"]
    for key in metric_keys:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.6g} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[P2BenchmarkRow], summary: dict[str, Any]) -> None:
    """Write a PNG diagnostic for P2 L1-vs-L2 benchmark metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [row.seed for row in rows]
    gaps = [row.utility_gap_per_cell_slot for row in rows]
    milp_times = [row.milp_time_s for row in rows]
    rolling_times = [row.rolling_time_s for row in rows]
    milp_handovers = [row.milp_handover_total for row in rows]
    rolling_handovers = [row.rolling_handover_total for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
    axes[0].plot(seeds, gaps, marker="o")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("utility gap per cell-slot")
    axes[0].set_title("rolling vs full MILP")

    width = 0.35
    x_pos = np.arange(len(seeds))
    axes[1].bar(x_pos - width / 2, milp_times, width, label="full MILP")
    axes[1].bar(x_pos + width / 2, rolling_times, width, label="rolling")
    axes[1].set_xticks(x_pos, [str(seed) for seed in seeds])
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("solve time (s)")
    axes[1].set_title(f"median speedup {summary['speedup_median']:.2f}x")
    axes[1].legend()

    axes[2].bar(x_pos - width / 2, milp_handovers, width, label="full MILP")
    axes[2].bar(x_pos + width / 2, rolling_handovers, width, label="rolling")
    axes[2].set_xticks(x_pos, [str(seed) for seed in seeds])
    axes[2].set_xlabel("seed")
    axes[2].set_ylabel("total handovers")
    axes[2].set_title("handover usage")
    axes[2].legend()

    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_artifacts(
    rows: list[P2BenchmarkRow],
    summary: dict[str, Any],
    out_dir: Path,
    scale: str,
    no_plot: bool,
) -> P2BenchmarkArtifacts:
    """Persist benchmark rows and aggregate artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"p2_benchmark_{scale}_n{len(rows)}_{stamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    png_path = None if no_plot else out_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
    if png_path is not None:
        write_plot(png_path, rows, summary)
    return P2BenchmarkArtifacts(
        rows_csv=str(csv_path),
        summary_json=str(json_path),
        summary_md=str(md_path),
        plot_png=str(png_path) if png_path is not None else None,
    )


def main() -> None:
    """Run the command-line benchmark."""
    args = parse_args()
    config = benchmark_config(args.scale, args.window, args.step)
    rows = run_benchmark(config, args.instances, args.seed, args.time_limit, args.mip_gap)
    summary = summarize(rows, config)
    artifacts = write_artifacts(rows, summary, args.out_dir, args.scale, args.no_plot)
    logger.info("P2 benchmark artifacts: %s", json.dumps(asdict(artifacts), sort_keys=True))


if __name__ == "__main__":
    main()
