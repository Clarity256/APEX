"""Calibrate the P2 capacity proxy against the P1 CVX oracle."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from leo_alloc.scenario import build_scenario
from leo_alloc.solvers import P1CVXSolver
from leo_alloc.solvers.p2_milp import _capacity_proxy, _scenario_data
from leo_alloc.utils.config import ScenarioConfig, SystemParams
from leo_alloc.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = PROJECT_ROOT / ".cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "p2_proxy_calibration"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
logger = get_logger(__name__)


@dataclass(frozen=True)
class CalibrationConfig:
    """P2 proxy calibration dimensions and sampling settings."""

    scale: str
    S: int
    C: int
    K: int
    M: int
    H_per_cell: int
    demand_base_high_mbps: float
    demand_base_low_mbps: float
    mismatch_ratio: float
    slots_per_instance: int
    assignments_per_slot: int
    demand_multiplier: float


@dataclass(frozen=True)
class CalibrationRow:
    """One cell-level proxy-vs-oracle comparison."""

    scenario_seed: int
    slot: int
    assignment_id: int
    cell: int
    selected_satellite: int
    visible_satellites: int
    demand_bits: float
    proxy_xi: float
    oracle_xi: float
    signed_error: float
    abs_error: float
    rel_error: float
    oracle_status: str
    oracle_solve_time_s: float


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Files produced by one calibration run."""

    rows_csv: str
    summary_json: str
    summary_md: str
    plot_png: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["toy", "medium", "stress"], default="toy")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots-per-instance", type=int, default=3)
    parser.add_argument("--assignments-per-slot", type=int, default=2)
    parser.add_argument(
        "--demand-multiplier",
        type=float,
        default=1.0,
        help="Scale generated demand before comparing proxy and P1 oracle.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--use-dpp", action="store_true", help="Enable CVXPY DPP caching.")
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG plot.")
    return parser.parse_args()


def calibration_config(
    scale: str,
    slots_per_instance: int,
    assignments_per_slot: int,
    demand_multiplier: float,
) -> CalibrationConfig:
    """Return calibration dimensions for a named scale."""
    if slots_per_instance <= 0:
        raise ValueError("slots_per_instance must be positive")
    if assignments_per_slot <= 0:
        raise ValueError("assignments_per_slot must be positive")
    if not np.isfinite(demand_multiplier) or demand_multiplier <= 0.0:
        raise ValueError("demand_multiplier must be a positive finite scalar")
    if scale == "toy":
        return CalibrationConfig(
            scale=scale,
            S=3,
            C=5,
            K=8,
            M=10,
            H_per_cell=2,
            demand_base_high_mbps=50.0,
            demand_base_low_mbps=10.0,
            mismatch_ratio=0.3,
            slots_per_instance=slots_per_instance,
            assignments_per_slot=assignments_per_slot,
            demand_multiplier=demand_multiplier,
        )
    if scale == "medium":
        return CalibrationConfig(
            scale=scale,
            S=4,
            C=12,
            K=30,
            M=10,
            H_per_cell=6,
            demand_base_high_mbps=50.0,
            demand_base_low_mbps=10.0,
            mismatch_ratio=0.3,
            slots_per_instance=slots_per_instance,
            assignments_per_slot=assignments_per_slot,
            demand_multiplier=demand_multiplier,
        )
    return CalibrationConfig(
        scale=scale,
        S=8,
        C=30,
        K=50,
        M=10,
        H_per_cell=12,
        demand_base_high_mbps=50.0,
        demand_base_low_mbps=10.0,
        mismatch_ratio=0.3,
        slots_per_instance=slots_per_instance,
        assignments_per_slot=assignments_per_slot,
        demand_multiplier=demand_multiplier,
    )


def scenario_config(config: CalibrationConfig, seed: int) -> ScenarioConfig:
    """Convert calibration config into a scenario config."""
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


def p1_sys_params(scenario: Any) -> dict[str, FloatArray | float]:
    """Return the P1 parameter dictionary for one scenario."""
    return {
        "N_PRB": np.asarray(scenario.N_PRB, dtype=np.float64),
        "P_max": np.asarray(scenario.P_max, dtype=np.float64),
        "W_PRB": float(scenario.W_PRB),
        "N0": float(scenario.N0),
        "T_f": float(scenario.T_f),
        "eps": float(scenario.eps),
    }


def sample_slots(
    slow_slot_count: int,
    sample_count: int,
    rng: np.random.Generator,
) -> IntArray:
    """Sample slow-slot indices without replacement when possible."""
    count = min(sample_count, slow_slot_count)
    slots = np.sort(rng.choice(slow_slot_count, size=count, replace=False))
    return np.asarray(slots, dtype=np.int64)


def sample_visible_assignment(
    visibility_at_slot: FloatArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    """Sample one visible satellite assignment for every cell."""
    satellite_count, cell_count = visibility_at_slot.shape
    x = np.zeros((satellite_count, cell_count), dtype=np.float64)
    selected = np.zeros(cell_count, dtype=np.int64)
    for cell in range(cell_count):
        candidates = np.flatnonzero(visibility_at_slot[:, cell] > 0.5)
        if candidates.size == 0:
            raise ValueError(f"cell {cell} has no visible satellite")
        satellite = int(rng.choice(candidates))
        selected[cell] = satellite
        x[satellite, cell] = 1.0
    return x, selected


def calibrate_one_scenario(
    config: CalibrationConfig,
    seed: int,
    solver: str,
    use_dpp: bool,
) -> list[CalibrationRow]:
    """Run proxy-vs-oracle calibration for one generated scenario."""
    scenario = build_scenario(
        scenario_config(config, seed),
        SystemParams(),
        scenario_id=f"p2_proxy_{config.scale}_seed{seed}",
    )
    if not np.isclose(config.demand_multiplier, 1.0):
        scenario = replace(
            scenario,
            a=np.asarray(scenario.a * config.demand_multiplier, dtype=np.float64),
        )
    p2_data = _scenario_data(scenario)
    proxy = _capacity_proxy(p2_data)
    p1_solver = P1CVXSolver(
        scenario.S,
        scenario.C,
        scenario.M,
        p1_sys_params(scenario),
        solver=solver,
        use_dpp=use_dpp,
    )
    rng = np.random.default_rng(seed + 17117)
    rows: list[CalibrationRow] = []
    assignment_id = 0
    for slot in sample_slots(scenario.K, config.slots_per_instance, rng):
        for _ in range(config.assignments_per_slot):
            x, selected = sample_visible_assignment(scenario.v[:, :, slot], rng)
            result = p1_solver.solve(x, scenario.a[:, slot, :], scenario.g[:, :, slot])
            proxy_xi = proxy[selected, np.arange(scenario.C), slot]
            demand = scenario.a[:, slot, :].sum(axis=1)
            visible_count = scenario.v[:, :, slot].sum(axis=0).astype(np.int64)
            for cell in range(scenario.C):
                signed_error = float(proxy_xi[cell] - result.xi[cell])
                abs_error = abs(signed_error)
                rel_error = abs_error / max(abs(float(result.xi[cell])), 1e-9)
                rows.append(
                    CalibrationRow(
                        scenario_seed=seed,
                        slot=int(slot),
                        assignment_id=assignment_id,
                        cell=cell,
                        selected_satellite=int(selected[cell]),
                        visible_satellites=int(visible_count[cell]),
                        demand_bits=float(demand[cell]),
                        proxy_xi=float(proxy_xi[cell]),
                        oracle_xi=float(result.xi[cell]),
                        signed_error=signed_error,
                        abs_error=abs_error,
                        rel_error=float(rel_error),
                        oracle_status=result.status,
                        oracle_solve_time_s=float(result.solve_time),
                    )
                )
            assignment_id += 1
    return rows


def run_calibration(
    config: CalibrationConfig,
    instances: int,
    seed: int,
    solver: str,
    use_dpp: bool,
) -> list[CalibrationRow]:
    """Run a deterministic proxy calibration batch."""
    if instances <= 0:
        raise ValueError("instances must be positive")
    rows: list[CalibrationRow] = []
    for offset in range(instances):
        scenario_seed = seed + offset
        scenario_rows = calibrate_one_scenario(config, scenario_seed, solver, use_dpp)
        rows.extend(scenario_rows)
        logger.info(
            "seed=%s rows=%d mean_abs_error=%.4g",
            scenario_seed,
            len(scenario_rows),
            float(np.mean([row.abs_error for row in scenario_rows])),
        )
    return rows


def summarize(
    rows: list[CalibrationRow],
    config: CalibrationConfig,
    solver: str,
    use_dpp: bool,
) -> dict[str, Any]:
    """Return aggregate proxy calibration metrics."""
    if not rows:
        raise ValueError("rows must be non-empty")
    proxy_xi = np.array([row.proxy_xi for row in rows], dtype=np.float64)
    oracle_xi = np.array([row.oracle_xi for row in rows], dtype=np.float64)
    signed = np.array([row.signed_error for row in rows], dtype=np.float64)
    abs_error = np.abs(signed)
    rel_error = np.array([row.rel_error for row in rows], dtype=np.float64)
    demand = np.array([row.demand_bits for row in rows], dtype=np.float64)
    if np.std(proxy_xi) > 0.0 and np.std(oracle_xi) > 0.0:
        corr = float(np.corrcoef(proxy_xi, oracle_xi)[0, 1])
    else:
        corr = float("nan")
    return {
        "scenario": f"p2_proxy_{config.scale}",
        "solver": solver,
        "use_dpp": use_dpp,
        "config": asdict(config),
        "cell_samples": len(rows),
        "oracle_solves": len({(row.scenario_seed, row.assignment_id) for row in rows}),
        "proxy_xi_mean": float(np.mean(proxy_xi)),
        "oracle_xi_mean": float(np.mean(oracle_xi)),
        "signed_error_mean": float(np.mean(signed)),
        "abs_error_mean": float(np.mean(abs_error)),
        "abs_error_median": float(np.median(abs_error)),
        "abs_error_p95": float(np.quantile(abs_error, 0.95)),
        "abs_error_max": float(np.max(abs_error)),
        "rel_error_median": float(np.median(rel_error)),
        "rel_error_p95": float(np.quantile(rel_error, 0.95)),
        "rel_error_max": float(np.max(rel_error)),
        "overestimate_rate": float(np.mean(signed > 0.0)),
        "underestimate_rate": float(np.mean(signed < 0.0)),
        "pearson_corr": corr,
        "demand_bits_mean": float(np.mean(demand)),
        "oracle_status_counts": _oracle_status_counts(rows),
        "oracle_time_total_s": float(
            sum(
                row.oracle_solve_time_s
                for row in {(row.scenario_seed, row.assignment_id): row for row in rows}.values()
            )
        ),
        "groups": {
            "demand_quantiles": _demand_quantile_groups(rows),
            "oracle_xi_bins": _oracle_xi_bin_groups(rows),
            "visible_satellites": _visible_satellite_groups(rows),
        },
        "largest_abs_error_rows": _largest_abs_error_rows(rows, limit=10),
    }


def _group_metrics(group_rows: list[CalibrationRow]) -> dict[str, float | int]:
    """Return compact calibration metrics for a row subset."""
    proxy_xi = np.array([row.proxy_xi for row in group_rows], dtype=np.float64)
    oracle_xi = np.array([row.oracle_xi for row in group_rows], dtype=np.float64)
    signed = np.array([row.signed_error for row in group_rows], dtype=np.float64)
    abs_error = np.abs(signed)
    return {
        "count": len(group_rows),
        "proxy_xi_mean": float(np.mean(proxy_xi)),
        "oracle_xi_mean": float(np.mean(oracle_xi)),
        "signed_error_mean": float(np.mean(signed)),
        "abs_error_mean": float(np.mean(abs_error)),
        "abs_error_median": float(np.median(abs_error)),
        "abs_error_p95": float(np.quantile(abs_error, 0.95)),
        "overestimate_rate": float(np.mean(signed > 0.0)),
    }


def _oracle_status_counts(rows: list[CalibrationRow]) -> dict[str, int]:
    """Return oracle status counts by unique solve."""
    unique_solves = {(row.scenario_seed, row.assignment_id): row.oracle_status for row in rows}
    counts: dict[str, int] = {}
    for status in unique_solves.values():
        counts[status] = counts.get(status, 0) + 1
    return counts


def _demand_quantile_groups(rows: list[CalibrationRow]) -> list[dict[str, float | int | str]]:
    """Return error metrics grouped by demand quartile."""
    demand = np.array([row.demand_bits for row in rows], dtype=np.float64)
    edges = np.quantile(demand, [0.0, 0.25, 0.5, 0.75, 1.0])
    groups: list[dict[str, float | int | str]] = []
    for index, label in enumerate(["q1", "q2", "q3", "q4"]):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 3:
            group_rows = [row for row in rows if lower <= row.demand_bits <= upper]
        else:
            group_rows = [row for row in rows if lower <= row.demand_bits < upper]
        if not group_rows:
            continue
        groups.append(
            {
                "group": label,
                "demand_min_bits": float(lower),
                "demand_max_bits": float(upper),
                **_group_metrics(group_rows),
            }
        )
    return groups


def _oracle_xi_bin_groups(rows: list[CalibrationRow]) -> list[dict[str, float | int | str]]:
    """Return error metrics grouped by oracle xi intervals."""
    bins = [
        ("[0,0.25)", 0.0, 0.25),
        ("[0.25,0.5)", 0.25, 0.5),
        ("[0.5,0.75)", 0.5, 0.75),
        ("[0.75,1]", 0.75, 1.0 + 1e-12),
    ]
    groups: list[dict[str, float | int | str]] = []
    for label, lower, upper in bins:
        group_rows = [row for row in rows if lower <= row.oracle_xi < upper]
        if not group_rows:
            continue
        groups.append({"group": label, **_group_metrics(group_rows)})
    return groups


def _visible_satellite_groups(rows: list[CalibrationRow]) -> list[dict[str, float | int | str]]:
    """Return error metrics grouped by number of visible satellites."""
    groups: list[dict[str, float | int | str]] = []
    for visible_count in sorted({row.visible_satellites for row in rows}):
        group_rows = [row for row in rows if row.visible_satellites == visible_count]
        groups.append(
            {
                "group": str(visible_count),
                "visible_satellites": visible_count,
                **_group_metrics(group_rows),
            }
        )
    return groups


def _largest_abs_error_rows(
    rows: list[CalibrationRow],
    limit: int,
) -> list[dict[str, float | int | str]]:
    """Return the largest absolute-error row records for diagnosis."""
    ranked = sorted(rows, key=lambda row: row.abs_error, reverse=True)
    return [asdict(row) for row in ranked[:limit]]


def write_csv(path: Path, rows: list[CalibrationRow]) -> None:
    """Write cell-level calibration rows to CSV."""
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
    """Write a compact Markdown calibration summary."""
    metric_keys = [
        "cell_samples",
        "oracle_solves",
        "proxy_xi_mean",
        "oracle_xi_mean",
        "signed_error_mean",
        "abs_error_mean",
        "abs_error_median",
        "abs_error_p95",
        "abs_error_max",
        "rel_error_median",
        "rel_error_p95",
        "rel_error_max",
        "overestimate_rate",
        "underestimate_rate",
        "pearson_corr",
        "demand_bits_mean",
        "oracle_time_total_s",
    ]
    lines = ["# P2 Proxy Calibration", "", "| Metric | Value |", "|---|---:|"]
    for key in metric_keys:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.6g} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    groups = summary["groups"]
    lines.extend(
        [
            "",
            "## Oracle Status Counts",
            "",
            "| Status | Solves |",
            "|---|---:|",
        ]
    )
    for status, count in summary["oracle_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(_group_markdown("Demand Quantiles", groups["demand_quantiles"]))
    lines.extend(_group_markdown("Oracle Xi Bins", groups["oracle_xi_bins"]))
    lines.extend(_group_markdown("Visible Satellite Groups", groups["visible_satellites"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_markdown(title: str, groups: list[dict[str, Any]]) -> list[str]:
    """Format grouped diagnostics as a Markdown table."""
    if not groups:
        return []
    lines = [
        "",
        f"## {title}",
        "",
        "| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | "
        "Median abs error | P95 abs error | Overestimate rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            "| `{group}` | {count} | {oracle:.6g} | {proxy:.6g} | {signed:.6g} | "
            "{median:.6g} | {p95:.6g} | {over:.6g} |".format(
                group=group["group"],
                count=group["count"],
                oracle=group["oracle_xi_mean"],
                proxy=group["proxy_xi_mean"],
                signed=group["signed_error_mean"],
                median=group["abs_error_median"],
                p95=group["abs_error_p95"],
                over=group["overestimate_rate"],
            )
        )
    return lines


def write_plot(path: Path, rows: list[CalibrationRow], summary: dict[str, Any]) -> None:
    """Write a PNG diagnostic for proxy-vs-oracle calibration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proxy_xi = np.array([row.proxy_xi for row in rows], dtype=np.float64)
    oracle_xi = np.array([row.oracle_xi for row in rows], dtype=np.float64)
    signed = np.array([row.signed_error for row in rows], dtype=np.float64)
    demand = np.array([row.demand_bits for row in rows], dtype=np.float64)
    visible = np.array([row.visible_satellites for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    axes[0, 0].scatter(
        oracle_xi,
        proxy_xi,
        c=np.log10(np.maximum(demand, 1.0)),
        s=20,
        alpha=0.7,
    )
    axes[0, 0].plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("P1 CVX oracle xi")
    axes[0, 0].set_ylabel("P2 proxy xi")
    axes[0, 0].set_title(f"corr {summary['pearson_corr']:.3f}")

    axes[0, 1].hist(signed, bins=30, color="#4C78A8", alpha=0.85)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("proxy - oracle")
    axes[0, 1].set_ylabel("cell samples")
    axes[0, 1].set_title(f"mean bias {summary['signed_error_mean']:.3g}")

    axes[0, 2].scatter(np.log10(np.maximum(demand, 1.0)), np.abs(signed), s=20, alpha=0.7)
    axes[0, 2].set_xlabel("log10 demand bits")
    axes[0, 2].set_ylabel("absolute error")
    axes[0, 2].set_title(f"p95 abs {summary['abs_error_p95']:.3g}")

    demand_groups = summary["groups"]["demand_quantiles"]
    axes[1, 0].bar(
        [group["group"] for group in demand_groups],
        [group["abs_error_p95"] for group in demand_groups],
        color="#F58518",
    )
    axes[1, 0].set_xlabel("demand quantile")
    axes[1, 0].set_ylabel("p95 absolute error")
    axes[1, 0].set_title("tail error by demand")

    xi_groups = summary["groups"]["oracle_xi_bins"]
    axes[1, 1].bar(
        [group["group"] for group in xi_groups],
        [group["abs_error_median"] for group in xi_groups],
        color="#54A24B",
    )
    axes[1, 1].set_xlabel("oracle xi bin")
    axes[1, 1].set_ylabel("median absolute error")
    axes[1, 1].set_title("error by service regime")
    axes[1, 1].tick_params(axis="x", rotation=20)

    axes[1, 2].scatter(visible, np.abs(signed), s=20, alpha=0.7, color="#B279A2")
    axes[1, 2].set_xlabel("visible satellites")
    axes[1, 2].set_ylabel("absolute error")
    axes[1, 2].set_title("error vs visibility choice")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_artifacts(
    rows: list[CalibrationRow],
    summary: dict[str, Any],
    out_dir: Path,
    scale: str,
    no_plot: bool,
) -> CalibrationArtifacts:
    """Persist calibration rows and aggregate artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"p2_proxy_calibration_{scale}_n{summary['oracle_solves']}_{stamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    png_path = None if no_plot else out_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
    if png_path is not None:
        write_plot(png_path, rows, summary)
    return CalibrationArtifacts(
        rows_csv=str(csv_path),
        summary_json=str(json_path),
        summary_md=str(md_path),
        plot_png=str(png_path) if png_path is not None else None,
    )


def main() -> None:
    """Run the command-line proxy calibration."""
    args = parse_args()
    config = calibration_config(
        args.scale,
        args.slots_per_instance,
        args.assignments_per_slot,
        args.demand_multiplier,
    )
    rows = run_calibration(config, args.instances, args.seed, args.solver, args.use_dpp)
    summary = summarize(rows, config, args.solver, args.use_dpp)
    artifacts = write_artifacts(rows, summary, args.out_dir, args.scale, args.no_plot)
    logger.info("P2 proxy calibration artifacts: %s", json.dumps(asdict(artifacts), sort_keys=True))


if __name__ == "__main__":
    main()
