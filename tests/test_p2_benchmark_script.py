"""Smoke tests for the P2 full-MILP-vs-rolling benchmark script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_p2_benchmark_script_smoke(tmp_path: Path) -> None:
    """The P2 benchmark script should write CSV, JSON, Markdown, and PNG artifacts."""
    command = [
        sys.executable,
        "scripts/run_p2_benchmark.py",
        "--scale",
        "toy",
        "--instances",
        "2",
        "--seed",
        "0",
        "--out-dir",
        str(tmp_path),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    summaries = list(tmp_path.glob("p2_benchmark_toy_n2_*.json"))
    rows = list(tmp_path.glob("p2_benchmark_toy_n2_*.csv"))
    reports = list(tmp_path.glob("p2_benchmark_toy_n2_*.md"))
    plots = list(tmp_path.glob("p2_benchmark_toy_n2_*.png"))
    assert len(summaries) == 1
    assert len(rows) == 1
    assert len(reports) == 1
    assert len(plots) == 1

    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["scenario"] == "p2_toy"
    assert summary["instances"] == 2
    assert "gap_median" in summary
    assert "speedup_median" in summary
    assert summary["rolling_handover_total_mean"] >= 0.0
