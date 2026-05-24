"""Smoke tests for the P2 proxy calibration command-line script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_p2_proxy_calibration_script_smoke(tmp_path: Path) -> None:
    """The calibration script should write CSV, JSON, and Markdown artifacts."""
    command = [
        sys.executable,
        "scripts/run_p2_proxy_calibration.py",
        "--scale",
        "toy",
        "--instances",
        "1",
        "--seed",
        "0",
        "--slots-per-instance",
        "1",
        "--assignments-per-slot",
        "1",
        "--out-dir",
        str(tmp_path),
        "--no-plot",
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    summaries = list(tmp_path.glob("p2_proxy_calibration_toy_n1_*.json"))
    rows = list(tmp_path.glob("p2_proxy_calibration_toy_n1_*.csv"))
    reports = list(tmp_path.glob("p2_proxy_calibration_toy_n1_*.md"))
    plots = list(tmp_path.glob("p2_proxy_calibration_toy_n1_*.png"))
    assert len(summaries) == 1
    assert len(rows) == 1
    assert len(reports) == 1
    assert len(plots) == 0

    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["scenario"] == "p2_proxy_toy"
    assert summary["cell_samples"] == 5
    assert summary["oracle_solves"] == 1
    assert 0.0 <= summary["proxy_xi_mean"] <= 1.0
    assert 0.0 <= summary["oracle_xi_mean"] <= 1.0
    assert summary["abs_error_median"] >= 0.0
    assert summary["abs_error_p95"] >= 0.0
