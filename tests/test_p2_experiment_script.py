"""Smoke tests for the P2 experiment command-line script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_p2_experiments_script_smoke(tmp_path: Path) -> None:
    """The P2 experiment script should write reusable arrays, summary, and plot."""
    command = [
        sys.executable,
        "scripts/run_p2_experiments.py",
        "--scale",
        "toy",
        "--seed",
        "0",
        "--out-dir",
        str(tmp_path),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    summaries = list(tmp_path.glob("p2_toy_seed0_*.json"))
    arrays = list(tmp_path.glob("p2_toy_seed0_*.npz"))
    plots = list(tmp_path.glob("p2_toy_seed0_*.png"))
    assert len(summaries) == 1
    assert len(arrays) == 1
    assert len(plots) == 1

    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["method"] == "milp"
    assert not summary["status"].startswith("infeasible")
    assert summary["handover_max_per_cell"] <= summary["budget_max_per_cell"]
    assert 0.0 <= summary["xi_min"] <= summary["xi_mean"] <= summary["xi_max"] <= 1.0

    with np.load(arrays[0]) as data:
        assert data["x"].shape == (3, 5, 8)
        assert data["h"].shape == (5, 8)
        assert data["xi"].shape == (5, 8)
        assert data["selected_satellite"].shape == (5, 8)
        assert np.all(data["handover_per_cell"] <= data["H"] + 1e-8)


def test_run_p2_experiments_script_rolling_smoke(tmp_path: Path) -> None:
    """The P2 script should also run the rolling-window method."""
    command = [
        sys.executable,
        "scripts/run_p2_experiments.py",
        "--scale",
        "toy",
        "--method",
        "rolling",
        "--window",
        "4",
        "--step",
        "2",
        "--seed",
        "0",
        "--out-dir",
        str(tmp_path),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    summaries = list(tmp_path.glob("p2_rolling_toy_seed0_*.json"))
    arrays = list(tmp_path.glob("p2_rolling_toy_seed0_*.npz"))
    plots = list(tmp_path.glob("p2_rolling_toy_seed0_*.png"))
    assert len(summaries) == 1
    assert len(arrays) == 1
    assert len(plots) == 1

    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["method"] == "rolling"
    assert summary["window"] == 4
    assert summary["step"] == 2
    assert not summary["status"].startswith("infeasible")
    assert summary["handover_max_per_cell"] <= summary["budget_max_per_cell"]
