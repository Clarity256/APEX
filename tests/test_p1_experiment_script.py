"""Smoke tests for the P1 experiment command-line script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_p1_experiments_script_smoke(tmp_path: Path) -> None:
    """The P1 experiment script should write reusable arrays and a summary."""
    command = [
        sys.executable,
        "scripts/run_p1_experiments.py",
        "--scale",
        "toy",
        "--method",
        "dual",
        "--out-dir",
        str(tmp_path),
        "--no-plot",
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    summaries = list(tmp_path.glob("p1_dual_toy_seed0_*.json"))
    arrays = list(tmp_path.glob("p1_dual_toy_seed0_*.npz"))
    assert len(summaries) == 1
    assert len(arrays) == 1

    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["method"] == "dual"
    assert summary["status"].startswith("dual_")
    assert summary["arrays_path"] == str(arrays[0])
    assert summary["plot_path"] is None

    with np.load(arrays[0]) as data:
        assert data["x"].shape == (2, 5)
        assert data["a"].shape == (5, 10)
        assert data["g"].shape == (2, 5)
        assert data["n"].shape == (2, 5, 10)
        assert data["p"].shape == (2, 5, 10)
        assert data["z"].shape == (5, 10)
        assert data["xi"].shape == (5,)
