"""Smoke tests for the P1 L1-vs-L2 benchmark script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_p1_benchmark_script_smoke(tmp_path: Path) -> None:
    """The benchmark script should write table, summary, and plot artifacts."""
    command = [
        sys.executable,
        "scripts/run_p1_benchmark.py",
        "--instances",
        "1",
        "--seed",
        "0",
        "--solver",
        "ECOS",
        "--out-dir",
        str(tmp_path),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    csv_files = list(tmp_path.glob("p1_benchmark_medium_overloaded_n1_*.csv"))
    json_files = list(tmp_path.glob("p1_benchmark_medium_overloaded_n1_*.json"))
    md_files = list(tmp_path.glob("p1_benchmark_medium_overloaded_n1_*.md"))
    png_files = list(tmp_path.glob("p1_benchmark_medium_overloaded_n1_*.png"))
    assert len(csv_files) == 1
    assert len(json_files) == 1
    assert len(md_files) == 1
    assert len(png_files) == 1

    summary = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert summary["scenario"] == "medium_overloaded"
    assert summary["instances"] == 1
    assert summary["solver"] == "ECOS"
    assert summary["gap_p95"] < 0.2
    assert summary["speedup_median"] > 1.0

    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "relative_gap" in csv_text
    assert "speedup" in csv_text
