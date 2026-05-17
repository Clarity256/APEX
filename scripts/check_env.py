"""Check local project environment and solver availability."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import mosek

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = PROJECT_ROOT / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))


def check_imports() -> None:
    """Import core packages used by the project."""
    import gymnasium
    import numpy
    import pandas  # type: ignore[import-untyped]
    import scipy  # type: ignore[import-untyped]
    import skyfield
    import torch

    print("imports: ok")
    print(f"numpy: {numpy.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"gymnasium: {gymnasium.__version__}")
    print(f"skyfield: {skyfield.__version__}")


def check_cvxpy_solvers() -> None:
    """Report installed CVXPY solvers and solve a tiny problem with each main solver."""
    print(f"cvxpy solvers: {cp.installed_solvers()}")  # type: ignore[no-untyped-call]
    for solver in ["ECOS", "CLARABEL", "GUROBI", "MOSEK"]:
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 1) ** 2))
        try:
            prob.solve(solver=solver, verbose=False)  # type: ignore[no-untyped-call]
        except (cp.error.SolverError, cp.error.DCPError, cp.error.DPPError, mosek.Error) as exc:
            message = str(exc).splitlines()[0]
            print(f"{solver}: unavailable ({message})")
            continue
        value = float(x.value) if x.value is not None else float("nan")
        print(f"{solver}: {prob.status}, x={value:.6f}")


def main() -> None:
    """Run all environment checks."""
    check_imports()
    check_cvxpy_solvers()


if __name__ == "__main__":
    main()
