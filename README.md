# LEO Resource Allocation

Research prototype for dual-timescale fair downlink resource allocation in LEO direct-access networks.

Start with the documentation in `doc/README.md`. The implementation is organized around:

1. P1: convex PRB-power allocation for fixed satellite-cell association.
2. P2: offline MILP association optimization with per-cell handover budgets.
3. P3: online hierarchical RL scheduling under demand uncertainty.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[solvers,rl,dev,notebook]"
```

For local runs in this workspace, use the helper below so Matplotlib and fontconfig write caches inside the project instead of the user home directory:

```bash
source scripts/activate_env.sh
```

Commercial solvers are optional at bootstrap time but required for paper-scale runs:

```bash
.venv/bin/python -m pip install -e ".[commercial-solvers]"
```

MOSEK and Gurobi also require local academic licenses.
