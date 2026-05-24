# P1 Technical Report: Convex Resource Allocation Kernel

Date: 2026-05-24

## Executive Conclusion

P1 is complete for the current research-prototype milestone. The repository now
contains a convex CVXPY ground-truth solver and a fast NumPy dual-style
approximation, both exposed through stable public APIs and covered by unit tests
and benchmark scripts.

The current conclusion is:

- The L1 CVXPY solver is suitable as the P1 ground-truth baseline under fixed
  association, demand, and channel inputs.
- The L2 dual approximation is much faster and tracks the L1 utility closely on
  the committed overloaded medium benchmark.
- P1 is ready to support P2/P3 as a resource-allocation oracle or calibration
  baseline, but larger paper-scale sweeps should still be run before final
  publication claims.

## Scope

P1 solves fast-timescale PRB and power allocation after the slow-timescale
satellite-cell association has already been fixed.

Inputs:

- `x`: binary association, shape `[S, C]`.
- `a`: fast-slot demand arrivals in bits, shape `[C, M]`.
- `g`: large-scale channel gains, shape `[S, C]`.
- system parameters: `N_PRB`, `P_max`, `W_PRB`, `N0`, `T_f`, `eps`.

Outputs:

- PRB allocation `n[S, C, M]`.
- power allocation `p[S, C, M]`.
- served bits `z[C, M]`.
- satisfaction ratio `xi[C]`.
- log-fair utility `U = sum_c log(eps + xi_c)`.

## Implemented Layers

| Layer | File | Role | Status |
|---|---|---|---|
| P1-L1 | `src/leo_alloc/solvers/p1_cvx.py` | CVXPY convex ground truth | Complete |
| P1-L2 | `src/leo_alloc/solvers/p1_dual.py` | Fast approximate solver | Complete |
| Benchmark | `scripts/run_p1_benchmark.py` | L1 vs L2 gap/speed comparison | Complete |
| Tests | `tests/test_p1_correctness.py`, `tests/test_p1_dual.py` | correctness and regression coverage | Complete |

## L1 Convex Kernel

The L1 solver uses the perspective form of the Shannon rate expression:

```text
n * log(1 + alpha * p / n) = -rel_entr(n, n + alpha * p)
```

This keeps the rate term concave in `(n, p)` and makes the optimization problem
DCP compliant. The implementation constructs the CVXPY problem once in
`P1CVXSolver.__init__`, updates parameters in `solve()`, and uses warm start for
repeated calls.

Key engineering choices:

- `bit_scale = 1e6` normalizes bit-valued variables for solver stability.
- `zero_demand` cells are explicitly marked and assigned `xi = 1`.
- `x = 0` satellite-cell pairs are masked through upper bounds on `n` and `p`.
- The preferred solver is MOSEK, with CLARABEL and ECOS fallback when installed.

## L2 Fast Approximation

`P1DualSolver` implements a fast dual-weighted allocation approximation. It
uses NumPy-only updates and a damping schedule to repeatedly:

1. compute spectral-efficiency-weighted resource shares,
2. allocate PRBs and power within satellite-slot resource budgets,
3. compute served demand subject to causal backlog,
4. update satisfaction weights.

It returns the same `P1Result` dataclass as the CVXPY solver, which keeps the
downstream interface stable.

## Validation Coverage

The current tests cover:

- `S=C=M=1` analytical behavior.
- symmetric cells.
- association masking.
- zero-demand cells.
- large demand scale stability.
- invalid input rejection.
- dual feasibility under masks and resource constraints.
- dual vs CVX agreement on high-demand toy instances.
- benchmark script smoke coverage.

Latest full validation command:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src/leo_alloc
```

Last recorded result: `86 passed`, Ruff passed, Mypy passed.

## Benchmark Result

Committed artifact set:

- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.csv`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.json`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.md`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.png`

Scenario:

- `S = 4`, `C = 10`, `M = 20`.
- `N_PRB = 100`, `P_max = 100 W`.
- demand range: `1e3` to `2e4` bit per fast slot.
- demand-channel mismatch is intentionally injected.
- 50 deterministic instances.

Summary:

| Metric | Value |
|---|---:|
| instances | 50 |
| median relative utility gap | `3.7637e-4` |
| p95 relative utility gap | `1.0909e-2` |
| max relative utility gap | `1.2875e-2` |
| median speedup | `71.90x` |
| p05 speedup | `19.60x` |
| median CVX time | `0.12798 s` |
| median dual time | `0.00149 s` |
| mean CVX satisfaction | `0.135923` |
| mean dual satisfaction | `0.135341` |

Interpretation:

- The dual approximation is consistently much faster than the CVX baseline.
- Median objective loss is nearly zero at the current medium overloaded scale.
- Tail gap remains around one percent, which is acceptable for engineering
  acceleration but should be reported honestly if used as a solver baseline.
- Low absolute satisfaction is expected in this overloaded stress setting; it
  creates useful separation for fair-allocation experiments.

## Completion Assessment

P1 is considered complete for:

- fixed-association convex allocation,
- reproducible ground-truth generation,
- fast approximate evaluation,
- downstream use by P2/P3 development,
- current benchmark/visualization needs.

P1 is not yet complete for final paper claims until:

- larger multi-scale sweeps are run,
- sensitivity to MOSEK vs open solvers is reported,
- P1 is calibrated against the P2 association proxy,
- final tables include confidence intervals or more seeds.

## Recommended Next Work

1. Run a paper-scale P1 benchmark with more seeds and multiple load levels.
2. Add a P1 proxy-calibration script for P2's capacity proxy.
3. Export benchmark tables in a manuscript-ready CSV/Markdown format.
4. Keep `P1CVXSolver` as the canonical oracle and use `P1DualSolver` for fast
   screening and ablations.
