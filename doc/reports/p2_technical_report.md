# P2 Technical Report: Handover-Constrained Association Optimization

Date: 2026-05-24

## Executive Conclusion

P2 is complete for the current surrogate-optimization milestone. The repository
now contains:

- P2-L1 full-horizon MILP baseline,
- P2-L2 rolling-window decomposition,
- P2-L3 dynamic programming solver for the current separable linear surrogate,
- generated scenario support,
- L1/L2/L3 benchmark artifacts.

The current conclusion is:

- P2-L1 provides an exact offline baseline for the implemented linear surrogate.
- P2-L2 is feasible and faster than full MILP, but it can spend more handovers
  because each window optimizes with limited future context.
- P2-L3 matches the full MILP score exactly on the current separable surrogate
  and is the best implemented solver for larger horizons.

The main boundary is important: this conclusion applies to the current P2
linear capacity-proxy surrogate, not yet to a fully calibrated P1-in-the-loop
nonlinear association objective.

## Scope

P2 optimizes slow-timescale satellite-cell association over `K` slow slots under
hard per-cell handover budgets.

Inputs come from `ScenarioInstance`-compatible objects:

- visibility `v[S, C, K]`,
- channel `g[S, C, K]`,
- demand `a[C, K, M]`,
- satellite resource parameters,
- per-cell handover budget `H[C]`,
- handover penalty `lambda_h`.

Outputs are carried by `P2Result`:

- association `x[S, C, K]`,
- handover indicators `h[C, K]`,
- proxy satisfaction `xi[C, K]`,
- log utility `U`,
- `handover_per_cell`,
- solve diagnostics.

## Implemented Layers

| Layer | File | Role | Status |
|---|---|---|---|
| P2-L1 | `src/leo_alloc/solvers/p2_milp.py` | full-horizon MILP baseline via SciPy HiGHS | Complete |
| P2-L2 | `src/leo_alloc/solvers/p2_rolling.py` | rolling-window MILP decomposition | Complete |
| P2-L3 | `src/leo_alloc/solvers/p2_dp.py` | per-cell dynamic programming for separable surrogate | Complete |
| Benchmark | `scripts/run_p2_benchmark.py` | L1/L2/L3 gap and speed comparison | Complete |
| Scenario | `src/leo_alloc/scenario/` | generated visibility/channel/demand instances | Complete for current tests |

## Current P2 Surrogate

The implemented P2 solvers optimize a linear surrogate score:

```text
sum_{c,k} xi[c,k] - lambda_h * sum_{c,k} h[c,k]
```

where `xi[c,k]` is bounded by a precomputed capacity proxy
`capacity_proxy[s,c,k]` selected by `x[s,c,k]`.

The proxy uses:

- visibility masks,
- channel-dependent nominal spectral efficiency,
- satellite PRB and power budgets,
- expected visible demand load.

The reported `U` still evaluates the log-fair expression:

```text
sum log(eps + xi) - lambda_h * sum(h)
```

However, for fair solver-to-solver benchmark comparisons, the gap table uses
the exact linear score that the current MILP/rolling/DP algorithms optimize.

## L1 Full MILP

`P2MILPSolver` uses SciPy HiGHS through `scipy.optimize.milp`.

Constraints include:

- every cell is assigned to exactly one satellite at each slow slot,
- assignment respects visibility,
- handover indicators upper-bound changes in assignment,
- each cell respects its hard handover budget,
- proxy satisfaction is limited by the selected satellite-cell capacity proxy.

The L1 MILP is the current offline ground-truth baseline for the implemented
surrogate.

## L2 Rolling Window

`P2RollingSolver` repeatedly solves MILP subproblems over a shorter horizon.
Each window commits only the leading `step` slots, carries the last committed
association into the next window, and updates the remaining handover budget.

This layer is useful when full-horizon MILP is too slow, but it is not globally
optimal because it has truncated future information.

Observed behavior:

- faster than full MILP on the stress benchmark,
- small score gap,
- higher total handover usage than full MILP/DP in the committed run.

## L3 Dynamic Programming

`P2DPSolver` exploits the current separable surrogate. Once capacity proxy
values are precomputed, cells do not couple with one another. Each cell can be
solved as a shortest/longest path dynamic program over:

```text
(slot, selected_satellite, used_handover_budget)
```

The transition reward is:

```text
capacity_proxy[s,c,k] - lambda_h * switch_indicator
```

The DP returns infeasible if no visibility-feasible path satisfies the hard
budget. Under the current surrogate, it is exact and avoids branch-and-bound
over all cells jointly.

## Validation Coverage

The current tests cover:

- stable best satellite selection,
- zero handover budget,
- forced visibility switches,
- infeasible budget exhaustion,
- multiple cells with independent budgets,
- rolling boundary handover accounting,
- rolling full-window equivalence to full MILP,
- DP vs full MILP objective agreement on small instances,
- generated scenario validation and benchmark-script smoke tests.

Latest full validation command:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src/leo_alloc
```

Last recorded result: `86 passed`, Ruff passed, Mypy passed.

## Benchmark Result

Committed artifact set:

- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.csv`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.json`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.md`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.png`

Stress scenario:

- `S = 8`, `C = 30`, `K = 50`, `M = 10`.
- `H_per_cell = 12`.
- rolling window `10`, step `5`.
- 3 deterministic instances.
- demand base: high `50 Mbps`, low `10 Mbps`.
- mismatch ratio `0.3`.

Summary:

| Metric | Value |
|---|---:|
| instances | 3 |
| rolling gap median | `0.001733` |
| rolling gap p95 | `0.002831` |
| rolling gap max | `0.002953` |
| DP gap median | `0` |
| DP gap max | `0` |
| rolling median speedup | `1.65x` |
| DP median speedup | `5.42x` |
| full MILP median time | `2.619 s` |
| rolling median time | `1.584 s` |
| DP median time | `0.510 s` |
| full MILP mean handovers | `47.0` |
| rolling mean handovers | `78.33` |
| DP mean handovers | `47.0` |
| full MILP mean xi | `0.999853` |
| rolling mean xi | `0.999846` |
| DP mean xi | `0.999853` |

Interpretation:

- DP exactly matches full MILP on the benchmark score while being substantially
  faster.
- Rolling achieves a small score gap, but uses more handovers in the current
  configuration.
- The high `xi` values indicate the stress scenario is mostly association and
  handover constrained, not strongly resource-starved under the current proxy.
  This is useful for validating hard budgets, but future experiments should add
  more overloaded P2 scenarios for stronger fairness separation.

## Initial Proxy Calibration

The first calibration script is now implemented at
`scripts/run_p2_proxy_calibration.py`. It samples visible P2 associations,
evaluates the P2 `capacity_proxy`, then calls `P1CVXSolver` on the same
slow-slot association to obtain the oracle `xi`.

Committed overloaded toy artifact set:

- `results/p2_proxy_calibration/p2_proxy_calibration_toy_n18_20260524T125458Z.csv`
- `results/p2_proxy_calibration/p2_proxy_calibration_toy_n18_20260524T125458Z.json`
- `results/p2_proxy_calibration/p2_proxy_calibration_toy_n18_20260524T125458Z.md`
- `results/p2_proxy_calibration/p2_proxy_calibration_toy_n18_20260524T125458Z.png`

Calibration settings:

- `S = 3`, `C = 5`, `K = 8`, `M = 10`.
- 3 scenarios, 3 sampled slow slots per scenario, 2 sampled assignments per slot.
- 18 P1 oracle solves and 90 cell-level samples.
- demand multiplier `100`, used to expose overloaded behavior.

Summary:

| Metric | Value |
|---|---:|
| proxy xi mean | `0.09906` |
| oracle xi mean | `0.11953` |
| mean signed error | `-0.02047` |
| median absolute error | `0.03669` |
| p95 absolute error | `0.20911` |
| max absolute error | `0.27038` |
| overestimate rate | `0.4667` |
| underestimate rate | `0.5333` |
| Pearson correlation | `0.3981` |

Interpretation:

- The proxy is directionally useful but not yet paper-grade calibrated.
- Mean signed error is negative, so the current proxy slightly underestimates
  P1 oracle satisfaction on this overloaded toy sample.
- The p95 error and modest correlation show that P2 claims should keep the
  "linear surrogate" qualifier until broader calibration and possible proxy
  correction are completed.

## Completion Assessment

P2 is considered complete for:

- current hard-budget association modeling,
- exact full-horizon linear-surrogate baseline,
- rolling-window benchmark,
- large-horizon DP solver under the separable surrogate,
- generated scenario smoke and stress benchmarks,
- producing labels for early P3 development.

P2 is not yet complete for final paper claims until:

- the capacity proxy is calibrated against P1 CVX outputs across larger regimes,
- larger stress sweeps are run with more seeds and load regimes,
- rolling-window budget policies are compared,
- the distinction between optimized linear score and reported log utility is
  finalized for the paper,
- optional nonseparable or P1-calibrated surrogate variants are evaluated.

## Recommended Next Work

1. Extend proxy calibration to medium/stress scales and multiple demand
   multipliers.
2. Use the calibration residuals to decide whether the proxy needs a correction
   model before final paper experiments.
3. Run P2 benchmarks with more seeds and at least two demand-pressure regimes.
4. Treat `P2DPSolver` as the default label generator for P3 while preserving
   `P2MILPSolver` as the audit baseline.
5. Add rolling-window ablations for `window`, `step`, and budget policy.
