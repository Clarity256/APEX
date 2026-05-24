# P2 技术报告：硬切换预算下的卫星-小区归属优化

日期：2026-05-24

## 结论摘要

P2 已完成当前 surrogate optimization 阶段的交付目标。仓库中已经包含：

- P2-L1 full-horizon MILP 基准求解器。
- P2-L2 rolling-window 分解求解器。
- P2-L3 面向当前可分线性 surrogate 的动态规划求解器。
- 可复现场景生成模块。
- L1/L2/L3 gap、speed、handover 对比 benchmark。

当前可以形成如下结论：

- P2-L1 是当前线性 surrogate 下的离线最优基准。
- P2-L2 rolling window 可行且比 full MILP 更快，但由于窗口只看局部未来，
  在当前 stress benchmark 中会使用更多 handover。
- P2-L3 DP 在当前可分 surrogate 下与 full MILP 的 score 完全一致，
  且速度更快，因此是目前最适合生成 P3 训练标签的实现。

关键边界：上述结论只针对当前 P2 linear capacity proxy surrogate。
它还不是经过 P1 CVX 全面校准后的最终非线性关联优化结论。

## 问题范围

P2 处理的是慢时间尺度卫星-小区归属优化。目标是在完整或局部未来信息
下，为每个小区在每个慢时隙选择可见卫星，同时严格满足逐小区 handover
预算。

输入来自 `ScenarioInstance` 兼容对象：

- 可见性 `v[S, C, K]`。
- 信道增益 `g[S, C, K]`。
- 需求 `a[C, K, M]`。
- 卫星资源参数 `N_PRB`、`P_max`。
- 逐小区 handover 预算 `H[C]`。
- handover 惩罚系数 `lambda_h`。

输出由 `P2Result` 承载：

- 归属序列 `x[S, C, K]`。
- 切换指示 `h[C, K]`。
- proxy 满足率 `xi[C, K]`。
- 对数效用 `U`。
- 每小区切换次数 `handover_per_cell`。
- 求解时间、MIP gap、状态信息。

## 已实现层级

| 层级 | 文件 | 作用 | 状态 |
|---|---|---|---|
| P2-L1 | `src/leo_alloc/solvers/p2_milp.py` | 基于 SciPy HiGHS 的 full-horizon MILP | 已完成 |
| P2-L2 | `src/leo_alloc/solvers/p2_rolling.py` | rolling-window MILP 分解 | 已完成 |
| P2-L3 | `src/leo_alloc/solvers/p2_dp.py` | 当前可分 surrogate 下的逐小区动态规划 | 已完成 |
| Benchmark | `scripts/run_p2_benchmark.py` | L1/L2/L3 gap 与 speed 对比 | 已完成 |
| Scenario | `src/leo_alloc/scenario/` | 可验证的场景生成、可见性、信道、需求模块 | 当前测试范围内完成 |

## 当前 P2 Surrogate

当前 P2 solver 实际优化的是线性 surrogate score：

```text
sum_{c,k} xi[c,k] - lambda_h * sum_{c,k} h[c,k]
```

其中 `xi[c,k]` 由预计算的 `capacity_proxy[s,c,k]` 给出，并由当前选择的
`x[s,c,k]` 决定。

该 capacity proxy 使用了：

- 可见性 mask。
- 信道相关的名义 spectral efficiency。
- 卫星 PRB 与功率预算。
- 基于可见需求的预期负载估计。

需要区分两个量：

- solver 优化的是线性 score，便于 MILP/DP 精确建模。
- `P2Result.U` 仍报告 log-fair utility：

```text
sum log(eps + xi) - lambda_h * sum(h)
```

因此当前 benchmark 中的 gap 使用“solver 实际优化的线性 score”，这是
L1/L2/L3 之间最公平的比较方式。

## P2-L1 Full MILP

`P2MILPSolver` 通过 `scipy.optimize.milp` 调用 HiGHS。

核心约束包括：

- 每个小区在每个慢时隙必须且只能选择一颗卫星。
- 归属必须满足可见性约束。
- handover 指示变量约束相邻慢时隙的归属变化。
- 每个小区必须满足硬 handover 预算。
- 满足率 `xi` 受所选卫星-小区 capacity proxy 上界约束。

该层是当前 surrogate 下的 full-horizon 离线最优基准，用来审计 L2/L3。

## P2-L2 Rolling Window

`P2RollingSolver` 将完整 horizon 切成多个窗口，每次只求解较短窗口内的
MILP，并只提交前 `step` 个慢时隙的决策。窗口之间会传递上一窗口最后的
归属状态，并扣减已经使用的 handover budget。

该方法适合 full MILP 过慢时使用，但它不是全局最优，因为每个窗口只能
看到有限未来。

当前观察到的行为：

- 在 stress benchmark 中比 full MILP 更快。
- score gap 较小。
- handover 总量高于 full MILP/DP，说明局部窗口可能过早消耗预算。

## P2-L3 Dynamic Programming

`P2DPSolver` 利用了当前 surrogate 的可分性。capacity proxy 预计算后，
不同小区之间没有耦合，因此每个小区可以单独求解一个动态规划问题。

DP 状态为：

```text
(slot, selected_satellite, used_handover_budget)
```

转移奖励为：

```text
capacity_proxy[s,c,k] - lambda_h * switch_indicator
```

如果某个小区不存在满足可见性和 handover 预算的路径，DP 会返回
infeasible，而不是静默放宽硬预算。

在当前 surrogate 下，DP 是精确算法；它避免了 MILP 对所有小区联合
branch-and-bound，因此在更大 horizon 上更适合做标签生成。

## 测试覆盖

当前测试覆盖了以下场景：

- 单小区稳定最佳卫星选择。
- `H=0` 时强制不切换。
- 可见性断裂导致的强制切换。
- 预算耗尽时返回 infeasible。
- 多小区独立 handover budget。
- rolling window 边界切换计数。
- full-window rolling 与 full MILP 等价。
- DP 与 full MILP 在小规模实例上的目标一致性。
- 场景生成器和 benchmark 脚本 smoke test。

最近一次完整验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src/leo_alloc
```

最近记录结果：`86 passed`，Ruff 通过，Mypy 通过。

## Benchmark 结果

已提交 artifact：

- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.csv`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.json`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.md`
- `results/p2_benchmarks/p2_benchmark_stress_n3_20260524T104945Z.png`

stress 场景设置：

- `S = 8`，`C = 30`，`K = 50`，`M = 10`。
- `H_per_cell = 12`。
- rolling window `10`，step `5`。
- 共 3 个确定性随机实例。
- 高需求基准 `50 Mbps`，低需求基准 `10 Mbps`。
- mismatch ratio `0.3`。

汇总结果：

| 指标 | 数值 |
|---|---:|
| 实例数 | 3 |
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

结果解释：

- DP 在 benchmark score 上与 full MILP 完全一致，同时速度显著更快。
- Rolling gap 很小，但 handover 使用更多，说明后续需要做窗口长度和预算
  策略消融。
- 当前 stress 场景下 `xi` 接近 1，说明该场景更多是在验证 association 和
  hard handover budget，而不是强资源拥塞下的公平性差异。后续应加入更
  overloaded 的 P2 场景。

## 完成度判断

P2 当前可以认为完成了以下目标：

- 硬 handover budget 建模。
- 当前线性 surrogate 下的 full-horizon MILP 基准。
- rolling-window 分解求解。
- 当前可分 surrogate 下的大 horizon DP 精确求解。
- 场景生成、smoke test 和 stress benchmark。
- 为 P3 初期 imitation learning / behavior cloning 提供标签来源。

P2 尚未达到最终论文实验闭环的部分包括：

- capacity proxy 仍需与 P1 CVX oracle 做系统校准。
- 需要更多随机种子和不同负载等级的 benchmark。
- rolling window 的 `window`、`step`、budget policy 需要消融。
- 论文中需要明确区分“实际优化的线性 score”和“报告的 log utility”。
- 可考虑评估非可分 surrogate 或 P1-calibrated surrogate 版本。

## 建议下一步

1. 新增 `run_p2_proxy_calibration.py`，采样 association 并比较 P2 proxy
   `xi` 与 P1 CVX `xi`。
2. 扩展 P2 benchmark 到更多 seeds 和至少两个 demand pressure regime。
3. 将 `P2DPSolver` 作为 P3 label generator 默认方案，同时保留
   `P2MILPSolver` 作为 audit baseline。
4. 对 rolling window 的 `window`、`step` 和 budget policy 做消融实验。
5. 在进入 P3 大规模训练前，先完成 P2 proxy calibration report。
