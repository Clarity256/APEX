# P1 技术报告：凸优化资源分配内核

日期：2026-05-24

## 结论摘要

P1 已完成当前研究原型阶段的交付目标。仓库中已经包含可作为
ground truth 的 CVXPY 凸优化求解器，以及用于快速评估的 NumPy 对偶近似
求解器；二者接口稳定，测试和 benchmark 均已接入。

当前可以形成如下结论：

- P1-L1 CVXPY 求解器已经可以作为固定归属 `x`、需求 `a`、信道 `g`
  下的基准最优解。
- P1-L2 对偶近似在已提交的 medium overloaded benchmark 上与 L1 的效用
  非常接近，同时速度显著更快。
- P1 已经具备支撑 P2/P3 的工程条件，可作为资源分配 oracle、代理函数
  校准基准和后续实验的 fast-slot 内核。

需要注意的是：当前结论适用于研究原型阶段。若要写入最终论文，还需要
在更多负载、更多随机种子和更多场景尺度上补充统计实验。

## 问题范围

P1 处理的是快时间尺度资源分配问题。慢时间尺度的卫星-小区归属已经由
外层问题给定，P1 只负责在固定归属下分配 PRB 和功率。

输入包括：

- `x`：二值归属矩阵，形状 `[S, C]`。
- `a`：快时隙业务到达量，单位 bit，形状 `[C, M]`。
- `g`：大尺度信道增益，形状 `[S, C]`。
- 系统参数：`N_PRB`、`P_max`、`W_PRB`、`N0`、`T_f`、`eps`。

输出包括：

- PRB 分配 `n[S, C, M]`。
- 功率分配 `p[S, C, M]`。
- 实际服务量 `z[C, M]`。
- 小区需求满足率 `xi[C]`。
- 对数公平效用 `U = sum_c log(eps + xi_c)`。

## 已实现层级

| 层级 | 文件 | 作用 | 状态 |
|---|---|---|---|
| P1-L1 | `src/leo_alloc/solvers/p1_cvx.py` | CVXPY 凸优化 ground truth | 已完成 |
| P1-L2 | `src/leo_alloc/solvers/p1_dual.py` | 快速对偶近似求解器 | 已完成 |
| Benchmark | `scripts/run_p1_benchmark.py` | L1 vs L2 gap/speed 对比 | 已完成 |
| Tests | `tests/test_p1_correctness.py`, `tests/test_p1_dual.py` | 正确性与回归测试 | 已完成 |

## P1-L1 凸优化内核

P1-L1 的关键是将 Shannon rate 写成 perspective 形式：

```text
n * log(1 + alpha * p / n) = -rel_entr(n, n + alpha * p)
```

该写法使速率项关于 `(n, p)` 联合凹，从而满足 CVXPY 的 DCP 规则。
实现中在 `P1CVXSolver.__init__` 一次性构造问题图，`solve()` 只更新
参数并 warm start，避免每次重复建模。

主要工程处理：

- 使用 `bit_scale = 1e6` 对 bit 类变量做数值归一，降低求解器病态风险。
- 对零需求小区显式记录 `zero_demand=True`，并设置 `xi=1`。
- 对 `x=0` 的 satellite-cell pair，通过 `n`、`p` 上界强制为 0。
- 默认使用 MOSEK；当本地环境不可用时，按顺序尝试 CLARABEL、ECOS。

这部分是 P1 的最重要基准实现，后续所有近似算法都应以它作为校验对象。

## P1-L2 对偶近似

`P1DualSolver` 是当前的快速近似实现。它不是重新调用 CVXPY，而是使用
NumPy 在资源权重和需求满足率之间做迭代更新。

核心流程为：

1. 根据信道增益计算 spectral efficiency。
2. 根据当前满足率和剩余需求构造小区权重。
3. 在每个 satellite-slot 资源池内分配 PRB 和功率。
4. 按快时隙 backlog 因果约束计算实际服务量。
5. 更新满足率权重，直到收敛或达到最大迭代次数。

它返回与 CVXPY 求解器相同的 `P1Result`，因此下游模块可以在不改接口的
情况下切换 L1/L2。

## 测试覆盖

当前测试覆盖了以下场景：

- `S=C=M=1` 的解析解行为。
- 两个对称小区的资源分配对称性。
- `x=0` mask 对未归属 satellite-cell pair 的强制置零。
- 零需求小区的 `xi=1` 处理。
- 大需求量级下的数值稳定性。
- 非法输入，包括 shape 错误、非二值 `x`、负需求、负信道增益。
- 对偶近似在 mask 和资源约束下的可行性。
- 对偶近似与 CVX 在高需求 toy case 上的对比。
- benchmark 脚本 smoke test。

最近一次完整验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src/leo_alloc
```

最近记录结果：`86 passed`，Ruff 通过，Mypy 通过。

## Benchmark 结果

已提交 artifact：

- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.csv`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.json`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.md`
- `results/p1_benchmarks/p1_benchmark_medium_overloaded_n50_20260524T043323Z.png`

场景设置：

- `S = 4`，`C = 10`，`M = 20`。
- `N_PRB = 100`，`P_max = 100 W`。
- 每个快时隙需求范围：`1e3` 到 `2e4` bit。
- 有意引入 demand-channel mismatch。
- 共 50 个确定性随机实例。

汇总结果：

| 指标 | 数值 |
|---|---:|
| 实例数 | 50 |
| median relative utility gap | `3.7637e-4` |
| p95 relative utility gap | `1.0909e-2` |
| max relative utility gap | `1.2875e-2` |
| median speedup | `71.90x` |
| p05 speedup | `19.60x` |
| median CVX time | `0.12798 s` |
| median dual time | `0.00149 s` |
| mean CVX satisfaction | `0.135923` |
| mean dual satisfaction | `0.135341` |

结果解释：

- L2 对偶近似相对 L1 CVX 有稳定的大幅加速。
- median gap 接近 0，说明当前中等规模 overloaded 场景下近似质量较好。
- tail gap 约为 1% 量级，可作为工程近似，但论文中需要如实报告。
- 绝对满足率较低是 overloaded 场景的预期结果，有利于体现公平资源分配
  在拥塞环境中的区分度。

## 完成度判断

P1 当前可以认为完成了以下目标：

- 固定归属下的凸优化资源分配。
- 可复现 ground truth 生成。
- 快速近似求解。
- P2/P3 下游调用接口。
- 当前阶段 benchmark 和可视化需求。

P1 尚未达到最终论文实验闭环的部分包括：

- 更大规模、多负载、多随机种子的系统实验。
- MOSEK 与开源求解器结果/速度差异分析。
- 与 P2 capacity proxy 的系统校准。
- 面向论文表格的置信区间和统计显著性整理。

## 建议下一步

1. 扩展 P1 benchmark 到多个负载等级和更多随机种子。
2. 新增 P1-vs-P2-proxy 校准脚本，量化 P2 代理函数偏差。
3. 导出论文可直接引用的 CSV/Markdown 表格。
4. 保持 `P1CVXSolver` 作为 oracle，`P1DualSolver` 用于快速筛选和消融。
