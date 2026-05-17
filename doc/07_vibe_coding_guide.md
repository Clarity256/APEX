# 07. AI Collaboration Guide

本文档定义与 Claude Code / Codex 协作实现本项目时的工作方式。目标不是让 AI 自由扩展想法，而是让它按研究路线稳定产出可验证模块。

## 协作原则

1. **先读文档，再写代码**：每次新任务至少阅读 `00_research_context.md`、`01_system_architecture.md`、`06_coding_conventions.md` 和对应模块 spec。
2. **先锁接口，再补实现**：`ScenarioInstance`、`P1Result`、`P2Result`、RL env 的 public API 一旦写出，不在同一轮任务中随意改动。
3. **先测简单场景，再跑论文规模**：toy 解析测试失败时，不允许直接调 medium/stress 参数。
4. **先保证正确性，再优化速度**：P1 L1、P2 L1 是 correctness anchor；L2/L3 和 RL 训练只能在 anchor 可信后推进。
5. **实验 claim 必须可追溯**：每个论文结论都要能追溯到脚本、配置、seed 列表和结果文件。

## 每轮任务启动清单

交给 AI 编程助手前，明确以下信息：

| 项目 | 必填内容 |
|---|---|
| 阶段 | P1 / P2 / P3 / evaluation / paper |
| 目标文件 | 允许修改的目录或文件 |
| 验收命令 | 例如 `pytest tests/test_p1_correctness.py` |
| 不可改内容 | 例如 `ScenarioInstance` 接口、实验参数 |
| 输出物 | 代码、测试、图表、日志或文档 |

如果目标文件或验收命令不明确，AI 应先阅读本地结构并提出最小执行方案；不要猜测隐藏需求。

## 推荐提示模板

### 实现模块

```text
请实现 P1 L1 CVXPY 求解器。阅读 doc/02_p1_kernel_spec.md 和 doc/06_coding_conventions.md。
只修改 src/leo_alloc/solvers/p1_cvx.py 与 tests/test_p1_correctness.py。
先写 toy 解析测试，再实现代码。完成后运行 pytest tests/test_p1_correctness.py。
不要修改 ScenarioInstance 或 P2/P3 文件。
```

### 修复 bug

```text
以下测试失败：<粘贴失败摘要>。
请定位根因并修复，只做必要改动。
修复后运行同一测试，并说明失败原因、改动文件和剩余风险。
```

### 做实验

```text
请运行 P2 代理函数校准实验。
使用 configs/scenario_medium.yaml，seeds=0..19。
输出 results/p2_proxy_calibration/ 下的 csv 和散点图。
不得改动 solver 代码，除非发现明确 bug 并先说明。
```

## 阶段推进门槛

| 阶段 | 进入条件 | 退出条件 |
|---|---|---|
| P1 | 项目骨架和基础配置存在 | `02_p1_kernel_spec.md` 的 7 条完成判定全部满足 |
| P2 | P1 L1/L2 可 import 且测试通过 | `03_p2_milp_spec.md` 的 6 条完成判定全部满足 |
| P3 | P2 能生成 BC 数据 | `04_p3_rl_spec.md` 的主方法 B5 完成判定满足 |
| 综合实验 | B0-B5 均可运行 | `05_experiment_design.md` 的核心 claim 有数据支撑 |

任何阶段超期两周，应优先降级可选项：Transformer、P2 L3、大规模 stress、复杂预测器。不要降级 P1 correctness、硬预算约束和无未来信息泄漏测试。

## 变更审查清单

每次 AI 完成改动后，至少检查：

1. 是否遵守 `doc/00_research_context.md` 的术语命名。
2. 是否新增了 public API；若新增，是否有 docstring 和测试。
3. 是否引入未来信息泄漏，尤其是 P3 的 demand predictor 和 BC 数据。
4. 是否静默放宽硬切换预算。
5. 是否把 dB 与线性单位混用。
6. 是否修改了与当前任务无关的文件。
7. 是否给出可复现命令和实际运行结果。

## 交接记录格式

长任务中断或跨助手交接时，在回复中使用以下格式：

```text
已完成：
- ...

已修改：
- path/to/file.py：...

已验证：
- pytest ...

未完成/风险：
- ...

下一步：
- ...
```

交接记录只写事实，不写泛泛建议。
