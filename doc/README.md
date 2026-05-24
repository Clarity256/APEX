# LEO Direct-Access Downlink Resource Allocation — Project Documentation

研究项目：双时间尺度公平资源分配与在线调度，面向 LEO 直连接入下行场景。

## 文档结构

本文档集面向**与 AI 编程助手协作开发**而设计。每个文件有明确的单一职责，可独立交付给 Claude Code 或 Codex 进行模块实现。

## 项目目标速览

本项目要构建一套 LEO 直连接入下行资源分配研究原型：先用凸优化解决给定归属下的 PRB+功率联合分配，再用 MILP 在完整未来信息下优化慢时隙卫星-小区归属和逐小区切换预算，最后用 BC+PPO 的分层 RL 策略在需求不确定时进行在线调度。论文目标是围绕“硬切换预算 + 需求不确定 + 公平资源分配”形成可投稿的 WCNC 2027 级别实验闭环。

## 当前仓库状态

当前仓库已经完成 P1/P2 研究原型的主要实现：`src/` 下包含 P1
凸优化/对偶求解器、P2 MILP/rolling/DP 求解器和场景生成模块，`tests/`
包含对应回归测试，`scripts/` 包含可复现实验入口。早期 `02_`、`03_`
规格文档仍保留原始设计语境；实际可运行状态以根目录 `README.md` 和
`doc/reports/` 技术报告为准。

```
doc/
├── README.md                       ← 你在这里
├── 00_research_context.md          研究目标与论文 claim
├── 01_system_architecture.md       项目代码架构与模块依赖
├── 02_p1_kernel_spec.md            P1-cvx 凸优化内核详规（首先实现）
├── 03_p2_milp_spec.md              P2-MILP 归属优化详规
├── 04_p3_rl_spec.md                P3 分层强化学习详规
├── 05_experiment_design.md         实验场景、参数、评估指标
├── 06_coding_conventions.md        代码规范与项目结构
├── 07_vibe_coding_guide.md         AI 协作开发方法论
├── 08_environment_setup.md         环境配置与手动安装项
└── reports/                        P1/P2 当前实现技术报告
```

## 当前技术报告

- `reports/p1_technical_report.md` — P1 convex allocation report, English
- `reports/p1_technical_report_zh.md` — P1 凸优化内核与对偶加速报告，中文
- `reports/p2_technical_report.md` — P2 handover-constrained association report, English
- `reports/p2_technical_report_zh.md` — P2 硬切换预算归属优化报告，中文
- `reports/p2_proxy_calibration_expanded_zh.md` — P2 proxy 扩展校准报告，中文

## 推荐阅读顺序

**第一次阅读（理解全局，约 30 分钟）**：
1. `00_research_context.md` — 这个项目要做什么、为什么做
2. `01_system_architecture.md` — 代码会怎么组织
3. `05_experiment_design.md` — 最终要交付什么实验

**开始实现前（每次开新模块前，约 10 分钟）**：
1. `06_coding_conventions.md` — 代码规范刷新
2. `07_vibe_coding_guide.md` — 协作方法论刷新
3. 对应模块的 spec 文件（`02_` / `03_` / `04_`）

## 开发推进顺序

```
Phase 1 : P1-cvx 内核        → 02_p1_kernel_spec.md
Phase 2 : P2-MILP 归属优化   → 03_p2_milp_spec.md
Phase 3 : P3 分层 RL         → 04_p3_rl_spec.md
Phase 4 : 完整实验与论文撰写 → 05_experiment_design.md
```

**重要原则**：严格按顺序推进。P1 是 P2/P3 的依赖；P2 是 P3 的训练数据来源。任何阶段未完成验证测试，不进入下一阶段。

## 给 AI 编程助手的提示

如果你是 Claude Code / Codex 在阅读本文档：

1. 首先阅读 `06_coding_conventions.md` 和 `07_vibe_coding_guide.md`
2. 不要跨模块"自由发挥"——严格按 spec 文件实现接口
3. 每完成一个函数，先运行 spec 中规定的验证测试，再继续
4. 对架构层面的疑问（"是否应该这样组织"），询问开发者而不是自行决策

## 文档完备性边界

这套文档已经覆盖研究目标、核心数学模型、模块接口、实验协议和编码规范。仍需在实现过程中持续补充三类内容：

1. 根目录 `README.md` 的可运行快速开始，需要等最小代码骨架完成后再写。
2. 每个模块的 `README.md` 与 API docs，需要随对应模块实现同步生成。
3. 论文中的外部参数和文献表述，在正式投稿前必须回到 `doc/paper/` 原文逐项核对。
