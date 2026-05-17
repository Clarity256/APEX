# 00. Research Context

## 一句话定位

研究 LEO 直连接入下行场景中，双时间尺度（慢时隙归属 + 快时隙资源分配）下的公平资源分配与在线调度方法，重点解决**逐小区切换预算硬约束**和**需求不确定性**两个未被现有工作充分处理的问题。

## 三个研究问题与递进关系

```
问题一 (P1-cvx)             问题二 (P2-MILP)              问题三 (P3-RL)
快时隙资源分配         →    慢时隙归属优化         →    在线分层调度
（凸优化）                  （MILP）                      （RL + 凸优化内核）

【给定归属，如何】          【完整未来信息下，】          【需求不确定下，】
【最优分 PRB 与功率】       【如何选归属与切换】          【如何在线决策】

                            P1 作为代理函数            P1 作为低层模块
                            的 ground truth            BC 数据来自 P2
```

三个问题构成一个**有机的技术栈**，下层是上层的依赖：P1 提供"给定归属下的最优收益"，P2 在此基础上做组合优化，P3 在 P2 的离线最优基础上做在线学习。

## 关键技术差异点（对标现有工作）

| 维度 | 现有工作（ICC 2023, WiSEE 2025 等） | 本工作 |
|---|---|---|
| 资源类型 | 仅 PRB 或仅功率 | PRB + 功率**联合**凸优化 |
| 切换控制 | 软惩罚 $-\lambda h$ | 逐小区**硬预算** $\sum h_c \le H_c$ |
| 需求建模 | 慢时隙聚合需求 | 快时隙逐步到达 + 因果性约束 |
| 不确定性 | 离线优化为主 | RL + 凸优化内核 + BC 热启 |
| 公平性 | Jain 指数后验评估 | 对数效用前向优化 |

## 论文目标会议

**首选**：IEEE WCNC 2027（B 类会议，6 页双栏 IEEE 格式）

**叙事角度**：选择一个统一故事（详见 `05_experiment_design.md`），从三个问题中提炼一个完整 contribution。

**降级备选**：如果 RL 部分实验来不及，仅提交 P1+P2 即可独立成文。

## 关键文献（必读）

按优先级排序：

1. **Leyva-Mayorga et al., "Efficient and Fair Downlink Resource Allocation for LEO", ICC 2023** — 最直接的基线工作，需逐字精读
2. **Afif et al., "Joint Satellite Power Consumption and Handover Optimization for LEO", WiSEE 2025** — 切换软惩罚的对照工作
3. **Tang et al., "Joint Service Deployment and Task Scheduling for Satellite Edge Computing: A Two-Timescale Hierarchical Approach", JSAC 2024** — 双时间尺度方法论参考
4. **Agrawal et al., "Differentiable Convex Optimization Layers", NeurIPS 2019** — cvxpylayers 工具基础
5. **Schulman et al., "Proximal Policy Optimization Algorithms", 2017** — PPO 原始论文
6. **Mohsin et al., "Hierarchical DRL for Spectrum Sharing in NTN-TN Networks", AAAI 2025** — 分层 RL 在卫星网络的最新参考

## 关键术语统一

整个项目代码中使用以下术语，**禁止混用同义词**：

| 中文 | 英文 / 变量名 | 不要用 |
|---|---|---|
| 卫星 | satellite, `s` | spacecraft, sat |
| 小区 | cell, `c` | region, area |
| 慢时隙 | slow_slot, `k` | epoch, frame |
| 快时隙 | fast_slot, `m` | mini_slot, sub_slot |
| 归属 | association, `x` | assignment, mapping |
| 切换 | handover, `h` | switching, transfer |
| 需求满足率 | satisfaction_rate, `xi` | service_rate |
| 业务到达 | demand_arrival, `a` | arrival, traffic |
| 实际兑现服务量 | served_demand, `z` | delivered, fulfilled |

这个表是 Claude Code / Codex 写注释和命名时的硬约束。

## 一年时间表（毕业论文 + WCNC 同步）

| 月份 | 阶段 | 关键交付 |
|---|---|---|
| Month 1-2 | P1-cvx 实现 + 验证 | 论文 §3，可调用模块 |
| Month 3-4 | P2-MILP 实现 + 实验 | 论文 §4，BC 训练数据生成器 |
| Month 5-7 | P3-RL 实现 + 训练 | 论文 §5（核心创新章节） |
| Month 8 | 综合实验与图表 | 全部结果图、消融实验完成 |
| Month 9 | WCNC 投稿撰写 | 6 页论文初稿 |
| Month 10-11 | 毕业论文撰写 | 完整毕业论文 |
| Month 12 | 答辩准备 | PPT、答辩演练 |

**节奏判断准则**：如果某阶段超期 ≥ 2 周，立刻评估是否降级。Transformer 策略、L2 对偶分解的复杂版本都是可降级项。
