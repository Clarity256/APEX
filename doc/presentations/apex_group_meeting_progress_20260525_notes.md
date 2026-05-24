# APEX 组会汇报讲稿

日期：2026-05-25  
主题：APEX - Adaptive Policy with Embedded conveX optimization  
目标：让同门理解我要解决的 LEO 直连接入资源分配问题，以及目前 P1/P2 的实现和实验进展。

## 建议汇报节奏

- 总时长：12-15 分钟。
- 问题背景：3 分钟，讲清楚为什么需要双时间尺度。
- 已完成工作：6-8 分钟，重点讲 P1 oracle、P2 MILP/DP、benchmark。
- 当前结论和下一步：3-4 分钟，强调 proxy calibration 是下一阶段核心风险。

## Slide 1 - 标题和当前状态

这一页先把项目定位讲清楚：APEX 的核心想法是把凸优化内核嵌入到策略或组合优化流程里，用可解释、可复现的优化结果支撑后续在线策略。  

目前不是只做了一个 demo，而是已经形成了 P1/P2 的实验闭环：P1 有 CVX oracle 和 fast dual 近似，P2 有 full MILP、rolling window 和 DP。右侧四个数字用于建立可信度：P1/P2 benchmark 已有批量样本，P2 当前 gap 很小，proxy calibration 已经扩展到 cell-level 样本。

转场：接下来先解释为什么这个问题不能只当作普通资源分配。

## Slide 2 - 问题压力

LEO 直连接入的问题压力来自三个方向：高速移动导致可见性和信道变化，PRB/power 资源需要在快时隙内分配，切换次数又不能无限增加。  

重点说明：如果只做快时隙资源分配，会忽略未来 satellite 可见性和 handover；如果只做慢时隙归属，又会误判真实 PRB/power 分配之后的服务满足率。这个矛盾决定了我们需要分层建模。

转场：所以我把问题拆成 P1/P2/P3 三层。

## Slide 3 - P1/P2/P3 拆分

这里讲分解逻辑。P1 是快时隙资源分配，在固定 association 下分配 PRB 和功率，并输出真实满足率 xi。P2 是慢时隙归属优化，在可见性和 hard handover budget 下选择 satellite。P3 是未来的在线分层策略，会用 P1/P2 的结果作为 label 或 reward 支撑。  

要强调当前进度：P1/P2 已经完成研究原型，P3 之前必须把 P2 proxy 校准清楚。因为 P3 学到的策略质量最终会受 proxy 和 reward 的可靠性影响。

## Slide 4 - P1 内核

这一页讲 P1 为什么是可信的 ground truth。速率项用的是 perspective 形式：n log(1 + alpha p / n)，在 CVXPY 里用 `-rel_entr(n, n + alpha p)` 表达，满足 DCP/DPP。  

实现上，P1-L1 是 CVXPY + MOSEK 的 oracle；P1-L2 是 dual-weighted NumPy approximation，用来做快速 screening 和 ablation。测试覆盖了解析解、mask、zero demand 和 bit scale 等边界情况。  

讲法上可以说：P1 的角色不是替代所有算法，而是提供“给定归属之后真实资源分配效果”的基准。

## Slide 5 - P1 实验证据

这里展示 P1-L2 在 medium/overloaded benchmark 下的效果。核心结论是：L2 保留了接近 CVX oracle 的效果，同时快一个数量级以上。  

三个指标分别对应 gap、speed 和资源分配差异。可以强调目前 L2 适合大批量实验、screening 和 ablation；真正作为论文最优性基准时仍以 P1-L1 为 oracle。

可能被问：L2 是否严格最优？回答：不是。它是快速近似，P1-L1 才是 ground truth；L2 的价值是速度和可扩展性。

## Slide 6 - P2 建模

P2 把 LEO 可见性变化转成 hard-budget association 问题。变量包括 `x[s,c,k]` 表示慢时隙归属，`h[c,k]` 表示相邻慢时隙是否切换，`xi[c,k]` 是当前使用的 proxy satisfaction，`H[c]` 是每个 cell 的 hard handover budget。  

三层实现分别是：P2-L1 full MILP，用完整 horizon 做离线最优；P2-L2 rolling window，每次解短窗口，只提交前几个时隙；P2-L3 dynamic programming，在当前 linear surrogate 可分时逐 cell 精确求解。

转场：接下来给出 P2 benchmark 结果。

## Slide 7 - P2 实验证据

目前在 linear surrogate 下，P2-L3 DP 能精确匹配 full MILP，并且明显更快。P2-L2 rolling window 的价值在于可以模拟在线有限预见窗口，但它和 full horizon 的结果会受窗口大小影响。  

关键点：这里的 “gap=0” 只说明 DP 对当前 surrogate 是精确的，不代表 surrogate 本身已经等价于真实 P1。这个区分很重要。

## Slide 8 - 为什么要优化 surrogate

P2 不能直接把 P1 CVX 嵌入组合优化。真实目标应该是每个慢时隙归属后再调用 P1 CVX，得到真实 xi；但这样不可直接放进 MILP/DP。  

当前做法是用 linear capacity proxy 近似真实 xi，使 P2 可解、可调、可 benchmark。风险是：如果 proxy 排序不准，P2 会选错 satellite。  

这页要把问题从“P2-L3 算法是否快”转到“proxy 是否可信”。目前算法层面 DP 已经足够快，研究风险转移到了 proxy fidelity。

## Slide 9 - Calibration 扩展范围

这一页说明我已经把 calibration 扩展到 medium/stress 和多个负载等级。样本包括 273 次 P1 CVX oracle 求解和 4140 个 cell-level 样本，负载倍率覆盖 1、10、100。  

讲的时候可以解释为什么要做这么多负载：proxy 的偏差很可能不是均匀的，而是和系统负载、资源竞争强度有关。如果只看一个轻载场景，会过度乐观。

## Slide 10 - Proxy 关键发现

核心发现：误差最大不是极端拥塞，而是刚开始资源竞争的中度过载区间。  

从图中可以看到，当前 proxy 会压缩动态范围：低真实 xi 的 cell 容易被高估，高真实 xi 的 cell 容易被低估。这对 P2 很危险，因为 P2 依赖的是 association 的相对排序，而不只是平均误差。  

这页可以作为组会讨论重点：下一步应该评估 rank correlation、top-k regret 和 handover-aware regret，而不只是 MAE/RMSE。

## Slide 11 - 当前算法判断

这里给出阶段性判断。P1 CVX oracle 已经可用，是 ground truth；P1 dual 已经足够快，可做 screening；P2 MILP 是审计基准；P2 DP 在当前 surrogate 下精确且更快。  

真正的高风险模块是 P2 proxy。也就是说，当前更值得优化的是 surrogate 质量，而不是重写 P2-L3 求解器。  

希望同门反馈的方向：proxy correction 应该优先做线性校准、load-aware 修正，还是先补 rank-based evaluation。

## Slide 12 - 下一阶段计划

下一步目标是把 proxy 从“能跑”推进到“可信可解释”。计划分四步：先做 rank-based calibration，确认 proxy 是否保留 satellite 排序；再做 calibrated correction，修正系统性偏差；然后加入 load-aware proxy，把 demand load 或资源竞争强度纳入；最后再接 P3 label pipeline。  

结束语可以这样说：P1/P2 原型已经跑通，现在论文质量的关键风险已经收敛到 P2 proxy fidelity。后续如果 proxy 校准有效，就可以进入 P3 的在线策略训练；如果校准效果不好，则需要回到 P2 surrogate 设计。

## 可能被问到的问题

1. P1 和 P2 是否已经完成？
   - P1 算法和测试已经比较完整；P2 的 MILP/rolling window/DP 也已经有可复现实验。严格说，P2 求解器完成了研究原型，但 P2 目标函数里的 proxy 还需要校准，所以论文层面的 P2 还没有完全定稿。

2. P2-L3 为什么现在叫 DP？
   - 因为当前 linear surrogate 在 cell 之间可分，每个 cell 只需要在时间维度上处理可见性和 handover budget，因此可以用动态规划精确求解当前 surrogate。

3. 为什么不直接把 P1 放进 P2？
   - P1 是凸优化，P2 是组合优化。直接嵌套会变成非常重的 bilevel/mixed-integer nonlinear workflow，不适合作为大规模 benchmark 或在线策略训练的数据生成管线。

4. 当前最需要补的实验是什么？
   - proxy 的 rank-based evaluation：Spearman/Kendall、top-k satellite choice regret、handover-aware regret，以及按负载分层的误差分析。

5. 下一步是否开始 P3？
   - 可以开始准备 P3 label pipeline，但在正式训练策略前，应先把 P2 proxy 校准和评价指标稳定下来，否则 P3 会学到 proxy bias。
