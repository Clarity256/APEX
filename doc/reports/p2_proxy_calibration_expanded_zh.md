# P2 Proxy Calibration 扩展报告

日期：2026-05-24

## 结论摘要

本轮扩展 calibration 的目标是回答：当前 P2 `capacity_proxy` 与 P1 CVX
oracle 的偏差在不同尺度和不同需求压力下是否稳定，以及后续算法优化应
优先改 DP/MILP 还是改 surrogate。

结论很明确：

- P2-L3 DP 本身不是当前主要瓶颈；它对当前 linear surrogate 是精确求解。
- 当前主要问题是 `capacity_proxy` 在“中度过载”区间存在明显误差和排序能力
  不足。
- 正常负载下 proxy 与 P1 oracle 几乎一致；极端过载下绝对误差也回落。
- 最大误差集中在过渡区：部分 cell 在 P1 中仍可高满足率服务，但 proxy
  过度保守；另一些 P1 已经低满足率的 cell，proxy 反而可能高估。

因此，下一步最有价值的算法工作不是重写 P2-L3，而是做
**load-aware / calibrated capacity proxy**。

## 扩展实验范围

本轮新增 5 组 calibration，共：

- 273 次 P1 CVX oracle solve。
- 4140 个 cell-level proxy-vs-oracle 样本。
- 覆盖 medium 与 stress 两个 P2 尺度。
- 覆盖 demand multiplier `1`、`10`、`100` 三个负载等级。

| Scale | Demand multiplier | Oracle solves | Cell samples | Artifact stem |
|---|---:|---:|---:|---|
| medium | 1 | 75 | 900 | `p2_proxy_calibration_medium_n75_20260524T130746Z` |
| medium | 10 | 75 | 900 | `p2_proxy_calibration_medium_n75_20260524T130803Z` |
| medium | 100 | 75 | 900 | `p2_proxy_calibration_medium_n75_20260524T130817Z` |
| stress | 10 | 24 | 720 | `p2_proxy_calibration_stress_n24_20260524T130857Z` |
| stress | 100 | 24 | 720 | `p2_proxy_calibration_stress_n24_20260524T130836Z` |

每组 artifact 均包含 `.csv`、`.json`、`.md`、`.png` 四类文件，位于：

```text
results/p2_proxy_calibration/
```

## 总体指标

| Scale | Demand multiplier | Proxy xi mean | Oracle xi mean | Mean signed error | Median abs error | P95 abs error | Pearson corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 1 | `0.999661` | `1.000000` | `-0.000339` | `1.16e-10` | `4.21e-08` | `0.0119` |
| medium | 10 | `0.460855` | `0.603049` | `-0.142194` | `0.280365` | `0.640425` | `0.1944` |
| medium | 100 | `0.046342` | `0.069454` | `-0.023112` | `0.030546` | `0.103211` | `0.2690` |
| stress | 10 | `0.317665` | `0.464084` | `-0.146419` | `0.210762` | `0.646470` | `0.2765` |
| stress | 100 | `0.031767` | `0.050594` | `-0.018827` | `0.021186` | `0.086683` | `0.2669` |

解释：

- `demand_multiplier=1` 时，系统容量充足，P1 oracle 基本都给出 `xi=1`。
  此时 proxy 几乎无误差。
- `demand_multiplier=10` 是当前 proxy 的主要失败区间：median abs error
  达到 `0.21-0.28`，p95 error 约 `0.64`。
- `demand_multiplier=100` 时，大多数 cell 都进入低满足率区间，绝对误差
  下降到 `0.02-0.03` median。
- mean signed error 全部为负，说明总体上 proxy 偏保守，但这掩盖了局部
  高估现象。

## 关键误差模式

### 1. 中度过载区间误差最大

medium/stress 的 multiplier `10` 都显示高误差：

| Scale | Median abs error | P95 abs error | Mean signed error |
|---|---:|---:|---:|
| medium, multiplier 10 | `0.280365` | `0.640425` | `-0.142194` |
| stress, multiplier 10 | `0.210762` | `0.646470` | `-0.146419` |

这说明 proxy 在“资源竞争刚开始变强，但系统还没有完全拥塞”的区域最难
拟合 P1。该区域也是 P2 association 决策最敏感的区域，因为不同 satellite
选择会显著改变 cell 的实际满足率。

### 2. Proxy 对低满足率 cell 容易高估，对高满足率 cell 容易低估

medium multiplier `10` 的 oracle xi 分组：

| Oracle xi bin | Count | Oracle xi mean | Proxy xi mean | Mean signed error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 180 | `0.141679` | `0.458104` | `+0.316425` | `0.658028` | `0.994444` |
| `[0.25,0.5)` | 155 | `0.411246` | `0.397645` | `-0.013601` | `0.357373` | `0.419355` |
| `[0.5,0.75)` | 235 | `0.624903` | `0.412253` | `-0.212650` | `0.445287` | `0.140426` |
| `[0.75,1]` | 330 | `0.929231` | `0.526655` | `-0.402576` | `0.690606` | `0.045455` |

stress multiplier `10` 也有相同方向：

| Oracle xi bin | Count | Oracle xi mean | Proxy xi mean | Mean signed error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 200 | `0.122955` | `0.301712` | `+0.178756` | `0.370302` | `0.965000` |
| `[0.25,0.5)` | 233 | `0.396164` | `0.273848` | `-0.122316` | `0.298925` | `0.154506` |
| `[0.5,0.75)` | 161 | `0.623320` | `0.345246` | `-0.278074` | `0.478737` | `0.037267` |
| `[0.75,1]` | 126 | `0.927687` | `0.388770` | `-0.538917` | `0.805822` | `0.000000` |

这说明当前 proxy 有明显的“压缩动态范围”现象：它把真实 P1 的高 `xi` 往下
压，把真实 P1 的低 `xi` 往上抬。对 P2 来说，这会削弱 satellite 选择的
排序能力。

### 3. 可见卫星越少，低估风险通常越强

medium multiplier `10` 的 visible-satellite 分组显示：

| Visible satellites | Count | Oracle xi mean | Proxy xi mean | Mean signed error | P95 abs error |
|---|---:|---:|---:|---:|---:|
| 1 | 18 | `0.560884` | `0.127640` | `-0.433243` | `0.836624` |
| 2 | 177 | `0.612211` | `0.309359` | `-0.302851` | `0.672941` |
| 3 | 324 | `0.608067` | `0.437158` | `-0.170910` | `0.651411` |
| 4 | 381 | `0.596516` | `0.567129` | `-0.029388` | `0.599565` |

这提示当前 expected-load proxy 对低可见性区域可能过度悲观。后续可以考虑
把“实际已选同星 cell 负载”或“可见集合大小”显式纳入修正项。

## Oracle 可靠性备注

多组 medium/stress 校准中，P1 CVX oracle 出现了较多 `optimal_inaccurate`：

| Scale | Demand multiplier | Status counts |
|---|---:|---|
| medium | 1 | `optimal_inaccurate: 29`, `optimal: 46` |
| medium | 10 | `optimal_inaccurate: 42`, `optimal: 33` |
| medium | 100 | `optimal_inaccurate: 26`, `optimal: 49` |
| stress | 10 | `optimal_inaccurate: 23`, `optimal: 1` |
| stress | 100 | `optimal_inaccurate: 18`, `optimal: 6` |

这不推翻当前趋势，但说明下一轮 paper-grade calibration 需要同时做 P1 数值
稳定性增强，例如 solver option tuning、bit scaling sensitivity、或对一部分
样本用更严格设置复核。

## 对后续算法优化的含义

当前数据支持以下判断：

1. 不应优先重写 P2-L3。DP 对当前 surrogate 已经是精确求解器。
2. 应优先改 `capacity_proxy`，尤其是中度过载区间的动态范围压缩问题。
3. 修正方向应关注排序能力，而不仅是平均误差。P2 association 需要选对
   satellite，proxy 的 rank correlation 比单点误差更关键。
4. 极端过载下 absolute error 较低，但 `xi` 都很小，这类场景对 association
   的区分度有限；真正能体现 P2 价值的是 multiplier `10` 一类的过渡区。

## 建议的下一步改进方向

建议按以下顺序推进：

1. **Calibrated affine correction**
   对 `proxy_xi` 做分段线性校正，例如按 demand pressure 或 proxy bin 学习
   `oracle_xi ≈ clip(a * proxy_xi + b, 0, 1)`。这是最小改动。

2. **Load-aware correction**
   当前 proxy 预估 expected load，但没有利用具体 association 下的同星负载。
   可以在 P2 求解前或迭代中估计 selected satellite 的实际 demand load。

3. **Piecewise service-regime correction**
   针对当前发现的动态范围压缩，在低 `proxy_xi`、中 `proxy_xi`、高 `proxy_xi`
   三段分别校正。

4. **Oracle numerical hardening**
   对 calibration 子集使用更严格 P1 solver 设置复核，降低
   `optimal_inaccurate` 对结论的干扰。

5. **Rank-based evaluation**
   下一轮报告应增加同一 cell-slot 下不同 satellite 的排序准确率，例如
   top-1 agreement、Kendall tau 或 regret against P1 oracle。

## 当前判断

P2-L3 仍然可以作为当前 label generator 使用，但在论文主结论中必须保留
“linear surrogate”限定。进入 P3 前，可以先使用现有 DP 生成标签；进入最终
论文实验前，应完成 proxy correction 和 rank-based calibration。
