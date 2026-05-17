# 05. Experiment Design

## 系统物理参数（参考 3GPP TR 38.821 + ICC 2023）

| 参数 | 符号 | 值 | 来源 |
|---|---|---|---|
| 载频 | $f$ | 2 GHz | 3GPP S 频段 NTN |
| 系统总带宽 | $B$ | 30 MHz | 5G NR FR1 |
| 单 PRB 带宽 | $W_{\text{PRB}}$ | 180 kHz | 5G NR 标准 |
| 噪声功率谱密度 | $N_0$ | $-174$ dBm/Hz | 标准热噪声 |
| 大气损耗 | $L_{\text{atm}}$ | 0.5 dB | TR 38.821 |
| 指向损耗 | $L_{\text{point}}$ | 3 dB | TR 38.821 |
| 卫星发射功率 | $P_s^{\max}$ | 75.35 dBm | ICC 2023 |
| 卫星天线增益 | $G_{\text{tx}}$ | 30 dBi | 直连接入典型值 |
| 用户天线增益 | $G_{\text{rx}}$ | 0 dBi | 手机直连 |
| 卫星轨道高度 | $h_{\text{orbit}}$ | 550 km | Starlink Shell 1 |
| 倾角 | $\delta$ | 53° | Starlink |
| 快时隙时长 | $T_f$ | 10 ms | 1 个 NR slot |
| 慢时隙时长 | $T_s$ | $M \cdot T_f$ | 派生 |
| 对数效用极小正数 | $\epsilon$ | $10^{-4}$ | 经验值 |
| 切换软惩罚 | $\lambda_h$ | $\in [0.05, 0.5]$ | 扫描 |

## 三档实验规模

### Toy（功能验证 + 论文 §3 图）
| 参数 | 值 |
|---|---|
| 卫星数 $S$ | 2 |
| 小区数 $C$ | 5 |
| 慢时隙数 $K$ | 10 |
| 快时隙数 $M$ | 10 |
| 切换预算 $H_c$ | $\{1, 2, 3\}$（扫描） |

**用途**：P1 单元测试、解析对照、KKT 性质图、调试。

### Medium（论文 §4 主结果 + §5 RL 训练）
| 参数 | 值 |
|---|---|
| 卫星数 $S$ | 4 |
| 小区数 $C$ | 10–15 |
| 慢时隙数 $K$ | 20–30 |
| 快时隙数 $M$ | 10 |
| 切换预算 $H_c$ | $K/4 \approx 5–7$ |

**用途**：主对比图、Pareto 前沿、消融实验。

### Stress（可扩展性展示，论文 §3 附图）
| 参数 | 值 |
|---|---|
| 卫星数 $S$ | 6–8 |
| 小区数 $C$ | 20–30 |
| 慢时隙数 $K$ | 50 |
| 快时隙数 $M$ | 10 |

**用途**：L2 vs L1 加速比、L3 拉格朗日松弛收敛、规模扩展性曲线。

## 场景生成协议

### 卫星轨道
- 数据源：Starlink TLE，从 [celestrak.org](https://celestrak.org) 下载
- 仿真起点：2026-01-01 00:00 UTC（固定）
- 工具：Skyfield 1.45+
- 选星策略：在目标服务区域（如欧洲中纬度）选可见的 $S$ 颗

### 服务区域
- **首选**：欧洲中纬度（40°N–55°N, 5°E–30°E），对标 ICC 2023
- **备选**：北美中纬度（35°N–50°N, -120°W–-70°W）
- 小区按经纬度均匀划分（每小区 0.15° × 0.15°）

### 信道增益
```python
def channel_gain(distance_m, freq_Hz):
    # 自由空间路径损耗
    FSPL_dB = 20*np.log10(distance_m) + 20*np.log10(freq_Hz) - 147.55
    # 对数正态阴影（NTN typical: σ=2 dB）
    shadow_dB = np.random.normal(0, 2.0)
    # 大气+指向损耗
    total_dB = FSPL_dB + shadow_dB + L_atm + L_point
    return G_tx * G_rx * 10**(-total_dB/10)
```

### 需求过程
**两类小区设计**（验证公平性的核心）：

```python
def generate_demand(C, K, M, mismatch_ratio=0.3):
    """
    Mismatch scenario: 
    - mismatch_ratio 的小区是"强信道低需求"
    - 其余是"弱信道高需求"
    """
    demand = np.zeros((C, K, M))
    
    for c in range(C):
        if c < int(C * mismatch_ratio):
            # Type A: high-load cells
            base = 5e7   # 50 Mbps base demand
        else:
            # Type B: low-load cells
            base = 1e7   # 10 Mbps base demand
        
        for k in range(K):
            # 周期性 + 泊松扰动
            periodic = 1 + 0.3 * np.sin(2 * np.pi * k / 24)
            lambda_k = base * periodic
            demand[c, k, :] = np.random.poisson(lambda_k * T_f, size=M)
    
    return demand
```

### 可见性矩阵
```python
def visibility_matrix(sat_positions, cell_positions, elevation_threshold_deg=20):
    """
    Returns v[s, c, k] = 1 if cell c sees satellite s at slot k with elevation > threshold
    """
    ...
```

生成 `v[s,c,k]` 后必须做硬预算可行性检查：对每个小区 $c$，确认在 $K$ 个慢时隙内存在一条可见卫星路径，且切换次数不超过 $H_c$。可用动态规划计算最小必要切换次数；若最小值大于 $H_c$，该 seed 应被过滤、重新采样或标记为 infeasible，不进入主实验。

## 主要 baseline 实现要求

| Baseline | 文件位置 | 作用 |
|---|---|---|
| B0 贪心 | `baselines/greedy.py` | 每慢时隙选 g 最大可见卫星 |
| B1 Oracle-P2 | `baselines/oracle.py` | 调用 P2 给出离线上界 |
| B2 MPC | `baselines/mpc.py` | 用预测值滚动求 P2 子问题 |
| B3 纯 PPO | `baselines/pure_ppo.py` | 无分层、无 BC、无 masking |
| B4 分层无 BC | RL 模块 | 从零 PPO 训练 |
| B5 分层+BC+PPO | RL 模块 | **主方法（MLP）** |
| B6 Transformer | RL 模块 | **主方法（Transformer，可选）** |

## 评估指标

### 性能指标
| 指标 | 定义 | 论文位置 |
|---|---|---|
| 平均满足率 | $\bar{\xi} = \frac{1}{KC}\sum_{k,c}\xi_{c,k}$ | §3, §4, §5 |
| Jain 公平指数 | $J = (\sum\xi)^2 / (n\sum\xi^2)$ | §3, §4, §5 |
| 最坏小区指标 | $\min_{c,k} \xi_{c,k}$ | §3, §4, §5 |
| 相对上界 gap | $(U_{B1} - U_{method}) / U_{B1}$ | §5 |
| 总服务量 | $\sum_{c,k} D_{c,k} \xi_{c,k}$ | §3 Pareto |

其中 $D_{c,k} = \sum_m a_{c,k,m}$。

### 切换指标
| 指标 | 定义 |
|---|---|
| 总切换次数 | $\sum_{c,k\ge 2} h_{c,k}$ |
| 最大单小区切换 | $\max_c \sum_k h_{c,k}$ |
| 预算违反次数 | 应严格为 0（验证 masking） |
| Emergency 切换次数 | 主实验应为 0；非 0 的 run 只能作为鲁棒性/不可行场景分析 |

### 计算指标
| 指标 | 定义 |
|---|---|
| 单次求解时间 | seconds |
| L2 vs L1 加速比 | $T_{L1} / T_{L2}$ |
| RL 训练 wall time | hours |
| RL 推理时间 | 每个慢时隙的推理 ms |

### 鲁棒性指标
| 指标 | 定义 |
|---|---|
| 预测误差敏感性 | 预测误差 $\in\{5\%, 10\%, 20\%\}$ 三档下的性能曲线 |
| 分布外泛化 | 训练分布 A，测试分布 B（如不同时段、不同区域） |

## 统计协议

**任何论文报告的数字都必须满足**：

1. **多次随机实验**：每个数据点至少 20 次独立随机种子，报告 mean ± std
2. **种子可重现**：种子号 $\in \{0, 1, ..., 19\}$ 固定
3. **配对对比**：B5 vs B2 必须在同一组场景上对比（相同 seed），不是分别独立采样
4. **显著性检验**：关键结论用 Wilcoxon signed-rank test，报告 p-value

## 必须画的图（论文最终交付）

### §3 P1 章节
| 图号 | 内容 | 类型 |
|---|---|---|
| Fig 1 | Jain 指数 vs 总吞吐量 Pareto 前沿 | 散点+连线 |
| Fig 2 | L1 vs L2 求解时间 vs 规模 | 双 y 轴 |
| Fig 3 | L2 gap 分布（toy/medium/stress） | 箱线图 |

### §4 P2 章节
| 图号 | 内容 | 类型 |
|---|---|---|
| Fig 4 | 切换预算 $H_c$ vs 平均满足率 trade-off | 曲线 |
| Fig 5 | 切换分布（最大-最小-中位数） | 柱状图 |
| Fig 6 | 代理函数校准散点 | 散点 |
| Fig 7 | 滚动窗口 vs 全窗口 MILP gap | 表格+柱状图 |

### §5 P3 章节
| 图号 | 内容 | 类型 |
|---|---|---|
| Fig 8 | 训练收敛曲线（reward vs episode） | 多曲线 |
| Fig 9 | 消融实验（B3/B4/B5/B6） | 柱状图 |
| Fig 10 | 鲁棒性（预测误差 vs 性能 gap） | 多曲线 |
| Fig 11 | B5 vs B2 在不同误差下的交叉 | 双曲线 |
| Fig 12 | 推理时间分布 | CDF |

**目标 12 张图，最终选 5–8 张放进 6 页 WCNC**。

## 关键论文 claim 与对应实验

为了避免"做了实验但没结论"，每个 claim 提前对应到具体实验：

| Claim | 验证实验 |
|---|---|
| C1: 联合 PRB-功率分配优于单独优化 | B1 (仅 PRB) vs L1 求解，Fig 1 |
| C2: 代理函数 gap < 10% | Fig 6 校准散点 |
| C3: 硬预算约束严格满足 | 表格：违反次数全为 0 |
| C4: BC 热启加速收敛 3-5× | B4 vs B5 收敛曲线 |
| C5: 学习方法在高预测误差时优于 MPC | Fig 11 交叉点 |
| C6: 分层设计降低约束违反 | B3 vs B5 违反统计 |
| C7: Transformer 增益（可选） | B5 vs B6 |

任何 claim 没有对应实验数据，都不能写进论文。
