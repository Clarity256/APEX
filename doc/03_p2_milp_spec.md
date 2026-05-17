# 03. P2-MILP Specification

> 实现前置条件：P1 模块全部测试通过，且可通过简单 import 调用。

## 模块职责

**单一职责**：在完整未来信息已知的离线场景下，对 K 个慢时隙联合优化卫星-小区归属 $\mathbf{x}$，最小化切换次数的同时最大化效用，**严格满足逐小区切换预算硬约束**。

**对外提供**：
1. 离线最优归属序列（论文 §4 主结果）
2. BC 训练数据（喂给问题三）
3. MPC 滚动基线（B2 baseline）

## 数学问题

### 决策变量

| 变量 | 形状 | 含义 |
|---|---|---|
| $x_{s,c,k}$ | `[S, C, K]`, $\in\{0,1\}$ | 归属指示 |
| $h_{c,k}$ | `[C, K]`, $\in\{0,1\}$ | 切换发生指示（$k \ge 2$） |
| $\xi_{c,k}$ | `[C, K]`, $\in[0,1]$ | 满足率 |

### 完整问题

$$
\begin{aligned}
\text{(P2)}\quad \max\quad & \sum_{k,c} \log(\epsilon + \xi_{c,k}) - \lambda_h \sum_{k\ge 2, c} h_{c,k} \\
\text{s.t.}\quad
& \sum_s x_{s,c,k} = 1, \quad \forall c, k \\
& x_{s,c,k} \le v_{s,c,k}, \quad \forall s,c,k \\
& h_{c,k} \ge x_{s,c,k} - x_{s,c,k-1}, \quad \forall s,c,k\ge 2 \\
& h_{c,k} \ge x_{s,c,k-1} - x_{s,c,k}, \quad \forall s,c,k\ge 2 \\
& \sum_{k\ge 2} h_{c,k} \le H_c, \quad \forall c \quad \text{(关键差异点)} \\
& \xi_{c,k} \le \tilde{V}_{c,k}(\mathbf{x}_k), \quad \forall c, k
\end{aligned}
$$

### 容量代理函数 $\tilde{V}_{c,k}$

理想情况下应取 $\hat{V}_{c,k}(\mathbf{x}_k) = $ P1Solver 的 $\xi^*$，但 MILP 内层调用 P1 不可行（每次评估都要解凸优化，组合爆炸）。

**代理函数构造**（卫星侧负载均衡近似）：

$$
\tilde{V}_{c,k}(\mathbf{x}_k) = \min\left\{\frac{\tilde{R}_{c,k}}{D_{c,k}}, 1\right\}, \quad
\tilde{R}_{c,k} = \sum_s x_{s,c,k} \cdot \omega_{s,c,k} \cdot \rho_{s,c,k}
$$

其中：
- $\rho_{s,c,k} = T_s W_{\text{PRB}} \log_2(1 + P_s^{\max} g_{s,c,k} / (W_{\text{PRB}} N_s^{\text{PRB}} N_0))$ 是 $(s,c)$ 独占卫星全部资源的名义速率
- $\omega_{s,c,k}(\mathbf{x}_k) = D_{c,k} / \sum_{c' \in \mathcal{C}_{s,k}} D_{c',k}$ 是负载均衡权重

这里 $D_{c,k} = \sum_m a_{c,k,m}$。若 $D_{c,k}=0$，实现时令 $\tilde{V}_{c,k}=1$，并在代理校准报告中单独统计 zero-demand 样本占比。

**关键**：$\omega$ 是 $\mathbf{x}$ 的非线性函数，需要在 MILP 中通过 McCormick 包络或离散化技巧线性化。

## 可行性假设

硬切换预算只在问题本身可行时有意义。场景生成器必须保证每个小区在预算 $H_c$ 内存在至少一条跨 $K$ 慢时隙的可见卫星路径；否则 P2 应返回 `infeasible`，而不是静默放宽预算。

主实验默认过滤或修复不可行场景。若为了鲁棒性测试保留不可行场景，必须把“因可见性断裂导致的强制 emergency handover”单独记为 `emergency_handover_count`，且该 run 不能用于证明“硬预算违反次数为 0”的主结论。

## 实现方案

### L1：直接 MILP 求解（小规模 ground truth）

#### 文件位置
`src/leo_alloc/solvers/p2_milp.py`

#### 适用场景
$K \le 10, C \le 20$。规模超过时 Gurobi 会爆。

#### 接口

```python
@dataclass
class P2Result:
    x: np.ndarray            # [S, C, K] 二值
    h: np.ndarray            # [C, K] 二值
    xi: np.ndarray           # [C, K] 实数 [0,1]
    U: float                 # 目标函数值
    handover_per_cell: np.ndarray  # [C]
    solve_time: float
    mip_gap: float           # Gurobi 报告的 gap

class P2MILPSolver:
    def __init__(self, scenario: ScenarioInstance, 
                 surrogate_v: Callable[[np.ndarray, ...], np.ndarray],
                 time_limit: float = 3600, mip_gap: float = 0.01):
        ...
    
    def solve(self) -> P2Result:
        ...
```

#### 实现要点

**要点 1：使用 gurobipy 而非 cvxpy 写 MILP**

cvxpy 支持 MILP 但性能不如直接调用 gurobipy。对于组合优化问题，直接用 gurobipy：

```python
import gurobipy as gp
from gurobipy import GRB

model = gp.Model()
x = model.addVars(S, C, K, vtype=GRB.BINARY, name='x')
h = model.addVars(C, K, vtype=GRB.BINARY, name='h')
xi = model.addVars(C, K, lb=0, ub=1, name='xi')
```

**要点 2：代理函数 $\omega$ 的线性化**

$\omega$ 含分式 $D_c / \sum_{c'} D_{c'} \cdot x_{s,c',k}$，是非线性的。处理方法：

**方案 A（推荐）**：预计算"负载场景"。对每个卫星 $s$ 每个慢时隙 $k$，列出所有可能的小区归属子集（如果可见小区不太多），预先算出每个子集对应的 $\omega$ 值，用 SOS1 约束选择。

**方案 B**：用对数效用近似 $\log(\xi) \approx \log(\tilde{R}/D)$，再把 $\tilde{R}$ 表示为 $\sum_s x_{s,c,k} \cdot \rho_{s,c,k} / |\mathcal{C}_{s,k}|$（假设均匀分配），完全线性。精度差但实现快。

**方案 C**：迭代式线性化。先固定 $\omega = $ 某个初值，解 MILP 得到 $\mathbf{x}^*$，再更新 $\omega$，迭代到收敛。这是 SCA 思路。

新手用方案 B，先跑通；论文版本用方案 A。

**要点 3：对数效用的分段线性化**

MILP 不能直接处理 $\log(\xi)$。两种做法：
- **分段线性近似**：把 $\xi \in [0,1]$ 划分为 $L$ 段，每段用一条切线/弦，引入辅助变量
- **SOS2 约束**：用 SOS2（一组变量中至多两个连续非零）建模分段线性函数

Gurobi 提供 `Model.setPWLObj()` 直接支持分段线性目标，最简洁。

### L2：滚动窗口分解（中等规模）

#### 文件位置
`src/leo_alloc/solvers/p2_rolling.py`

#### 适用场景
$K \in [20, 50]$。

#### 算法

```python
class P2RollingSolver:
    def __init__(self, scenario, surrogate_v, 
                 window: int = 5, step: int = 3):
        """
        window: 每次 MILP 求解的窗口长度
        step: 滑动步长（每次提交 step 个决策）
        """
    
    def solve(self):
        x_all = np.zeros((S, C, K))
        h_budget_remaining = self.scenario.H.copy()  # [C]
        x_prev = None  # 上一窗口的最后一个 x
        
        for win_start in range(0, K, self.step):
            win_end = min(win_start + self.window, K)
            
            # 子问题：在 [win_start, win_end) 上求 MILP
            # 切换预算约束改为：sum_{k in window} h[c,k] <= h_budget_remaining[c] * (window_len/remaining_K)
            
            sub_result = self._solve_subproblem(
                win_start, win_end, x_prev, h_budget_remaining)
            
            # 只提交前 step 个时隙的决策
            commit_end = min(win_start + self.step, K)
            x_all[:, :, win_start:commit_end] = sub_result.x[:, :, :commit_end-win_start]
            
            # 更新剩余预算
            h_used = sub_result.h[:, :commit_end-win_start].sum(axis=1)
            h_budget_remaining -= h_used
            
            x_prev = x_all[:, :, commit_end-1]
        
        return P2Result(...)
```

**关键设计**：预算分摊策略（前期 vs 后期分多少）需要消融实验确定。最简单的是按窗口长度比例分摊，更精细的可以按"预期需求方差"加权。

### L3：拉格朗日松弛（大规模启发式）

#### 文件位置
`src/leo_alloc/solvers/p2_lagrange.py`

#### 算法思路

对切换预算约束 (C4) 引入对偶变量 $\eta_c \ge 0$，松弛后：

$$
\max \sum_{k,c} \log(\epsilon + \xi_{c,k}) - \lambda_h \sum h_{c,k} - \sum_c \eta_c (\sum_k h_{c,k} - H_c)
$$

松弛后问题按 $k$ 解耦，每个 $k$ 是一个小型 MILP。然后用次梯度法更新 $\eta$。

**Rounding 策略**：松弛解可能违反整数性，用"按 $\hat{V}$ 降序贪心 + 局部 swap 修复可见性"做 rounding。

## 验证测试集

`tests/test_p2_milp.py`

```python
class TestP2BasicCorrectness:
    def test_single_cell_no_handover(self):
        """C=1 时无切换可言，验证 h 全为 0"""
        ...
    
    def test_handover_budget_strict(self):
        """H_c=0 时强制无切换，验证 x[:,c,k] 沿 k 不变"""
        ...
    
    def test_invisibility_reports_infeasible_when_budget_exhausted(self):
        """预算耗尽且前一卫星不可见时，应报告 infeasible 或 emergency，不得静默违反硬预算"""
        ...

class TestP2vsP1Consistency:
    def test_proxy_vs_groundtruth_calibration(self):
        """50 个随机 x_k 实例，比较 V̂(via P1) 与 Ṽ(代理)，gap 中位数 < 10%"""
        ...

class TestP2RollingVsMILP:
    @pytest.mark.parametrize("K", [10, 20])
    def test_rolling_gap_under_5pct(self, K):
        """L2 rolling 相对 L1 全窗口 MILP，gap < 5%"""
        ...
```

## 代理函数精度校准协议

`scripts/run_p2_proxy_calibration.py`：

1. 在 50 个随机场景上，对每个场景随机生成 100 个可行的 $\mathbf{x}_k$ 配置
2. 对每个 $\mathbf{x}_k$，分别计算 $\hat{V} = $ P1Solver.solve(x_k).xi 和 $\tilde{V} = $ 代理函数
3. 报告 $|\hat{V} - \tilde{V}| / \hat{V}$ 的均值、中位数、95 分位数
4. 在 jupyter notebook 中绘制散点图（x 轴 $\hat{V}$，y 轴 $\tilde{V}$），目视检验偏差模式
5. 如果偏差有系统性模式（如负载高时 $\tilde{V}$ 高估），考虑加入校正项

**论文中 §4.2 必须报告这个校准结果**，否则代理函数的合理性受质疑。

## BC 训练数据生成

`scripts/generate_bc_data.py` 是问题三的依赖：

```python
N_demo = 10000  # 示范轨迹数
out_dir = 'data/demonstrations/'

for i in range(N_demo):
    seed = i
    scenario = generate_scenario(seed=seed, scale='medium')
    p2_solver = P2RollingSolver(scenario, ...)
    result = p2_solver.solve()
    
    # 保存 (s_k, x_k*) 序列，注意 s_k 只能用 k 时刻可观测的信息
    save_demonstration(
        path=f"{out_dir}/traj_{i:05d}.npz",
        scenario=scenario,
        x_optimal=result.x,
        # 不能保存真实未来 demand，会信息泄漏到 BC
    )
```

**关键**：BC 数据生成是 batch process，可以离线跑数天数周。优先级低于 P2 算法本身，但必须在 P3 BC 训练前完成。

## 完成判定

1. ✅ `tests/test_p2_milp.py` 全部通过
2. ✅ L1 在 K=10, C=20 规模下 1 小时内出解
3. ✅ L2 在 K=30, C=15 规模下 10 分钟内出解
4. ✅ 代理函数 vs ground truth 的 gap 中位数 ≤ 10%
5. ✅ 可成功生成 100 条 BC 训练轨迹作为验证
6. ✅ 论文 §4 全部主结果图可生成

**未达到全部 6 条，不进入 P3 阶段。**
