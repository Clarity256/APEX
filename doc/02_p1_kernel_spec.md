# 02. P1 Kernel Specification (Most Detailed)

> **这是首个要实现的模块，整个项目的基石。**
> 阅读完本文档后开始实现：先 L1（CVXPY 基准），后 L2（对偶分解）。

## 模块职责

**单一职责**：给定固定的卫星-小区归属 $\mathbf{x}$ 和快时隙业务到达序列 $\mathbf{a}$，求解 PRB 与功率的联合最优分配，最大化对数公平效用。

**对外提供**：可被问题二（代理函数 ground truth）、问题三（RL 低层模块、reward 计算）反复调用的 Python API。

**性能要求**：
- L1（CVXPY）：单次 < 2s（medium 规模）
- L2（对偶分解）：单次 < 100ms（medium 规模）

## 数学问题（必读）

### 决策变量

| 变量 | 形状 | 含义 |
|---|---|---|
| $n_{s,c,m}$ | `[S, C, M]`, $\ge 0$ | 卫星 s 在快时隙 m 分给小区 c 的等效 PRB 数 |
| $p_{s,c,m}$ | `[S, C, M]`, $\ge 0$ | 总发射功率（W） |
| $z_{c,m}$ | `[C, M]`, $\ge 0$ | 实际兑现服务量（bit） |
| $\xi_c$ | `[C]`, $\in [0,1]$ | 需求满足率 |

### 完整优化问题

$$
\begin{aligned}
\text{(P1)}\quad \max_{n,p,z,\xi}\quad & \sum_{c} \log(\epsilon + \xi_c) \\
\text{s.t.}\quad
& \sum_c n_{s,c,m} \le N_s^{\text{PRB}}, \quad \forall s, m \quad \text{(C1a)} \\
& \sum_c p_{s,c,m} \le P_s^{\max}, \quad \forall s, m \quad \text{(C1b)} \\
& 0 \le n_{s,c,m} \le N_s^{\text{PRB}} \cdot x_{s,c}, \quad \forall s,c,m \quad \text{(C2a)} \\
& 0 \le p_{s,c,m} \le P_s^{\max} \cdot x_{s,c}, \quad \forall s,c,m \quad \text{(C2b)} \\
& 0 \le z_{c,m} \le \sum_s r_{s,c,m}, \quad \forall c, m \quad \text{(C3)} \\
& \sum_{\tau \le m} z_{c,\tau} \le \sum_{\tau \le m} a_{c,\tau}, \quad \forall c, m \quad \text{(C4)} \\
& D_c \xi_c \le \sum_m z_{c,m}, \quad \forall c \quad \text{(C5a)} \\
& 0 \le \xi_c \le 1, \quad \forall c \quad \text{(C5b)}
\end{aligned}
$$

其中速率函数（**关键非线性项**）：
$$
r_{s,c,m} = T_f W_{\text{PRB}} \cdot n_{s,c,m} \log_2\left(1 + \frac{p_{s,c,m} g_{s,c}}{W_{\text{PRB}} n_{s,c,m} N_0}\right)
$$

需求总量定义为 $D_c = \sum_m a_{c,m}$。当 $D_c = 0$ 时，工程实现必须把该小区从对数公平项中剔除，或令 $\xi_c = 1$ 并记录 `zero_demand=True`，避免 (C5a) 退化导致无意义的满足率。

### 凸化关键

令 $\alpha_{s,c} = g_{s,c} / (W_{\text{PRB}} N_0)$，则核心项 $n \log_2(1 + \alpha p / n)$ 是 **perspective function**，关于 $(n, p) \ge 0$ **联合凹**。

**CVXPY 表达技巧**：
```python
# 数学：n * log(1 + α*p/n) = -n * log(n / (n + α*p)) = -rel_entr(n, n + α*p)
# CVXPY: scale = T_f * W_PRB / log(2)
r = scale * (-cp.rel_entr(n, n + alpha * p))   # 这是 concave 表达式
```

CVXPY 会通过 DCP 规则识别 `-rel_entr(·, ·)` 是 concave 的，约束 `z <= r` 是合法的凸约束（"线性 ≤ 凹"）。

## 实现 L1：CVXPY + MOSEK 基准求解器

### 文件位置
`src/leo_alloc/solvers/p1_cvx.py`

### 接口

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class P1Result:
    n: np.ndarray            # [S, C, M]
    p: np.ndarray            # [S, C, M]
    z: np.ndarray            # [C, M]
    xi: np.ndarray           # [C]
    U: float
    status: str
    solve_time: float

class P1CVXSolver:
    def __init__(self, S: int, C: int, M: int, sys_params: dict,
                 solver: str = 'MOSEK', use_dpp: bool = True):
        """
        sys_params 须包含：N_PRB[S], P_max[S], W_PRB, N0, T_f, eps
        use_dpp=True 启用参数化复用（推荐用于反复调用）
        """
    
    def solve(self, x: np.ndarray, a: np.ndarray, g: np.ndarray) -> P1Result:
        """
        x : [S, C] 二值归属（np.float64 或 np.int8）
        a : [C, M] 快时隙业务到达
        g : [S, C] 大尺度信道增益（线性，非 dB）
        """
```

### 关键实现要点（必读）

#### 要点 1：DPP 参数化（避免重复编译）

错误做法（每次 solve 重新构造问题）：
```python
def solve(self, x, a, g):
    n = cp.Variable(...)
    # 构造所有约束 ...
    prob = cp.Problem(...)
    prob.solve()                  # ← 99% 时间花在编译
```

正确做法（构造一次，反复 update value）：
```python
def __init__(self, ...):
    # 变量
    self.n = cp.Variable((S, C, M), nonneg=True)
    self.p = cp.Variable((S, C, M), nonneg=True)
    self.z = cp.Variable((C, M), nonneg=True)
    self.xi = cp.Variable(C, nonneg=True)
    
    # 参数（运行时更新值）
    self.x_param = cp.Parameter((S, C), nonneg=True)
    self.a_param = cp.Parameter((C, M), nonneg=True)
    self.alpha_param = cp.Parameter((S, C), nonneg=True)  # α = g/(W·N0)
    
    # 构造约束（一次性）
    constraints = self._build_constraints()
    objective = cp.Maximize(cp.sum(cp.log(self.eps + self.xi)))
    self.prob = cp.Problem(objective, constraints)

def solve(self, x, a, g):
    self.x_param.value = x.astype(np.float64)
    self.a_param.value = a
    self.alpha_param.value = g / (self.W_PRB * self.N0)
    self.prob.solve(solver='MOSEK', warm_start=True)
    return self._extract_result()
```

DPP 要求：所有 Parameter 只能在 affine 位置出现（不能被 rel_entr 直接包含）。所以 $\alpha p$ 这种结构在 DPP 下需要小心处理，可能需要重新参数化。**如果 DPP 调试困难，可以先做非 DPP 版本，性能不达标再优化。**

#### 要点 2：数值尺度归一

业务量 $D_c$ 通常在 $10^7 - 10^9$ bit 量级，会导致 solver 数值病态。

**做法**：在 `__init__` 中定义 `self.bit_scale = 1e6`，所有 bit 类量都除以该值后再喂给 solver：

```python
def solve(self, x, a, g):
    a_scaled = a / self.bit_scale
    # solve with a_scaled
    # 输出时把 z 乘回去
    z_real = self.z.value * self.bit_scale
```

#### 要点 3：x=0 时的边界处理

当 $x_{s,c} = 0$，希望 $n_{s,c,:} = p_{s,c,:} = 0$。处理方式：

**方式 A（推荐）**：在约束中强制：
```python
# 对所有 x[s,c]=0 的位置，n 和 p 被严格约束为 0
constraints.append(n <= N_PRB_tensor * x_param_broadcast)
constraints.append(p <= P_max_tensor * x_param_broadcast)
# 当 x=0 时上界为 0，结合 nonneg=True 强制为 0
```

**方式 B**：构造 (s, c) 索引列表，只对 x=1 的 pair 写 rel_entr 项。代码更繁琐但避免 rel_entr(0, 0) 的潜在数值问题。

新手用方式 A 即可。

#### 要点 4：r_total 的高效构造

不要用三重 for 循环：
```python
# BAD：太慢
for c in range(C):
    for m in range(M):
        r_total = 0
        for s in range(S):
            r_total += scale * (-cp.rel_entr(n[s,c,m], n[s,c,m] + alpha[s,c] * p[s,c,m]))
```

用向量化：
```python
# GOOD：向量化
# alpha shape: [S, C], broadcast to [S, C, M]
alpha_bc = alpha[:, :, None]  # [S, C, 1]
r_per_term = scale * (-cp.rel_entr(n, n + cp.multiply(alpha_bc, p)))  # [S, C, M]
r_total = cp.sum(r_per_term, axis=0)  # [C, M]
constraints.append(z <= r_total)
```

注意 `cp.multiply` 用于 elementwise，`cp.rel_entr` 支持 vectorized inputs。

## 实现 L2：对偶分解快速求解器

### 文件位置
`src/leo_alloc/solvers/p1_dual.py`

### 算法原理

对资源池约束引入对偶变量 $\lambda_{s,m}, \mu_{s,m}$，对满足率约束引入 $\nu_c$。固定对偶变量时，部分拉格朗日按 $(s,c,m)$ 三元组**完全解耦**，每个三元组求解单调凸优化（一维水填结构）。

**外层**：对偶变量的次梯度更新（带衰减步长）  
**内层**：每个 $(s,c,m)$ 三元组的局部水填求解（纯 NumPy）

### 接口

```python
class P1DualSolver(P1Solver):
    def __init__(self, S, C, M, sys_params, 
                 tol: float = 1e-3, max_iter: int = 200,
                 step_init: float = 1.0):
        ...
    
    def solve(self, x, a, g) -> P1Result:
        ...
```

### 详细伪代码

```python
def solve(self, x, a, g):
    # 初始化
    lam = np.zeros((self.S, self.M))   # PRB 影子价格
    mu  = np.zeros((self.S, self.M))   # 功率影子价格
    nu  = np.ones(self.C) * 0.5         # 满足率对偶
    
    alpha = g / (self.W_PRB * self.N0)  # [S, C]
    
    for it in range(self.max_iter):
        # === 原变量更新（按 (s,c,m) 解耦） ===
        n, p = self._water_filling(lam, mu, nu, x, alpha)  # NumPy 向量化
        
        # === ξ 更新（闭式解）===
        xi = self._update_xi(nu, ...)
        
        # === 计算约束违反 ===
        n_use = n.sum(axis=1)              # [S, M]
        p_use = p.sum(axis=1)              # [S, M]
        n_violate = n_use - self.N_PRB[:, None]
        p_violate = p_use - self.P_max[:, None]
        
        # === 对偶次梯度更新（投影到非负）===
        step = self.step_init / np.sqrt(it + 1)
        lam = np.maximum(lam + step * n_violate, 0)
        mu  = np.maximum(mu  + step * p_violate, 0)
        # nu 类似处理
        
        # === 收敛判据 ===
        primal_obj = np.sum(np.log(self.eps + xi))
        dual_gap = self._compute_gap(...)
        if dual_gap < self.tol:
            break
    
    # 用最终 (n, p) 通过 (C4) 因果性约束反推 z
    z = self._compute_z_from_np(n, p, a, g)
    
    return P1Result(n=n, p=p, z=z, xi=xi, U=primal_obj, ...)
```

### 关键引理（论文附录）

**引理 1（perspective 凹性）**：$\phi(n, p) = n \log(1 + \alpha p / n)$ 在 $\mathbb{R}_+^2$ 上联合凹。

**引理 2（Slater 条件）**：当 $\sum_s x_{s,c} = 1$ 且 $N_s^{\text{PRB}}, P_s^{\max} > 0$ 时存在严格内点，强对偶成立。

**引理 3（次梯度收敛率）**：对偶 gap 以 $O(1/\sqrt{K})$ 速率收敛。

> 这三个引理的证明放论文附录，但实现 L2 时必须先在草稿纸上推完，否则代码写不对。

## 验证测试集（pytest）

### 文件位置
`tests/test_p1_correctness.py`

### Test Suite 设计

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sys_params_default():
    return {
        'N_PRB': np.array([100.0]),     # 1 satellite, 100 PRBs
        'P_max': np.array([100.0]),     # 100 W
        'W_PRB': 180e3,                  # 180 kHz
        'N0': 1e-15,                     # noise PSD
        'T_f': 0.01,                     # 10 ms
        'eps': 1e-4,
    }
```

```python
# tests/test_p1_correctness.py

class TestP1AnalyticalCases:
    """有解析解的简单场景"""
    
    def test_single_sat_single_cell_single_slot(self, sys_params_default):
        """S=C=M=1：所有资源给唯一小区，闭式可验证"""
        solver = P1CVXSolver(S=1, C=1, M=1, sys_params=sys_params_default)
        x = np.array([[1.0]])
        a = np.array([[1e7]])  # 10 Mbit
        g = np.array([[1e-12]])
        
        result = solver.solve(x, a, g)
        
        # 验证最优解
        assert result.status == 'optimal'
        # n* = N_PRB（全分配给唯一小区）
        np.testing.assert_allclose(result.n[0,0,0], sys_params_default['N_PRB'][0], rtol=1e-3)
        # p* = P_max
        np.testing.assert_allclose(result.p[0,0,0], sys_params_default['P_max'][0], rtol=1e-3)
        # 解析计算 r
        n, p, g_val = result.n[0,0,0], result.p[0,0,0], g[0,0]
        r_analytical = (
            sys_params_default['T_f'] * sys_params_default['W_PRB'] * 
            n * np.log2(1 + p * g_val / (sys_params_default['W_PRB'] * n * sys_params_default['N0']))
        )
        # ξ = min(r/D, 1)
        xi_expected = min(r_analytical / a[0,0], 1.0)
        np.testing.assert_allclose(result.xi[0], xi_expected, rtol=1e-2)
    
    def test_symmetric_two_cells(self, sys_params_default):
        """两个完全相同的小区，最优解应该完全对称"""
        ...
        # 验证 n[0,0,:] ≈ n[0,1,:]，p[0,0,:] ≈ p[0,1,:]


class TestP1FairnessProperty:
    """验证公平性确实在生效"""
    
    def test_log_fairness_helps_weak_cell(self):
        """强信道+低需求 vs 弱信道+高需求，log 公平应让弱小区拿到更多功率"""
        ...
    
    def test_jain_index_monotonicity(self):
        """加入更多公平偏向时 Jain 指数不应下降"""
        ...


class TestP1NumericalStability:
    """数值稳定性"""
    
    def test_extreme_channel_ratio(self):
        """g 跨度 1e3 时不爆"""
        ...
    
    def test_large_demand_scale(self):
        """D 在 1e9 量级时不出现数值警告"""
        ...


class TestL1vsL2Consistency:
    """L2 输出与 L1 一致性"""
    
    @pytest.mark.parametrize("scenario_size", ["toy", "medium"])
    def test_gap_within_5_percent(self, scenario_size):
        """对随机 50 个 instance，L2 相对 L1 的 gap ≤ 5%"""
        ...
    
    def test_l2_faster_than_l1(self):
        """L2 求解时间应至少快 10 倍"""
        ...
```

**测试通过门槛**：
- L1 全部 analytical case 误差 ≤ 0.5%
- L1 数值稳定性测试无 warning
- L2 vs L1：gap 中位数 ≤ 3%，95 分位数 ≤ 5%
- L2 加速比 ≥ 10×（medium 规模）

## 文档与示例

### docstrings
每个 public 函数必须有 docstring，遵循 numpy 格式：

```python
def solve(self, x: np.ndarray, a: np.ndarray, g: np.ndarray) -> P1Result:
    """
    Solve the fast-slot PRB-power allocation problem.
    
    Parameters
    ----------
    x : ndarray of shape (S, C)
        Binary association matrix from the slow-slot decision.
    a : ndarray of shape (C, M)
        Per-fast-slot demand arrivals in bits.
    g : ndarray of shape (S, C)
        Large-scale channel gain (linear, not dB).
    
    Returns
    -------
    P1Result
        Container with n, p, z, xi, U, status, solve_time.
    
    Notes
    -----
    Uses the perspective function convexification:
        n * log(1 + α*p/n) = -rel_entr(n, n + α*p)
    where α = g / (W_PRB * N0).
    """
```

### 示例脚本
`scripts/run_p1_experiments.py` 提供完整的"读取 scenario → 调用 P1Solver → 保存结果 → 画图"流程，作为后续问题二/三的 reference。

## 开发完成判定

P1 模块完成的硬指标：

1. ✅ `tests/test_p1_correctness.py` 全部通过
2. ✅ L1 单次求解 < 2s（medium: S=4, C=10, M=20）
3. ✅ L2 单次求解 < 100ms（medium 规模）
4. ✅ L2 vs L1 gap 95 分位数 ≤ 5%
5. ✅ 可被 import 并通过 5 行代码完成一次完整求解
6. ✅ docstring 覆盖率 100%（public API）
7. ✅ 一份 jupyter notebook 演示完整流程并可重复运行

**未达到全部 7 条，不进入 P2 阶段。**
