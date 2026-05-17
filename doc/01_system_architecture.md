# 01. System Architecture

## 项目目录结构

```
leo-resource-alloc/
├── README.md
├── pyproject.toml              依赖与项目元信息
├── requirements.txt            pip 兼容依赖列表
├── .gitignore
│
├── src/leo_alloc/              主代码包
│   ├── __init__.py
│   ├── scenario/               场景与数据生成
│   │   ├── __init__.py
│   │   ├── orbit.py            Skyfield + TLE 轨道生成
│   │   ├── channel.py          路径损耗 + 阴影衰落
│   │   ├── demand.py           需求过程生成（泊松、周期）
│   │   └── visibility.py       可见性矩阵计算
│   │
│   ├── solvers/                求解器层
│   │   ├── __init__.py
│   │   ├── p1_cvx.py           P1-cvx L1 求解器（CVXPY+MOSEK）
│   │   ├── p1_dual.py          P1 L2 对偶分解求解器
│   │   ├── p2_milp.py          P2-MILP L1 全窗口求解器
│   │   ├── p2_rolling.py       P2 L2 滚动窗口分解
│   │   └── p2_lagrange.py      P2 L3 拉格朗日松弛
│   │
│   ├── surrogate/              代理函数模块
│   │   ├── __init__.py
│   │   └── v_proxy.py          容量代理函数 V̂
│   │
│   ├── rl/                     强化学习
│   │   ├── __init__.py
│   │   ├── env.py              gym-style 环境
│   │   ├── policy_mlp.py       MLP 策略网络
│   │   ├── policy_transformer.py  Transformer 策略网络（可选）
│   │   ├── masking.py          Action masking 工具
│   │   ├── predictor.py        需求预测模块（P0/P1/P2 三档）
│   │   ├── bc_trainer.py       BC 预训练器
│   │   └── ppo_trainer.py      PPO 微调器
│   │
│   ├── baselines/              对比基线
│   │   ├── __init__.py
│   │   ├── greedy.py           B0 贪心
│   │   ├── oracle.py           B1 Oracle-P2
│   │   ├── mpc.py              B2 MPC 滚动优化
│   │   └── pure_ppo.py         B3 纯 PPO 无分层
│   │
│   ├── evaluation/             评估与指标
│   │   ├── __init__.py
│   │   ├── metrics.py          Jain 指数、Gap、切换统计等
│   │   └── runner.py           实验批量执行器
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py               结果序列化、加载
│       ├── config.py           dataclass 形式的配置
│       ├── logging.py          日志工具
│       └── numerics.py         数值稳定性工具（scale, clip）
│
├── tests/                      pytest 测试
│   ├── test_p1_correctness.py  P1 求解正确性（金标准）
│   ├── test_p1_kkt.py          KKT 条件验证
│   ├── test_p2_milp.py         P2 求解正确性
│   ├── test_rl_env.py          RL 环境正确性
│   ├── test_masking.py         Action masking 正确性
│   └── conftest.py             pytest fixtures（场景生成器）
│
├── scripts/                    可执行脚本
│   ├── run_p1_experiments.py
│   ├── run_p2_experiments.py
│   ├── train_bc.py
│   ├── train_ppo.py
│   └── evaluate_all.py
│
├── configs/                    实验配置（YAML 或 Python dataclass）
│   ├── system_params.yaml      系统参数（载频、带宽等）
│   ├── scenario_toy.yaml       toy 场景
│   ├── scenario_medium.yaml    medium 场景
│   └── scenario_stress.yaml    stress 场景
│
├── data/                       数据
│   ├── tle/                    Starlink TLE 文件
│   └── demonstrations/         BC 训练用的离线最优轨迹
│
├── results/                    实验结果
│   ├── p1_runs/
│   ├── p2_runs/
│   └── p3_runs/
│
└── notebooks/                  Jupyter 探索性分析
    ├── 01_p1_sanity_check.ipynb
    ├── 02_p2_proxy_calibration.ipynb
    └── 03_p3_training_diagnostics.ipynb
```

## 模块依赖图

```
                     ┌─────────────────┐
                     │  scenario/      │
                     │  orbit, channel │
                     │  demand, visib. │
                     └────────┬────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
      ┌──────────────┐                ┌──────────────┐
      │ solvers/     │                │ rl/env.py    │
      │ p1_cvx       │                │              │
      │ p1_dual      │                │              │
      └──────┬───────┘                └──────┬───────┘
             │                               │
             │  作为子模块被调用              │
             ▼                               ▼
      ┌──────────────┐                ┌──────────────┐
      │ surrogate/   │                │ rl/policy_*  │
      │ v_proxy      │                │ rl/masking   │
      └──────┬───────┘                │ rl/predictor │
             │                        └──────┬───────┘
             ▼                               │
      ┌──────────────┐                       │
      │ solvers/     │ ──── 提供 BC 数据 ───►│
      │ p2_milp      │                       │
      │ p2_rolling   │                       ▼
      └──────────────┘                ┌──────────────┐
                                      │ rl/bc_trainer│
                                      │ rl/ppo_train.│
                                      └──────┬───────┘
                                             │
              ┌──────────────────────────────┤
              ▼                              ▼
        ┌──────────────┐              ┌──────────────┐
        │ baselines/   │              │ evaluation/  │
        └──────────────┘              └──────────────┘
```

**关键依赖原则**：
- `scenario/` 是叶子节点（无内部依赖），所有模块依赖它
- `solvers/p1_*` 是 `surrogate/`, `solvers/p2_*`, `rl/env` 的依赖
- `solvers/p2_*` 是 `rl/bc_trainer` 的依赖（用于生成示范数据）
- 上层模块（rl, evaluation）不应反向依赖下层模块

## 关键接口契约

以下接口一旦写出，**任何修改都需要 review**，因为下游代码都依赖它们。

### Scenario 接口

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ScenarioInstance:
    """单个场景实例（一次完整的仿真窗口）"""
    # 系统维度
    S: int                   # 卫星数
    C: int                   # 小区数
    K: int                   # 慢时隙数
    M: int                   # 每个慢时隙内的快时隙数
    
    # 时变张量
    g: np.ndarray            # 信道增益, shape [S, C, K]
    v: np.ndarray            # 可见性, shape [S, C, K], 0/1
    a: np.ndarray            # 业务到达, shape [C, K, M]
    
    # 系统常量
    N_PRB: np.ndarray        # 每卫星 PRB 池, shape [S]
    P_max: np.ndarray        # 每卫星功率上限, shape [S]
    H: np.ndarray            # 每小区切换预算, shape [C]
    
    # 系统物理常量
    W_PRB: float             # 单 PRB 带宽 (Hz)
    N0: float                # 噪声功率谱密度 (W/Hz)
    T_f: float               # 快时隙时长 (s)
    eps: float               # 对数效用极小正数
    lambda_h: float          # 切换软惩罚系数
    
    # 元信息
    seed: int                # 随机种子（可重复）
    scenario_id: str         # 场景标识
```

### P1 Solver 接口

```python
@dataclass
class P1Result:
    n: np.ndarray            # [S, C, M] PRB 分配
    p: np.ndarray            # [S, C, M] 功率分配
    z: np.ndarray            # [C, M] 实际兑现服务量
    xi: np.ndarray           # [C] 需求满足率
    U: float                 # 目标函数值
    status: str              # solver 状态
    solve_time: float        # 求解耗时（秒）

class P1Solver(ABC):
    """P1-cvx 求解器抽象基类"""
    def __init__(self, S: int, C: int, M: int, sys_params: dict): ...
    
    @abstractmethod
    def solve(self, x: np.ndarray, a: np.ndarray, g: np.ndarray) -> P1Result:
        """
        x: [S, C] 二值归属矩阵
        a: [C, M] 快时隙到达序列
        g: [S, C] 大尺度信道增益（当前慢时隙）
        """
```

### P2 Solver 接口

```python
@dataclass
class P2Result:
    x: np.ndarray            # [S, C, K] 二值归属
    h: np.ndarray            # [C, K] 切换指示
    xi: np.ndarray           # [C, K] 满足率
    U: float                 # 目标函数值
    handover_per_cell: np.ndarray  # [C]
    solve_time: float

class P2Solver(ABC):
    def __init__(self, scenario: ScenarioInstance, p1_solver: P1Solver): ...
    
    @abstractmethod
    def solve(self) -> P2Result: ...
```

### RL Environment 接口

```python
class LEOSchedulingEnv(gym.Env):
    """gym-style 环境，与 stable-baselines3 / cleanrl 兼容"""
    
    observation_space: gym.spaces.Dict   # 包含 channel, visibility, demand_pred, ...
    action_space: gym.spaces.MultiDiscrete  # 按小区分解
    
    def reset(self, scenario: ScenarioInstance) -> Tuple[Dict, Dict]: ...
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]: ...
    def get_action_mask(self) -> np.ndarray: ...  # [C, max_S_cand]
```

## 数据流总览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TLE 文件        │───►│ orbit.py        │───►│ visibility.py   │
│ system_params   │    │                 │    │ channel.py      │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │ demand.py       │  泊松+周期需求生成
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ ScenarioInstance│  完整场景对象（不可变）
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
    P1 跑批          P2 跑批         RL 环境
    （评估）         （生成 BC 数据） （训练）
       │               │               │
       └───────────────┴───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ metrics.py      │
              │ Jain, Gap, ...  │
              └─────────────────┘
```

## 技术栈选型（冻结）

| 用途 | 选择 | 理由 |
|---|---|---|
| 凸优化建模 | CVXPY 1.5+ | DCP + DPP + 多 solver 后端 |
| 凸优化 solver | MOSEK 10（学术 license）或 ECOS 备用 | 处理 perspective 函数最稳 |
| MILP | Gurobi 11（学术 license） | 远优于开源 MILP solver |
| RL 框架 | stable-baselines3 2.x 或 cleanrl | 成熟 + 可定制 |
| 神经网络 | PyTorch 2.x | MPS 后端支持 Mac |
| 轨道仿真 | Skyfield 1.45+ | TLE 解析准、API 干净 |
| 配置管理 | dataclasses + YAML | 类型安全 + 易读 |
| 实验追踪 | wandb 或 tensorboard | 二选一，保持一致 |
| 测试框架 | pytest | 标准 |
| Python 版本 | 3.10+ | 充分使用 type hints |

**版本锁定原则**：`pyproject.toml` 中所有依赖必须 pin minor 版本（如 `cvxpy>=1.5,<1.6`），避免 silent 升级破坏复现。
