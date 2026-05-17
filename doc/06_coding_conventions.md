# 06. Coding Conventions

> 这份文件是写给 Claude Code / Codex 的硬约束。任何代码不符合本文档要求，必须重写。

## Python 版本与依赖

- Python 3.10+
- 严格 type hints（所有 public 函数）
- 使用 `pyproject.toml`，不用 `setup.py`
- 依赖 minor 版本 pin（如 `cvxpy>=1.5,<1.6`）

## 项目根目录的 `CLAUDE.md`（必备）

在项目根目录创建 `CLAUDE.md`，内容是给 Claude Code 的 standing instruction：

```markdown
# Claude Code Instructions for this Project

## You must always:
1. Read `doc/06_coding_conventions.md` before any code change
2. Write tests BEFORE implementation when working on solver modules
3. Use existing types from `src/leo_alloc/utils/config.py`; do NOT redefine them
4. Variable names follow `doc/00_research_context.md` 术语表
5. Each function ≤ 50 lines; split if longer
6. Each module file ≤ 500 lines; split if longer

## You must never:
1. Modify `ScenarioInstance` dataclass without explicit user approval
2. Create new files outside the directory structure in `doc/01_system_architecture.md`
3. Use `print()` for logging; use `logger` from `utils/logging.py`
4. Catch generic `Exception`; catch specific types only
5. Use `np.random` directly; use `rng = np.random.default_rng(seed)` pattern
6. Make architectural decisions; ask the user instead

## When unsure:
Ask. Do not guess.
```

把这个 `CLAUDE.md` 放在项目根目录，Claude Code 启动时会自动加载。

## 命名规范

### 变量命名
严格遵循 `00_research_context.md` 中的术语表。**禁止**自创同义词。

正确：
```python
def solve_p1(x: np.ndarray, a: np.ndarray, g: np.ndarray) -> P1Result:
    """x: association, a: demand_arrival, g: channel_gain"""
```

错误：
```python
def solve_p1(assignment, traffic, gain):  # ← 用了禁词
```

### 文件命名
- 模块用 snake_case：`p1_cvx.py`, `bc_trainer.py`
- 测试文件：`test_<module>_<feature>.py`
- 脚本：动词开头，`run_p1_experiments.py`, `train_bc.py`
- 配置：`scenario_<scale>.yaml`

### 类命名
- PascalCase
- 接口类后缀 `Solver`, `Trainer`, `Predictor`
- 数据类（dataclass）后缀 `Result`, `Instance`, `Config`

```python
class P1CVXSolver: ...
class P1Result: ...
class ScenarioInstance: ...
class BCTrainer: ...
```

## Type Hints

**所有 public 函数必须有完整 type hints**：

```python
from typing import Optional
import numpy as np
from numpy.typing import NDArray

def solve_p1(
    x: NDArray[np.float64],         # shape: (S, C)
    a: NDArray[np.float64],         # shape: (C, M)
    g: NDArray[np.float64],         # shape: (S, C)
    sys_params: dict[str, float],
    solver: str = 'MOSEK',
    timeout: Optional[float] = None,
) -> P1Result:
    ...
```

**形状放在 inline 注释里**——NDArray 不支持 shape 类型，但注释是写给 AI 看的"软类型"。

## Docstring 规范

使用 NumPy 风格：

```python
def compute_jain_index(xi: NDArray[np.float64]) -> float:
    """
    Compute Jain's fairness index over satisfaction rates.
    
    Parameters
    ----------
    xi : ndarray of shape (C,) or (C, K)
        Satisfaction rates per cell (optionally per slot).
        Values in [0, 1].
    
    Returns
    -------
    float
        Jain index in (0, 1]. 1 means perfect fairness.
    
    Notes
    -----
    Computed as (sum(xi))^2 / (n * sum(xi^2)).
    For 2D input, averages across slots.
    
    Examples
    --------
    >>> compute_jain_index(np.array([0.5, 0.5, 0.5]))
    1.0
    >>> compute_jain_index(np.array([1.0, 0.0, 0.0]))
    0.333...
    """
```

## 错误处理

### 不要 catch generic Exception
```python
# BAD
try:
    result = solver.solve(...)
except Exception as e:
    return None

# GOOD
try:
    result = solver.solve(...)
except cp.error.SolverError as e:
    logger.error(f"CVXPY solver failed: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
```

### Solver 失败的处理
```python
def solve(self, x, a, g) -> P1Result:
    ...
    self.prob.solve(...)
    
    # 关键：检查求解状态
    if self.prob.status not in ['optimal', 'optimal_inaccurate']:
        raise SolverError(
            f"P1Solver failed with status {self.prob.status}. "
            f"Input shapes: x={x.shape}, a={a.shape}, g={g.shape}"
        )
    
    return P1Result(...)
```

### 数值断言
对关键不变量做断言：

```python
def solve(self, x, a, g) -> P1Result:
    # 输入校验
    assert x.shape == (self.S, self.C), f"Expected x shape {(self.S, self.C)}, got {x.shape}"
    assert np.all((x == 0) | (x == 1)), "x must be binary"
    assert np.all(x.sum(axis=0) == 1), "Each cell must have exactly one satellite"
    assert np.all(g >= 0), "Channel gains must be non-negative"
    ...
```

## 随机性管理

**禁止**：
```python
np.random.seed(42)            # global state，会污染其他模块
x = np.random.randn(10)       # 来自 global state
```

**使用**：
```python
def generate_scenario(seed: int) -> ScenarioInstance:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(10)
    ...
```

每个函数明确接受 `seed` 或 `rng` 参数，**绝不依赖 global 状态**。

## 日志

```python
# src/leo_alloc/utils/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

使用：
```python
from leo_alloc.utils.logging import get_logger
logger = get_logger(__name__)

logger.info(f"P1 solver started, problem size: S={S}, C={C}, M={M}")
logger.warning(f"Solver returned status {status}, retrying with ECOS")
```

**禁止 print()**。

## 单元测试

### Pytest 结构

```python
# tests/conftest.py — 共享 fixture
import pytest
import numpy as np

@pytest.fixture
def default_sys_params():
    return {...}

@pytest.fixture
def small_scenario(default_sys_params):
    rng = np.random.default_rng(42)
    return generate_scenario(...)

# tests/test_p1.py
class TestP1Correctness:
    def test_analytical_solution(self, default_sys_params):
        ...
    
    @pytest.mark.parametrize("scale", ["toy", "medium"])
    def test_l2_gap(self, scale):
        ...

class TestP1NumericalStability:
    @pytest.mark.slow
    def test_extreme_channel_ratio(self):
        ...
```

### 测试覆盖率目标
- 求解器模块：>= 90%
- 工具模块：>= 80%
- RL 训练代码：>= 70%（训练循环测试较难）

### 测试命名
- `test_<feature>_<condition>_<expected>`
- 例：`test_p1_extreme_demand_returns_optimal`, `test_masking_zero_budget_forbids_handover`

## 配置管理

使用 dataclass + YAML：

```python
# src/leo_alloc/utils/config.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SystemParams:
    freq_Hz: float = 2e9
    bandwidth_Hz: float = 30e6
    W_PRB_Hz: float = 180e3
    N0_W_per_Hz: float = 4e-21
    P_sat_max_W: float = 1000.0
    G_tx_dBi: float = 30.0
    G_rx_dBi: float = 0.0
    T_fast_slot_s: float = 0.01
    eps: float = 1e-4

@dataclass(frozen=True)
class ScenarioConfig:
    S: int
    C: int
    K: int
    M: int
    H_per_cell: int
    demand_base_high_mbps: float = 50.0
    demand_base_low_mbps: float = 10.0
    mismatch_ratio: float = 0.3
    seed: int = 0
```

YAML 文件可加载：
```python
import yaml
with open('configs/scenario_medium.yaml') as f:
    cfg = ScenarioConfig(**yaml.safe_load(f))
```

## Git 工作流

### 提交粒度
- 一个 commit 解决一个具体问题
- Commit message 用动词开头：`add P1Solver class`, `fix shape mismatch in masking`
- 大 feature 用 feature branch，PR 合并

### Pre-commit hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
```

每次 commit 前自动跑 ruff（格式化+linter）和 mypy（类型检查）。

## 数值约定

| 量 | 单位 | 内部表示 |
|---|---|---|
| 时间 | 秒 | float |
| 频率 | Hz | float |
| 功率 | W（线性，非 dB） | float |
| 信道增益 | 线性，非 dB | float |
| 业务量 | bit | float（可被 scale 归一） |
| 数据率 | bit/s | float |
| 索引 | 整数从 0 起 | int |
| 切换次数 | 整数 | int 或 np.int32 |

**禁止**在代码中混用 dB 和线性。dB 只在打印日志和最终图标时使用，通过工具函数：
```python
def linear_to_dB(x: float) -> float: ...
def dB_to_linear(x: float) -> float: ...
```

## 性能 profiling

任何"重头"代码（solver, RL 训练）必须能用 `cProfile` 跑：
```python
import cProfile
profiler = cProfile.Profile()
profiler.enable()
result = solver.solve(...)
profiler.disable()
profiler.dump_stats('p1_solve.prof')
# 用 snakeviz p1_solve.prof 查看
```

## 文档生成

- README.md 必须包含"快速开始"，包含 5 行能跑通的示例
- 关键模块有独立 `<module>/README.md` 解释设计思路
- 自动生成 API docs（用 mkdocs + mkdocstrings）
