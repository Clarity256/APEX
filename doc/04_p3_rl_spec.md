# 04. P3 RL Specification

> 实现前置条件：P1 和 P2 模块全部测试通过，BC 训练数据已生成（至少 5000 条轨迹）。

## 模块职责

**单一职责**：在需求不确定环境下，学习一个在线策略 $\pi_\theta(\mathbf{x}_k | s_k)$，最大化期望长期效用，**严格满足切换预算约束**。

**两阶段训练**：
1. **BC 预训练**：监督学习模仿 P2 离线最优
2. **PPO 在线微调**：在带噪环境中 fine-tune

**可选扩展**：BC 阶段或 PPO 阶段使用 Transformer 策略替代 MLP（详见后文）。

## 数学问题

### 决策对象

$$
\pi_\theta : s_k \mapsto \pi(\mathbf{x}_k | s_k)
$$

参数化为按小区分解的 softmax：
$$
\pi_\theta(\mathbf{x}_k | s_k) = \prod_{c} \pi_\theta(x_{:,c,k} | s_k)
$$

### MDP 定义

- **状态** $s_k = (\{g_{s,c,k}\}, \{v_{s,c,k}\}, \{\hat{a}_{c,k+\tau}\}_\tau, \mathbf{x}_{k-1}, \{H_c^{\text{rem}}\}, k)$
- **动作** $a_k = \mathbf{x}_k \in \prod_c \mathcal{S}_{c,k}^{\text{cand}}$
- **奖励** $r_k = \sum_c \log(\epsilon + \xi_{c,k}) - \lambda_h \sum_c h_{c,k}$
- **状态转移**：由卫星轨道（确定）+ 需求过程（随机）决定

### 训练目标

$$
\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{k=1}^{K} r_k\right]
$$

## 模块实现

### 4.1 RL 环境

#### 文件位置
`src/leo_alloc/rl/env.py`

#### 接口

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LEOSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}
    
    def __init__(self, scenario_generator: Callable, 
                 p1_solver: P1Solver,
                 predictor: DemandPredictor,
                 episode_K: int = 30):
        ...
        self.observation_space = spaces.Dict({
            'g': spaces.Box(low=0, high=1e-8, shape=(S, C)),
            'v': spaces.MultiBinary((S, C)),
            'demand_pred': spaces.Box(low=0, high=1e10, shape=(C, M*H)),
            'prev_x': spaces.MultiBinary((S, C)),
            'h_remaining': spaces.Box(low=0, high=K, shape=(C,), dtype=np.int32),
            'k': spaces.Discrete(K),
        })
        # 按小区分解的离散动作（每个小区选一颗候选卫星）
        max_S_cand = self._compute_max_candidates()
        self.action_space = spaces.MultiDiscrete([max_S_cand] * C)
    
    def reset(self, *, seed=None, options=None):
        """生成新场景，返回初始观测"""
        self.scenario = self.scenario_generator(seed=seed)
        self.k = 0
        ...
        return self._get_obs(), {}
    
    def step(self, action):
        """
        action: [C] 每个小区选择的卫星索引（在候选集合内）
        """
        # 1. 解码 action → x_k 矩阵
        x_k = self._decode_action(action)
        
        # 2. 暴露当前慢时隙的真实需求 a_k
        a_true = self.scenario.a[:, self.k, :]
        
        # 3. 调用 P1 求解器
        result = self.p1_solver.solve(x_k, a_true, self.scenario.g[:, :, self.k])
        
        # 4. 计算 reward 和 h_k
        h_k = self._compute_handover(x_k, self.prev_x)
        r_k = np.sum(np.log(self.eps + result.xi)) - self.lambda_h * np.sum(h_k)
        
        # 5. 更新内部状态
        self.prev_x = x_k.copy()
        self.h_remaining -= h_k
        self.k += 1
        
        done = (self.k >= self.episode_K)
        return self._get_obs(), r_k, done, False, {'xi': result.xi, 'h': h_k}
    
    def get_action_mask(self) -> np.ndarray:
        """返回当前动作掩码 [C, max_S_cand]，True=允许"""
        ...
```

#### 关键设计

**1. P1Solver 嵌入**：env 持有 p1_solver 引用，每次 step 调用一次。这是 RL 训练的主要瓶颈，必须用 DPP 优化的 L1 或 L2 版本。

**2. 信息边界**：状态 $s_k$ 中的 demand_pred 来自 predictor，**绝不能**直接访问 `scenario.a[:, k:, :]`（这是未来真实数据，泄漏会让训练分数虚高）。在测试时严格检查。

**3. 终止条件**：episode 长度固定为 K（不用 early termination，便于对比）。

**4. 硬预算边界**：主训练和主测试场景必须满足 P2 文档中的可行性假设。若 action mask 触发 emergency override，应在 `info` 中返回 `emergency_handover=True`，并把该 episode 从“严格硬预算满足”的主统计中剔除。

### 4.2 Action Masking

#### 文件位置
`src/leo_alloc/rl/masking.py`

```python
def compute_action_mask(scenario: ScenarioInstance, k: int, 
                        prev_x: np.ndarray,           # [S, C]
                        h_remaining: np.ndarray        # [C]
                       ) -> np.ndarray:
    """
    Returns mask of shape [C, max_S_cand], where True means allowed.
    
    Mask logic:
    1. Cell c can only choose visible satellites: v[s,c,k] = 1
    2. If h_remaining[c] == 0, force same satellite as prev_x
    3. Conflict (prev satellite invisible AND budget=0): allow any visible
       (emergency override), set an emergency flag, and exclude the episode
       from strict-budget headline metrics
    """
    C = scenario.C
    max_S_cand = ...
    mask = np.zeros((C, max_S_cand), dtype=bool)
    
    for c in range(C):
        visible_sats = np.where(scenario.v[:, c, k] == 1)[0]
        candidates = ...  # 候选集合索引
        
        if h_remaining[c] > 0:
            mask[c, candidates_visible] = True
        else:
            prev_sat = np.argmax(prev_x[:, c])
            if scenario.v[prev_sat, c, k] == 1:
                mask[c, idx_of(prev_sat, candidates)] = True
            else:
                # 紧急切换：保持决策可行，但 emergency flag = True
                mask[c, candidates_visible] = True
    
    return mask
```

#### 在策略网络中使用

```python
def sample_action(logits, mask):
    """采样前掩盖不可行动作的 logits"""
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    probs = F.softmax(masked_logits, dim=-1)
    action = Categorical(probs).sample()
    return action, masked_logits
```

### 4.3 策略网络（MLP 版本，必交）

#### 文件位置
`src/leo_alloc/rl/policy_mlp.py`

```python
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, max_S_cand: int, C: int, 
                 hidden_dims=(256, 256, 256)):
        super().__init__()
        # 共享编码器
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        
        # C 个独立的 softmax 头
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, max_S_cand) for _ in range(C)
        ])
        
        # 价值函数（PPO 需要）
        self.value_head = nn.Linear(in_dim, 1)
    
    def forward(self, obs_flat: torch.Tensor):
        """
        obs_flat: [B, obs_dim] 拉平的观测
        returns: logits [B, C, max_S_cand], value [B]
        """
        z = self.encoder(obs_flat)
        logits = torch.stack([head(z) for head in self.heads], dim=1)  # [B, C, max_S_cand]
        value = self.value_head(z).squeeze(-1)
        return logits, value
```

#### 观测拉平的注意事项

`obs_dim` 由各分量维度决定：
- g: S*C
- v: S*C
- demand_pred: C * (M*H) 其中 H 是预测视野
- prev_x: S*C
- h_remaining: C
- k 用 sin/cos 编码: 2

写一个 `obs_to_flat(obs_dict)` 函数，确保顺序一致。

### 4.4 策略网络（Transformer 版本，可选）

#### 文件位置
`src/leo_alloc/rl/policy_transformer.py`

```python
class TransformerPolicy(nn.Module):
    def __init__(self, obs_dim_per_step: int, max_S_cand: int, C: int,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 context_len: int = 20):
        super().__init__()
        # 分模态嵌入
        self.embed_channel = nn.Linear(S*C, d_model)
        self.embed_visibility = nn.Linear(S*C, d_model)
        self.embed_demand = nn.Linear(C*M*H, d_model)
        self.embed_assoc = nn.Linear(S*C, d_model)
        self.embed_budget = nn.Linear(C, d_model)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(context_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, 
            dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出头（与 MLP 相同）
        self.heads = nn.ModuleList([
            nn.Linear(d_model, max_S_cand) for _ in range(C)
        ])
        self.value_head = nn.Linear(d_model, 1)
    
    def forward(self, obs_seq: dict):
        """
        obs_seq: dict of [B, L, ...] tensors, L = context length
        returns: logits [B, C, max_S_cand], value [B]
        """
        # 分模态嵌入
        e_g = self.embed_channel(obs_seq['g'].flatten(2))
        e_v = self.embed_visibility(obs_seq['v'].flatten(2))
        e_d = self.embed_demand(obs_seq['demand_pred'].flatten(2))
        e_x = self.embed_assoc(obs_seq['prev_x'].flatten(2))
        e_b = self.embed_budget(obs_seq['h_remaining'])
        
        # 加性融合 + 位置编码
        e = e_g + e_v + e_d + e_x + e_b + self.pos_embed[:e_g.shape[1]]
        
        # Transformer 编码（因果 mask）
        mask = nn.Transformer.generate_square_subsequent_mask(e.shape[1]).to(e.device)
        h = self.transformer(e, mask=mask)  # [B, L, d_model]
        
        # 取最后一个 token
        h_last = h[:, -1, :]
        logits = torch.stack([head(h_last) for head in self.heads], dim=1)
        value = self.value_head(h_last).squeeze(-1)
        return logits, value
```

#### 滑动窗口缓存

```python
class StateBuffer:
    def __init__(self, context_len: int, obs_template: dict):
        self.L = context_len
        self.buffer = {k: deque(maxlen=context_len) for k in obs_template}
        # 初始化时用 zeros 填充
        for k, v in obs_template.items():
            for _ in range(context_len):
                self.buffer[k].append(np.zeros_like(v))
    
    def append(self, obs: dict):
        for k in self.buffer:
            self.buffer[k].append(obs[k])
    
    def get_sequence(self) -> dict:
        return {k: np.stack(list(v)) for k, v in self.buffer.items()}
```

### 4.5 需求预测器（3 档）

#### 文件位置
`src/leo_alloc/rl/predictor.py`

```python
class DemandPredictor(ABC):
    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """history: [C, T_history], returns [C, horizon]"""

class P0NaivePredictor(DemandPredictor):
    """直接复用上一时隙"""
    def predict(self, history, horizon):
        last = history[:, -1:]
        return np.tile(last, (1, horizon))

class P1EWMAPredictor(DemandPredictor):
    """指数加权移动平均 + 周期项"""
    def __init__(self, alpha=0.3, period=24):
        ...
    def predict(self, history, horizon):
        ...

class P2LSTMPredictor(DemandPredictor):
    """2-layer LSTM, 64 hidden units"""
    def __init__(self, hidden_dim=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(...)
        ...
    
    def fit(self, train_demands):
        """监督训练"""
        ...
```

**训练**：预测器在 BC/PPO 训练前预先训练好并 freeze，不参与策略训练。

### 4.6 BC 预训练器

#### 文件位置
`src/leo_alloc/rl/bc_trainer.py`

```python
class BCTrainer:
    def __init__(self, policy: nn.Module, lr: float = 3e-4,
                 batch_size: int = 256, n_epochs: int = 50):
        ...
    
    def train(self, demonstrations: List[Path]):
        """
        demonstrations: 由 generate_bc_data.py 生成的 .npz 文件路径列表
        每个文件包含 (s_seq, x_optimal_seq)
        """
        # DataLoader
        dataset = BCDataset(demonstrations)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr)
        
        for epoch in range(self.n_epochs):
            for batch in loader:
                obs, x_star, mask = batch
                logits, _ = self.policy(obs)
                
                # Mask logits before computing loss
                logits_masked = logits.clone()
                logits_masked[~mask] = -1e9
                
                # 按小区独立交叉熵
                loss = 0
                for c in range(C):
                    loss += F.cross_entropy(
                        logits_masked[:, c, :], 
                        x_star[:, c]  # 每个小区的最优卫星索引
                    )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 验证集评估
            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch}: val_acc={val_acc:.3f}")
```

### 4.7 PPO 微调器

#### 文件位置
`src/leo_alloc/rl/ppo_trainer.py`

**推荐方案**：使用 `stable-baselines3` 的 PPO 实现，自定义 policy 类即可：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPPOPolicy(ActorCriticPolicy):
    """包装我们的 MLPPolicy 或 TransformerPolicy 给 SB3 用"""
    ...

# 训练
model = PPO(
    policy=CustomPPOPolicy,
    env=LEOSchedulingEnv(...),
    learning_rate=3e-5,           # BC 阶段的 1/10
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    target_kl=0.02,
    tensorboard_log="./logs/",
)

# 从 BC checkpoint 加载初始权重
model.policy.load_state_dict(torch.load('bc_checkpoint.pt'))

model.learn(total_timesteps=int(2e6))
```

**自定义关键点**：
- Action masking 需要在 `forward` 中实现（SB3 的 `MaskablePPO` 也可用）
- BC-PPO Hybrid loss：在 PPO 前期保留 20% 的 BC loss
- Custom callback 记录约束违反次数（应永远为 0）

## 验证测试集

```python
# tests/test_rl_env.py

class TestEnvCorrectness:
    def test_reset_returns_valid_obs(self):
        ...
    
    def test_step_advances_k(self):
        ...
    
    def test_episode_length(self):
        """episode 长度应严格为 K"""
        ...
    
    def test_no_future_leakage(self):
        """观测中不应包含未来真实需求"""
        # 随机修改 scenario.a[:, k+1:, :] 后再 step，
        # 验证 obs 不变（除非通过 predictor 间接影响）
        ...

class TestActionMasking:
    def test_invisible_satellite_masked(self):
        ...
    
    def test_zero_budget_forces_no_handover(self):
        ...
    
    def test_emergency_override(self):
        ...

    def test_emergency_flag_excludes_strict_budget_metric(self):
        """触发 emergency override 时，info 必须显式标记，评估器不得计入主硬预算统计"""
        ...

class TestBCTraining:
    def test_overfit_small_dataset(self):
        """10 条轨迹 100 epoch 应能完全记住（loss → 0）"""
        ...

class TestPPOConvergence:
    def test_pure_ppo_on_trivial_env(self):
        """C=2, K=5 的简单场景，1k step 内应达到 oracle 的 80%"""
        ...
```

## 实验消融设计

主消融实验对比（论文 §5）：

| 标签 | 方法 | 验证的设计选择 |
|---|---|---|
| B0 | 贪心 | 下界 |
| B1 | Oracle (P2 离线最优) | 上界 |
| B2 | MPC | 模型驱动强基线 |
| B3 | 纯 PPO（无分层 + MLP） | 分层的必要性 |
| B4 | 分层 + MLP + 无 BC | BC 热启的价值 |
| B5 | 分层 + MLP + BC + PPO | **本文主方法（MLP 版）** |
| B6 | 分层 + Transformer + BC + PPO | **本文方法（Transformer 版，可选）** |

**核心叙事**：B3 < B4 < B5 < B6 < B1，且 B5/B6 在预测误差 ≥ 15% 时优于 B2。

## 完成判定

1. ✅ 所有测试通过
2. ✅ B5（MLP 版本）训练收敛，相对 B1 的 gap ≤ 20%
3. ✅ 主实验约束违反次数 = 0，且 `emergency_handover_count = 0`
4. ✅ 鲁棒性曲线（预测误差 5%/10%/20%）数据完整
5. ✅ 完整消融实验图表生成
6. ✅ （可选）B6 训练完成且优于 B5

**B6 是加分项，不达到不影响毕业。**
