# 08. Environment Setup

本文档记录项目推进所需环境、当前已配置内容，以及必须手动完成的部分。

## 已配置内容

当前工作区已创建 `.venv`，使用本机 `python3`：

```bash
Python 3.13.7
```

已安装依赖组：

```bash
.venv/bin/python -m pip install -e ".[solvers,rl,dev,notebook]"
.venv/bin/python -m pip install -e ".[commercial-solvers]"
```

已验证可用：

| 类别 | 状态 |
|---|---|
| NumPy / SciPy / Pandas | 已安装 |
| CVXPY | 已安装 |
| ECOS / CLARABEL / SCS / OSQP | 已安装 |
| PyTorch / Gymnasium / stable-baselines3 / sb3-contrib | 已安装 |
| Skyfield / sgp4 | 已安装 |
| pytest / ruff / mypy | 已安装并通过基础检查 |
| Gurobi Python 包 | 已安装，license 可用 |
| MOSEK Python 包 | 已安装，license 缺失 |

## 使用环境

推荐每次进入项目后运行：

```bash
source scripts/activate_env.sh
```

这个脚本会激活 `.venv`，并把 Matplotlib/fontconfig 缓存写入项目内 `.cache/`，避免用户 home 目录不可写导致 warning。

## 环境检查

运行：

```bash
.venv/bin/python scripts/check_env.py
```

检查内容包括关键包导入、CVXPY solver 列表，以及 ECOS/CLARABEL/GUROBI/MOSEK 的最小求解测试。

## 需要手动完成

### 1. MOSEK academic license

MOSEK 当前报错：

```text
License cannot be located. The default search path is ':/Users/clarity256/mosek/mosek.lic:'.
```

需要你手动申请并放置 license 文件。推荐路径：

```bash
mkdir -p /Users/clarity256/mosek
cp /path/to/mosek.lic /Users/clarity256/mosek/mosek.lic
```

也可以设置环境变量：

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

P1 的 CVXPY 基准求解可以先用 ECOS/CLARABEL 推进；论文规模和数值稳定性实验建议补齐 MOSEK。

### 2. Starlink TLE 数据

`data/tle/` 目录已创建。推荐使用脚本下载一个带时间戳和 SHA256 的本地快照：

```bash
.venv/bin/python scripts/download_starlink_tle.py --source gp
```

默认源为 CelesTrak Current GP Data 的 Starlink group。也可以使用 SpaceX-derived supplemental source：

```bash
.venv/bin/python scripts/download_starlink_tle.py --source supplemental
```

论文实验应读取已保存的本地快照，不要在每次实验运行时在线拉取最新 TLE。这样 seed、TLE epoch、可见性矩阵和结果才能复现。

如果需要历史指定日期的 Starlink 数据，CelesTrak 当前接口不适合作为历史数据库；建议使用 Space-Track，并由你手动配置账号凭据。不要把凭据提交进仓库。

### 3. 更稳妥的 Python 版本（可选）

当前 Python 3.13.7 已能安装并导入项目依赖。若后续遇到求解器或 RL 包兼容问题，建议手动安装 Python 3.11 或 3.12，并重建 `.venv`。现在没有必要中断推进。
