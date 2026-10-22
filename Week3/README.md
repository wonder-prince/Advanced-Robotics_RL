# Week 3：深度强化学习原理实践

依据《第3周-深度强化学习原理实践.pptx》整理。四段练习均保留课件中的环境、算法和主要超参数，代码按当前 Gymnasium 接口写成可独立使用的脚本。本次只创建文件；**没有创建 uv 环境、安装依赖或运行训练与评估**。

## 文件与课件对应

| 课件内容 | 文件 | 环境与方法 |
| --- | --- | --- |
| 策略梯度 | `reinforce.py` | CartPole-v1、FlappyBird-v0；REINFORCE |
| Actor-Critic | `panda_a2c.py` | PandaReachDense-v3；Stable Baselines3 A2C |
| 值函数表格法 | `q_learning.py` | FrozenLake-v1、Taxi-v3；Q-Learning |
| 深度 Q 网络 | `dqn.yml`、`dqn_smoke.yml` | SpaceInvadersNoFrameskip-v4；RL Baselines3 Zoo DQN |

## uv 环境

在 `Week3` 目录执行。`.python-version` 选择 Python 3.11，`pyproject.toml` 按实验拆分可选依赖，避免只做 FrozenLake 时安装 PyTorch、Panda 和 Atari 依赖。以下均为**以后需要运行时**的命令，本次没有执行。

```bash
cd Week3
uv sync                         # Q-Learning 的基础依赖
uv sync --extra reinforce       # CartPole REINFORCE
uv sync --extra flappy          # FlappyBird REINFORCE
uv sync --extra panda           # Panda A2C
uv sync --extra atari           # Atari DQN
```

需要多组依赖时，可以在同一次 `uv sync` 中重复 `--extra`。下面的 `uv run --extra ...` 会按对应依赖组同步环境；如果已经手工 `uv sync`，也可以使用 `uv run --no-sync ...`。

## 1. REINFORCE：CartPole 与 FlappyBird

策略网络根据状态产生动作分布。每回合记录 `log π(a|s)` 与奖励，倒序计算折扣回报，将回报标准化，再以 `-log π(a|s) × G` 的和更新策略。CartPole 使用一层 16 单元隐藏层；FlappyBird 使用两层 64 单元隐藏层。

| 环境 | 训练回合 | 评估回合 | 每回合最多步数 | γ | 学习率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CartPole-v1 | 1,000 | 10 | 1,000 | 1.0 | 0.01 |
| FlappyBird-v0 | 5,000 | 10 | 1,000 | 0.99 | 0.0001 |

```bash
uv run --extra reinforce python reinforce.py train --env cartpole
uv run --extra reinforce python reinforce.py evaluate --env cartpole

uv run --extra flappy python reinforce.py train --env flappy
uv run --extra flappy python reinforce.py evaluate --env flappy
```

模型分别保存在 `outputs/reinforce/cartpole.pt` 和 `outputs/reinforce/flappy.pt`。`--episodes`、`--seed`、`--checkpoint` 可覆盖默认值。评估时按照课件从策略分布采样 10 回合，并输出平均奖励与标准差。

## 2. A2C：Panda 机械臂

`PandaReachDense-v3` 的观测是字典，因此使用 `MultiInputPolicy`。四个并行环境经 `VecNormalize` 归一化观测和奖励，训练 1,000,000 步。训练时同时保存模型与归一化统计；评估时加载同一统计，关闭统计更新和奖励归一化。

```bash
uv run --extra panda python panda_a2c.py train
uv run --extra panda python panda_a2c.py evaluate
```

输出位于 `outputs/panda_a2c/model.zip` 和 `outputs/panda_a2c/vecnormalize.pkl`。可用 `--timesteps`、`--episodes`、`--seed`、`--output-dir` 调整。

## 3. Q-Learning：FrozenLake 与 Taxi

使用 ε-greedy 探索，并按 `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)]` 更新。环境真正终止时，不再引入下一状态价值。FrozenLake 固定为 4×4 且 `is_slippery=False`。

| 环境 | Q 表形状 | 训练回合 | 评估回合 | 最多步数 | α | γ | ε 衰减系数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FrozenLake-v1 | 16×4 | 10,000 | 100 | 99 | 0.7 | 0.95 | 0.0005 |
| Taxi-v3 | 500×6 | 25,000 | 100 | 99 | 0.7 | 0.95 | 0.005 |

两者都从 ε=1.0 开始，最低降至 0.05。评估采用贪心动作，并输出 100 回合平均奖励和标准差。

```bash
uv run python q_learning.py train --env frozenlake
uv run python q_learning.py evaluate --env frozenlake

uv run python q_learning.py train --env taxi
uv run python q_learning.py evaluate --env taxi
```

每个环境的 `q_table.npy` 与 `config.json` 保存在 `outputs/q_learning/<环境名>/`。可用 `--episodes`、`--seed`、`--output-dir` 调整。

## 4. DQN：Atari Space Invaders

课件这一段借助 RL Baselines3 Zoo，并非从零手写 DQN。`dqn.yml` 是课件的完整 10,000,000 步配置：AtariWrapper、4 帧堆叠、`CnnPolicy`、100,000 容量经验回放、0.0001 学习率和 ε 探索。`dqn_smoke.yml` 是课件的 50,000 步短程配置，仅用于将来检查流程。

```bash
uv run --extra atari python -m rl_zoo3.train \
  --algo dqn --env SpaceInvadersNoFrameskip-v4 \
  --conf-file dqn.yml --log-folder logs/

# 如需先检查命令链路，换用短程配置：
uv run --extra atari python -m rl_zoo3.train \
  --algo dqn --env SpaceInvadersNoFrameskip-v4 \
  --conf-file dqn_smoke.yml --log-folder logs/

# 训练完成后评估/回放：
uv run --extra atari python -m rl_zoo3.enjoy \
  --algo dqn --env SpaceInvadersNoFrameskip-v4 \
  --folder logs/ --no-render --n-timesteps 5000
```

Zoo 会通过 `ale-py` 注册 Atari 环境；`logs/` 是它自己的模型与日志目录。短程配置的结果不能用来判断 DQN 最终表现。

## 对课件代码做的必要整理

- 笔记本中隐式使用的 `env`、模型和参数改为脚本内创建与传递。
- REINFORCE 回报只倒序计算一次，并修正评估累加奖励时的变量名；使用 Gymnasium 的 `reset()`、`step()` 返回值。
- A2C 评估加载训练时保存的 `VecNormalize` 统计，避免评估阶段继续更新均值和方差。
- Q-Learning 的终止状态不继续 bootstrap；保存 Q 表后可单独评估。
- DQN 使用当前 RL Zoo 的 `--conf-file` 参数。当前 [RL Zoo 的 Atari 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml) 也采用 AtariWrapper 与 4 帧堆叠。

依赖接口参照：[FlappyBird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)、[Panda Gym 环境列表](https://github.com/qgallouedec/panda-gym/blob/master/docs/usage/environments.rst)、[RL Baselines3 Zoo 训练入口](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/train.py)、[ALE 环境说明](https://ale.farama.org/environments/)。FlappyBird 包的现行奖励规则可能与课件描述不同，训练表现也会受包版本和随机种子影响。由于按要求没有安装或运行，本项目还没有实际运行结果。
