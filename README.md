# Advanced-Robotics_RL

《机器人进阶》课程的代码练习，按周整理。

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| [Week1](Week1/README.md) | LunarLander-v3 强化学习入门 | PPO 训练、评估与录制 |
| [Week2](Week2/README.md) | 深度学习基本原理实践 | PyTorch 基础、函数拟合、MNIST |
| [Week3](Week3/README.md) | 深度强化学习原理实践 | REINFORCE、A2C、Q-Learning、Atari DQN |
| [Week5](Week5/README.md) | 模仿学习与 BeyondMimic 实践 | 行为克隆、DAgger、G1 运动追踪 |
| [Week6](Week6/README.md) | 离线强化学习实践 | Pointmass 探索、DQN/CQL/AWAC/IQL、在线微调 |
| [Week7](Week7/README.md) | 基于模型的强化学习实践 | 动力学模型集成、MPC/CEM、MBPO/SAC |
| [Stage1](Stage1/README.md) | 强化学习进阶实践 | LunarLander PPO 原理实现、ViZDoom 生存场景 |

各目录的 README 包含依赖和使用说明。Week2、Week3、Stage1 使用 `uv` 管理 Python 环境。Stage1 的两个项目使用独立环境。

训练输出、模型权重、视频、数据缓存和本地虚拟环境不纳入仓库。

## Week5–7：使用 uv

在 Windows 上，模仿学习、Week6 和 Week7 分别使用独立的 Python 3.10 环境。三个项目的 `.venv` 已按各自的 `uv.lock` 同步，可在对应目录直接用 `uv run` 启动：

```powershell
cd Week5/imitation_learning
uv run --frozen --offline python cs285/scripts/run_hw1.py --help

cd ../../Week6
uv run --frozen --offline python cs285/scripts/run_hw5_explore.py --help

cd ../Week7
uv run --frozen --offline python cs285/scripts/run_hw4.py --help
```

在已有缓存的机器上，可在上述每个目录运行 `uv sync --frozen --offline` 恢复环境，无需联网。若缓存缺失，需要在线解析或安装普通 PyPI 依赖，可运行 `uv sync --default-index https://mirrors.aliyun.com/pypi/simple/`；项目中的 CUDA 版 PyTorch 仍由 `pyproject.toml` 指定的 PyTorch 官方 CUDA 12.1 索引提供。重新解析后检查生成的 `uv.lock`。

Week5 的 `beyondmimic/` 另需 Isaac Sim 4.5.0、Isaac Lab 2.1.0、RSL-RL、机器人模型和动作数据，不属于上述三个 uv 环境；参见 [BeyondMimic 说明](Week5/beyondmimic/README.md)。
