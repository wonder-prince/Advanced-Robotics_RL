# Advanced-Robotics_RL

《机器人进阶》课程的代码练习，按周整理。

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| [Week1](Week1/README.md) | LunarLander-v3 强化学习入门 | PPO 训练、评估与录制 |
| [Week2](Week2/README.md) | 深度学习基本原理实践 | PyTorch 基础、函数拟合、MNIST |
| [Week3](Week3/README.md) | 深度强化学习原理实践 | REINFORCE、A2C、Q-Learning、Atari DQN |
| [Week4](Stage1_Week4/README.md) | 强化学习进阶实践 | LunarLander PPO 原理实现、ViZDoom 生存场景 |
| [Week5](Week5/README.md) | 模仿学习与 BeyondMimic 实践 | 行为克隆、DAgger、G1 运动追踪 |
| [Week6](Week6/README.md) | 离线强化学习实践 | Pointmass 探索、DQN/CQL/AWAC/IQL、在线微调 |
| [Week7](Week7/README.md) | 基于模型的强化学习实践 | 动力学模型集成、MPC/CEM、MBPO/SAC |
| [Week8](Week8/README.md) | 多智能体强化学习实践 | Unity ML-Agents、PPO/RND、MA-POCA 自博弈 |
| [Week9](Week9/README.md) | 实例分割进阶实验 | CenterMask-Lite 推理及置信度阈值对比 |

各目录的 README 包含依赖和使用说明。Week4 的两个项目使用独立的 uv 环境。

训练输出、模型权重、视频、数据缓存和本地虚拟环境不纳入仓库。

## Week5–7：使用 uv

模仿学习、Week6 和 Week7 分别使用独立的 Python 3.10 环境。锁文件支持 Windows 和 Apple Silicon macOS，Week6–7 也支持 x86-64 Linux（含 WSL 2）。MuJoCo 固定为 2.2.2。Windows 使用 CUDA 12.1，Week6–7 在 Linux 使用 CUDA 12.8，Apple Silicon 使用原生 PyTorch。同步后可在对应目录用 `uv run` 启动：

```powershell
cd Week5/imitation_learning
uv run --frozen --offline python cs285/scripts/run_hw1.py --help

cd ../../Week6
uv run --frozen --offline python cs285/scripts/run_hw5_explore.py --help

cd ../Week7
uv run --frozen --offline python cs285/scripts/run_hw4.py --help
```

已有缓存时可用 `uv sync --locked --offline` 恢复环境。缓存缺失时使用 `uv sync --locked` 安装锁定的依赖。

## Linux / WSL 2 上的 NVIDIA GPU

在 x86-64 Ubuntu 上执行。WSL 2 使用 Windows 主机上的 NVIDIA 驱动，在 Ubuntu 内通过 `nvidia-smi` 检查设备。Week3 Atari 锁定的 Linux PyTorch 使用 CUDA 13，驱动需支持该版本。Week6–7 使用 CUDA 12.8。uv 会安装对应的 PyTorch 运行库。

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglfw3 libglib2.0-0
nvidia-smi
bash setup-gpu.sh week3
```

可将 `week3` 换成 `week6`、`week7` 或 `all`。脚本安装锁定依赖，并用 CUDA 张量乘法检查 GPU。Week3 的 CNN DQN 可从仓库根目录这样启动。

```bash
cd Week3
uv run --no-sync python -m rl_zoo3.train \
  --algo dqn --env SpaceInvadersNoFrameskip-v4 --conf-file dqn.yml \
  --device cuda --log-folder logs/course-gpu --save-freq 250000
uv run --no-sync python -m rl_zoo3.enjoy \
  --algo dqn --env SpaceInvadersNoFrameskip-v4 \
  --folder logs/course-gpu --device cuda --no-render --n-timesteps 5000
```

用 `--trained-agent <模型.zip>` 可以加载已有权重。中途保存的权重未附经验回放池。从头跑可保留完整的 1,000 万步实验设置。Week6–7 同步后按各周 README 的训练命令执行，省略 `--no_gpu` 并检查启动日志中的 GPU 设备。Week6 离线与微调需要先生成或复制 `datasets/*.pkl`。

本机的 CPU 测速估算 Week7 的 HalfCheetah 多轮随机 MPC 和 CEM 仅规划部分就分别需要约 3.7 和 4.9 小时。在仓库根目录执行 `bash setup-gpu.sh week7` 后，可用以下命令运行原始课件配置。模拟和逐步规划仍可能受 CPU 速度限制。

```bash
cd Week7
uv run --no-sync python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_multi_iter.yaml
uv run --no-sync python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_cem.yaml
```

Week5 的 `beyondmimic/` 另需 Isaac Sim 4.5.0、Isaac Lab 2.1.0、RSL-RL、机器人模型和动作数据，不属于上述三个 uv 环境；参见 [BeyondMimic 说明](Week5/beyondmimic/README.md)。
