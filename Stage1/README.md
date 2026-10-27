# Stage1：强化学习进阶实践

按《强化学习进阶实践内容(终).pptx》的 37 页内容整理为两个独立项目：

| 项目 | 目录 | 复现内容 |
| --- | --- | --- |
| 月球着陆器 | [`lunar_lander/`](./lunar_lander/) | 从零实现离散动作 PPO：Actor/Critic、并行采样、GAE、裁剪损失、评估和视频 |
| Doom 生存小游戏 | [`doom/`](./doom/) | 使用 Sample Factory 的 ViZDoom 场景集成：PPO 参数、卷积编码器、并行采样、评估和视频 |

两个项目各有自己的 `pyproject.toml` 和 uv 环境。这样可以隔离 Sample Factory 2.1.1 所需的旧版 Gymnasium，避免和 LunarLander 的新版 Gymnasium 依赖冲突。本次没有安装依赖，也没有启动训练；下列命令供后续使用。

## 1. LunarLander：PPO 月球着陆

### 环境和奖励

课件写 `LunarLander-v2`，代码使用 Gymnasium 当前的 `LunarLander-v3`。状态是 8 维向量：位置、速度、姿态、角速度和两条着陆腿触地情况；动作是 4 个：不点火、左侧引擎、主引擎、右侧引擎。[Gymnasium 官方文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/)列出了状态、动作和奖励规则。代码直接使用环境奖励：靠近着陆点、降低速度、摆正姿态和触地会影响每步奖励；点火有燃料惩罚，着陆或坠毁有终局奖励。没有在脚本中再次添加奖励，以免重复计算。

课件把每步奖励写成势函数差 `Φ(s') - Φ(s)`，这用于解释环境中的 reward shaping。若讨论“保持最优策略不变”的严格条件，通常需要使用 `γΦ(s') - Φ(s)` 并满足相应终止边界条件；本项目不修改 Gymnasium 的原始奖励。

### 网络和 PPO 更新

[`lunar_lander/ppo.py`](./lunar_lander/ppo.py) 中的 Actor 和 Critic 各自使用 `8→64→64`、Tanh 激活；Actor 输出 4 个动作的 logits，Critic 输出一个状态价值。`get_action_and_value()` 返回采样动作、对数概率、策略熵和价值。

默认 4 个环境各采样 128 步，每轮得到 512 条样本；随机分成 4 个 minibatch，训练 4 个 epoch。使用 `γ=0.99`、`λ=0.95` 逆序计算 GAE，优势加上旧价值构成回报目标。策略目标计算新旧动作概率比并裁剪到 `[0.8, 1.2]`；价值目标取普通平方误差与裁剪价值平方误差中较大者；总损失为 `策略损失 + 0.5 × 价值损失 − 0.01 × 熵`。Adam 初始学习率为 `2.5e-4`，随 PPO 迭代线性下降。对梯度范数设 `0.5` 上限。

训练时按 Gymnasium 的 `terminated` 和 `truncated` 分开处理。遇到时间截断，会在重置前对最终状态做价值 bootstrap；真正终止则不做。评估使用独立环境和确定性动作。

```bash
cd Stage1/lunar_lander
uv sync
uv run python ppo.py inspect
uv run python ppo.py train --total-timesteps 500000
uv run python ppo.py evaluate --episodes 10
uv run python ppo.py record
uv run tensorboard --logdir outputs/tensorboard
```

课件演示命令中的 `50,000` 步可写成 `--total-timesteps 50000`，用于短程实验。默认模型保存在 `outputs/lunar_ppo.pt`，视频在 `outputs/videos/`。一次 PPO 更新必须收集完整的 512 步，因此实际步数可能略高于请求值。Week1 的 [`lunar_lander.py`](../Week1/lunar_lander.py) 已用 Stable Baselines3 完成过同一类任务；这里单独实现 PPO 内部计算，便于对照课件第 12–14 页。

## 2. ViZDoom：Health Gathering Supreme

场景目标是在酸性地板持续伤害下移动并收集医疗包，延长存活时间。代码使用 [Sample Factory 官方 ViZDoom 集成](https://www.samplefactory.dev/09-environment-integrations/vizdoom/)中的 `doom_health_gathering_supreme`。它会注册场景、创建 ViZDoom 环境，并使用 [DoomGatheringRewardShaping](https://github.com/alex-petrenko/sample-factory/blob/v2.1.1/sf_examples/vizdoom/doom/wrappers/scenario_wrappers/gathering_reward_shaping.py)：生命值增加时增加 1 分，回合结束时保留原始环境奖励作为 `true_objective`。

[`doom/train.py`](./doom/train.py) 将课件配置转换成 Sample Factory 命令：8 个 rollout worker，每个 4 个环境，共 32 个环境；每个环境收集 256 步后得到 8,192 条样本，恰好对应 2,048 条的 minibatch × 4 个 batch。每组数据训练 4 个 epoch，总训练目标为 4,000,000 环境步。设置 `async_rl=False` 和 `with_vtrace=False`，让采样与更新按同步 PPO/GAE 流程进行。环境采样、策略推理、经验缓冲、检查点由 Sample Factory 实现。

图像输入设为 `3×72×128`。Sample Factory 的 [`convnet_simple`](https://github.com/alex-petrenko/sample-factory/blob/v2.1.1/sample_factory/model/encoder.py) 与课件卷积结构一致：

```text
3×72×128
→ Conv(3,32,8,stride=4)   → 32×17×31
→ Conv(32,64,4,stride=2)  → 64×7×14
→ Conv(64,128,3,stride=2) → 128×3×6
→ Flatten(2304) → Linear(2304,512) → ReLU
→ Actor logits / Critic value
```

像素先除以 255，禁用额外的输入统计归一化；共享编码器，不启用 RNN。配置 `γ=0.99`、`λ=0.95`、策略裁剪系数 `0.2`、价值裁剪 `0.2`、价值损失权重 `0.5`、熵探索权重 `0.01`、Adam 学习率 `1e-4`。

```bash
cd Stage1/doom
uv sync
uv run python train.py --dry-run       # 只打印将调用的 Sample Factory 命令
uv run python train.py                 # 完整训练，计算量很大
uv run python evaluate.py --dry-run
uv run python evaluate.py              # 评估 10 局并录制一局
uv run tensorboard --logdir train_dir
```

训练结果位于 `train_dir/default_experiment/`。[`doom/evaluate.py`](./doom/evaluate.py) 用同名实验加载检查点，评估 10 局并请求保存一局回放视频。`--device gpu` 仅适用于 Sample Factory 能识别的 GPU；默认使用 CPU，便于先检查命令链路，但完整 Doom 训练会很慢。Sample Factory 和 ViZDoom 可能还需要系统图形库，详见[官方安装说明](https://www.samplefactory.dev/09-environment-integrations/vizdoom/#installation)。

### 与课件示意代码的差异

| 课件写法 | 实际复现方式 |
| --- | --- |
| Doom 策略头输出 3 个动作 | [官方场景注册](https://github.com/alex-petrenko/sample-factory/blob/v2.1.1/sf_examples/vizdoom/doom/doom_utils.py)是 `Discrete(5)`；策略头由 Sample Factory 按环境动作空间自动创建，输出 5 个 logits。 |
| `encoder_conv_architecture="vizdoom"` | Sample Factory 2.1.1 支持的名称是 `convnet_simple`；其三层卷积恰好匹配课件。 |
| `entropy_loss_coeff=0.01` | 该版本参数名为 `exploration_loss_coeff=0.01`，并显式选择 `exploration_loss=entropy`。 |
| `lr_schedule="linear"` | 该版本只提供 constant 和两种自适应 KL 调度；Doom 脚本使用固定 `1e-4`。月球着陆器脚本实现了课件的线性退火。 |
| `ppo_clip_ratio=0.2` 画成严格 `[0.8,1.2]` | Sample Factory 的具体裁剪实现与课件简化公式稍有不同；月球着陆器脚本则按 `[0.8,1.2]` 实现。 |
| `train.py`、`evaluate.py` 直接调用框架函数 | 这里封装官方 `sf_examples.vizdoom` 训练和评估入口，保留框架的环境注册与检查点处理。 |

两个项目没有共享模型权重，Week1 的已训练模型也不直接用于此处。由于没有训练，本目录尚无得分、视频或模型检查点。
