# Week 1：复现 PPO 月球着陆器

本项目按 Hugging Face [Deep RL Course Unit 1 notebook](https://github.com/huggingface/deep-rl-class/blob/main/notebooks/unit1/unit1.ipynb) 的顺序，复现“认识环境 → 创建并行环境 → 建立 PPO 智能体 → 训练 → 评估 → 录制回放 → 可选上传 Hub”。代码在 [`lunar_lander.py`](./lunar_lander.py)，以下命令都从 `Week1` 文件夹运行。

本机已完成一次完整训练、评估与回放录制，具体见 [`RESULTS.md`](./RESULTS.md)。默认完整训练是 100 万环境交互步，计算与安装都可能花较长时间。

## 1. 理解任务

`LunarLander-v3` 给智能体一个 8 维观测：位置 `(x, y)`、线速度 `(vx, vy)`、角度、角速度、左右着陆腿是否触地。离散动作有 4 个：不点火、左侧引擎、主引擎、右侧引擎。每次执行动作后，环境返回新观测、奖励，以及本局是否结束。目标是学到累计奖励更高的着陆策略。

课程 notebook 使用 `LunarLander-v2`。当前 [Gymnasium 文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/) 的环境 ID 是 `LunarLander-v3`，因此这里统一使用 v3。v3 修复了环境重置问题，结果不要直接当作课程 v2 排行榜成绩比较。

## 2. 准备环境

建议 Python 3.10–3.12。在支持 Python、Box2D 的电脑或 Colab 终端中运行：

```bash
cd Week1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。如果 Box2D 构建提示缺少 SWIG，先安装系统的 `swig`，再重试 `pip install -r requirements.txt`。原 notebook 中的 `apt`、`xvfb` 和强制重启内核是针对 Colab 的；本地用 Gymnasium 的 `rgb_array` 渲染与 `RecordVideo`，不需要照抄这些单元格。

## 3. 观察环境，完成一次交互

```bash
python lunar_lander.py inspect
```

它会打印状态空间、动作空间、初始 8 维观测，并随机执行一步，对应 notebook 的 Gymnasium 入门部分。随机一步不代表模型已经学会着陆。

## 4. 创建并行环境和 PPO 模型

`train` 内部先执行 `make_vec_env("LunarLander-v3", n_envs=16)`，再用 `PPO("MlpPolicy", env, ...)`。`MlpPolicy` 适合这里的数值向量观测。模型沿用 notebook 解答中的参数：

| 参数 | 值 | 用途 |
| --- | ---: | --- |
| `n_envs` | 16 | 同时采样 16 个环境 |
| `n_steps` | 1024 | 每个环境一次收集的步数 |
| `batch_size` | 64 | 每次梯度更新的样本数 |
| `n_epochs` | 4 | 每批数据的训练轮数 |
| `gamma` | 0.999 | 未来奖励折扣系数 |
| `gae_lambda` | 0.98 | 优势估计参数 |
| `ent_coef` | 0.01 | 鼓励探索的熵系数 |

这些参数的默认值写在脚本中，可以通过命令行修改。训练时使用 CPU，因为这个小型 MLP 配合并行环境通常受环境采样与数据传输影响，GPU 并非必要条件。

## 5. 训练并保存

完整复现课程设置：

```bash
python lunar_lander.py train --timesteps 1000000 --n-envs 16
```

模型保存到 `outputs/ppo-LunarLander-v3.zip`。PPO 每次收集 `n_envs × n_steps` 个交互步，因此实际步数会向上取整到完整 rollout；默认设置下是 `1,015,808` 步。

若以后只想检查流程是否能运行，可以缩小配置；这**不会**训练出会着陆的模型：

```bash
python lunar_lander.py train --timesteps 64 --n-envs 1 --rollout-steps 32 --batch-size 32
```

启动后会先提示加载 Gymnasium、PyTorch 和训练库，再提示创建环境与开始训练。首次加载依赖可能比这 64 步训练还慢；看到“开始收集经验并训练”后，才算真正进入训练阶段。如果几分钟后仍只看到第一条提示，可以按 `Ctrl+C` 停止，再用 `python -u lunar_lander.py train ...` 重试，并保留终端报错供排查。模型只有在训练结束时才会保存。

完整训练的得分会随随机种子、依赖版本和机器变化，不能保证一次就达到 200 分。

## 6. 用独立环境评估

```bash
python lunar_lander.py evaluate --episodes 10
```

脚本会新建带 `Monitor` 的环境，使用确定性动作评估 10 局，输出 `mean_reward`、`std_reward` 和 `mean_minus_std`。课程把 `mean_reward - std_reward` 用作挑战指标。短跑测试得到的分数不代表完整训练效果。

## 7. 录制一局回放

```bash
python lunar_lander.py record
```

脚本通过 `render_mode="rgb_array"` 与 `RecordVideo` 将一局保存为 `outputs/videos/` 下的 MP4。需要先完成训练，且录制依赖 `requirements.txt` 中的 `moviepy`。

## 8. 可选：上传到 Hugging Face Hub

上传是显式的独立步骤，不会在训练时自动发生。先完成 [Hugging Face 登录](https://huggingface.co/docs/huggingface_hub/en/guides/cli)：

```bash
hf auth login
python lunar_lander.py upload --repo-id 你的用户名/ppo-LunarLander-v3
```

该命令调用 `huggingface_sb3.package_to_hub`，上传模型，并生成评估结果、模型卡和回放。需要网络连接与自己账户的写入权限。若只想本地复现，到第 7 步即可。

## 文件

```text
Week1/
├── README.md             # 中文教学与操作步骤
├── requirements.txt      # Python 依赖
├── lunar_lander.py       # inspect/train/evaluate/record/upload 命令
└── outputs/              # 运行后生成，已忽略版本控制
```

参考资料：[课程 notebook](https://github.com/huggingface/deep-rl-class/blob/main/notebooks/unit1/unit1.ipynb)、[Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)、[SB3 并行环境](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html)、[Hugging Face SB3 集成](https://huggingface.co/docs/hub/en/stable-baselines3)。
