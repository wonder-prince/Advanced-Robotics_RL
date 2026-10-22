# 本机复现结果

已使用 `uv` 创建 `Week1/.venv` 并安装 `requirements.txt`。环境检查、短跑训练、完整训练、独立评估和视频录制均成功。

## 运行配置

- Python 3.11.15；Gymnasium 1.3.0；Stable-Baselines3 2.9.0
- 环境：`LunarLander-v3`
- 算法：PPO + `MlpPolicy`，设备：CPU
- 训练：16 个环境，目标 1,000,000 步，实际 1,015,808 步（62 个完整 rollout）
- 随机种子：训练 42，评估及回放 123
- 训练日志显示约 335 秒；最后一次记录的最近 100 局平均奖励为 254

## 独立评估

在新建的 `Monitor` 环境中运行 10 局，使用确定性动作：

| 指标 | 结果 |
| --- | ---: |
| 平均奖励 | 263.43 |
| 奖励标准差 | 17.91 |
| 平均奖励 − 标准差 | 245.52 |

录制的一局奖励为 **273.33**。这些是本次运行结果；更换随机种子或依赖版本后可能不同。

## 生成文件

- [完整训练模型](outputs/ppo-LunarLander-v3.zip)
- [10 局评估数据](outputs/evaluation.json)
- [训练日志](outputs/train.log)
- [一局 MP4 回放](outputs/videos/ppo-lunarlander-v3-episode-0.mp4)

`outputs/` 和 `.venv/` 已列入 `.gitignore`，不会作为源码提交。上传 Hub 尚未执行。
