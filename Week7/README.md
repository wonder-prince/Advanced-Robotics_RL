# 第 7 周：基于模型的强化学习实践

本目录对应《第7周-基于模型的强化学习实践.pptx》。代码以 Berkeley CS285 Homework 4 的 MIT 许可项目骨架为基础，补齐课件列出的动力学模型、MPC、CEM、MBPO 与 SAC 实践。原始项目：<https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw4>；许可见 [LICENSE](LICENSE)。

## 实现内容

- [model_based_agent.py](cs285/agents/model_based_agent.py)：集成模型分别训练；状态与动作输入、状态增量输出的标准化；预测反标准化；跨模型和规划时域的奖励估计；Random Shooting 与 CEM。
- [run_hw4.py](cs285/scripts/run_hw4.py)：随机策略收集初始数据、后续策略收集真实轨迹、分别采样训练每个动力学模型；从真实状态出发，用 SAC 与集成模型生成虚拟轨迹，加入 SAC 回放池；记录评估回报和损失。
- [soft_actor_critic.py](cs285/agents/soft_actor_critic.py)：SAC actor、多个 Q 网络、目标 Q 网络、熵奖励与软更新，用于 MBPO 对比。
- [experiments/mpc](experiments/mpc)：Obstacles、Reacher、HalfCheetah 的单次与多次迭代、超参数消融、CEM 配置。
- [experiments/sac](experiments/sac)：模型自由 SAC、长度为 1 的 MBPO 虚拟轨迹，以及长度为 5 的 Dyna 风格虚拟轨迹对比。后者是课件中“Dyna vs MBPO”的长短 rollout 对照，不代表通用 Dyna 算法的所有实现形式。

## 对应课件问题

| 问题 | 配置与输出 |
| --- | --- |
| 1 动力学模型训练 | 在 MPC 配置中调整 `learning_rate`、`num_layers`、`hidden_size`；首次迭代保存 `itr_0_loss_curve.png`，损失写入 TensorBoard |
| 2–3 单次 / 多次 MPC | `obstacles_1_iter.yaml`、`obstacles_multi_iter.yaml`、`reacher_multi_iter.yaml`、`halfcheetah_0_iter.yaml`、`halfcheetah_multi_iter.yaml` |
| 4 集成大小与规划时域消融 | 复制并修改 `reacher_ablation.yaml` 中的 `ensemble_size`、`mpc_horizon`、`mpc_num_action_sequences` |
| 5 CEM 与随机采样 | `halfcheetah_cem.yaml` 对比 `halfcheetah_multi_iter.yaml` |
| 6 MBPO / 长 rollout / 模型自由 SAC | `halfcheetah_mbpo.yaml` 搭配 `halfcheetah_clipq.yaml`、`halfcheetah_long_rollout.yaml` 或 `halfcheetah_model_free.yaml` |

在 `Week7` 目录中，命令形式如下；本次交付没有执行这些命令：

```bash
python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_multi_iter.yaml
python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_cem.yaml
python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
python cs285/scripts/run_hw4.py --config_file experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_model_free.yaml
```

运行前需要满足 [requirements.txt](requirements.txt) 中的依赖和 MuJoCo 环境。训练日志与曲线会写到 `Week7/data/`。本次没有安装依赖、启动训练或生成实验结果。
