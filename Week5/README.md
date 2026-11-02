# Week 5：模仿学习与 BeyondMimic 实践

本目录对应《第5周-模仿学习原理与方法实践》和《第5周-beyondmimic实践》两份课件。模仿学习项目已准备好 uv 环境，并完成 Hopper 的 BC/DAgger 缩短训练与评估；BeyondMimic 仍需单独准备 Isaac Sim、Isaac Lab 和机器人资源。

| 子目录 | 内容 |
| --- | --- |
| [`imitation_learning/`](imitation_learning/README.md) | CS285 风格的 MuJoCo 行为克隆和 DAgger 练习 |
| [`beyondmimic/`](beyondmimic/README.md) | BeyondMimic 官方运动追踪代码快照及实践步骤 |

两部分相互独立：前者使用 Gym、PyTorch 和课件中的专家数据；后者使用 Isaac Sim、Isaac Lab、RSL-RL、WandB 与 G1 机器人资源。课件没有附上专家 `.pkl`、人体动作 CSV、G1 描述文件或训练权重，因此这些输入没有伪造或加入仓库。
