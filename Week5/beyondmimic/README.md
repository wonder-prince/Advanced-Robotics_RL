# BeyondMimic 运动追踪实践

本目录是 [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) 在提交 [`9db9158`](https://github.com/HybridRobotics/whole_body_tracking/commit/9db915838b17785c6fec90c91fe80b95b147e1bd) 的源码快照，保留其目录结构及 `LICENCE`。该仓库是 PPT 中介绍的 BeyondMimic **运动追踪训练**实现，采用 MIT 许可。这里不包含论文中另一个部分的 Guided Diffusion 训练代码，也不包含独立的 MuJoCo 真机部署项目。

## 与课件的对应关系

| 课件环节 | 代码 |
| --- | --- |
| 重定向后的 CSV 插值、正向运动学、速度计算、NPZ 与 WandB Registry | `scripts/csv_to_npz.py` |
| 从 Registry 下载并在 Isaac Sim 回放动作 | `scripts/replay_npz.py` |
| G1 执行器、阻抗、动作缩放 | `source/whole_body_tracking/whole_body_tracking/robots/g1.py` |
| 参考动作命令、自适应采样、奖励、观测、终止、扰动 | `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/` |
| 追踪环境与 G1 平地配置 | `source/whole_body_tracking/whole_body_tracking/tasks/tracking/` |
| RSL-RL PPO 参数、训练、评估和 ONNX 导出 | `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`、`scripts/rsl_rl/` |

## 课件给出的操作顺序

以下命令仅作后续参考，**本次没有执行或安装依赖**。在本目录执行，并替换花括号中的值：

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv \
  --input_fps 30 --output_name {motion_name} --headless

python scripts/replay_npz.py \
  --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}

python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}

python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
  --num_envs=2 --wandb_path={organization}/{project_name}/{run_id}
```

源码需要外部的 Isaac Sim 4.5.0、Isaac Lab 2.1.0、RSL-RL、WandB 账号和 Registry，还需要 G1 `unitree_description` 机器人文件及**已重定向到 G1 广义坐标**的动作 CSV。课件未提供这些资源，本目录只交付代码。原仓库 README 的[资源说明](https://github.com/HybridRobotics/whole_body_tracking/blob/9db915838b17785c6fec90c91fe80b95b147e1bd/README.md)可用于以后准备输入；LaFAN1 的原始 BVH 不能直接传给 `csv_to_npz.py`。

PPT 最后一环展示的 MuJoCo 部署使用另一仓库 [RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)，不属于这份运动追踪源码。导出的 ONNX 策略还需要部署侧的机器人模型、观测处理和控制参数才能使用。
