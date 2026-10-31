# 第 6 周：离线强化学习实践

依据《第6周-深度强化学习拓展-离线强化学习实践.pptx》整理。代码以 [Berkeley CS285 Fall 2023 HW5](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/f401a13ce9139d633c1d242c2557ae29506a7e05/hw5) 的 MIT 许可练习骨架为基础，补全课件中的算法 TODO 和离线到在线训练流程，并修正会阻断实验的轨迹采样与输出路径问题。原许可见 [`LICENSE`](LICENSE)。**本次没有安装依赖，也没有运行代码、训练或评估。**

## 内容对应

| PPT 实践 | 代码 |
| --- | --- |
| Pointmass Easy / Medium / Hard 迷宫 | `cs285/envs/pointmass.py` |
| Random 与 RND 探索、RND 误差归一化 | `cs285/agents/random_agent.py`、`rnd_agent.py`、`cs285/infrastructure/running_state.py` |
| DQN / Double DQN 与 CQL | `cs285/agents/dqn_agent.py`、`cql_agent.py` |
| AWAC 与 IQL | `cs285/agents/awac_agent.py`、`iql_agent.py` |
| 数据收集、离线训练、在线微调 | `cs285/scripts/run_hw5_explore.py`、`run_hw5_offline.py`、`run_hw5_finetune.py` |
| 基本实验配置 | `experiments/exploration/`、`experiments/offline/`、`experiments/finetuning/` |

RND 的 `rnd_bonus_normalization` 支持 `none`、`mean_std` 和 `zfilter`，对应 PPT 的三种对照。CQL 的 `cql_alpha`、离线数据集大小、AWAC/IQL 使用的数据集、微调时是否保留离线 Buffer，都能通过配置文件或 `--set KEY=VALUE` 调整。`--set` 不改变磁盘上的 YAML。

## 后续使用顺序

以下命令仅供以后运行时参考，**本次没有执行**。所有命令从 `Week6` 目录发起。`requirements.txt` 记录课件依赖版本，并补上 Pointmass 用到的 NumPy 与 NetworkX；没有执行安装。

### 1. 收集离线数据

```bash
python cs285/scripts/run_hw5_explore.py \
  --cfg experiments/exploration/pointmass_hard_random.yaml \
  --dataset_dir datasets

python cs285/scripts/run_hw5_explore.py \
  --cfg experiments/exploration/pointmass_hard_rnd.yaml \
  --dataset_dir datasets
```

可将配置路径中的 `hard` 改为 `easy` 或 `medium`。探索脚本保存 `datasets/<dataset_name>.pkl` 和状态覆盖图。RND 归一化与数据量对照示例：

```bash
python cs285/scripts/run_hw5_explore.py \
  --cfg experiments/exploration/pointmass_hard_rnd.yaml --dataset_dir datasets \
  --set rnd_bonus_normalization=none --set total_steps=20000 \
  --set dataset_name=pointmass_hard_rnd_20k_none
```

将 `none` 改为 `mean_std`、`zfilter` 可比较归一化方式；将 `total_steps` 与 `dataset_name` 一起改为 10000、20000、50000、200000 对应数据量消融。

### 2. 离线训练

```bash
python cs285/scripts/run_hw5_offline.py \
  --cfg experiments/offline/pointmass_hard_dqn.yaml --dataset_dir datasets
python cs285/scripts/run_hw5_offline.py \
  --cfg experiments/offline/pointmass_hard_cql.yaml --dataset_dir datasets
python cs285/scripts/run_hw5_offline.py \
  --cfg experiments/offline/pointmass_hard_awac.yaml --dataset_dir datasets
python cs285/scripts/run_hw5_offline.py \
  --cfg experiments/offline/pointmass_hard_iql.yaml --dataset_dir datasets
```

若使用上面另存的数据集，在命令末尾加 `--set dataset_name=pointmass_hard_rnd_20k_none`。CQL α 消融可加 `--set cql_alpha=0.2` 等；`α=0` 对应无 CQL 惩罚的消融。离线训练会保存 `data/<实验目录>/final_agent.pt`。PPT 中的曲线和性能表是课件结果，本目录没有生成新的实验结果。

### 3. 离线预训练后在线微调

```bash
python cs285/scripts/run_hw5_finetune.py \
  --cfg experiments/finetuning/pointmass_hard_cql_finetune.yaml \
  --dataset_dir datasets
```

该入口先用离线数据训练 `offline_steps`，再创建新的在线 Replay Buffer，按当前策略收集数据并继续更新。默认清空旧 Buffer；加 `--retain_offline_buffer` 可做 PPT 中的保留数据对照。目录中也提供 DQN、AWAC、IQL 的 Hard 微调配置。保存的权重为 `offline_agent.pt` 和 `finetuned_agent.pt`。

## 输入与限制

- 离线训练与微调读取本目录探索脚本生成的 `datasets/*.pkl`；PPT 没有附上这些数据，因此没有伪造数据文件。
- `datasets/`、实验日志、覆盖图和权重已在 `Week6/.gitignore` 排除。
- 课件的依赖版本较旧，实际运行需要匹配的 Python 环境；本次未验证运行兼容性或训练效果。
