# 行为克隆与 DAgger

按第一份课件的项目结构实现了以下内容：

| 课件要求 | 文件 |
| --- | --- |
| 两层、每层 64 单元的 Tanh MLP，高斯连续动作，重参数化采样，MSE 更新 | `cs285/policies/MLP_policy.py` |
| 加载 CS285 格式的专家策略 `.pkl` | `cs285/policies/loaded_gaussian_policy.py` |
| 单条及批量轨迹采样 | `cs285/infrastructure/utils.py` |
| 存储并随机抽取状态、专家动作 | `cs285/infrastructure/replay_buffer.py` |
| BC 首轮训练、DAgger 收集学生轨迹并用专家重标注、评估和日志 | `cs285/scripts/run_hw1.py` |

代码兼容 Gym 的旧版四元组和新版五元组 `step` 返回格式。DAgger 只用学生动作与环境交互，存入缓冲区前将动作标签替换为专家在相同观测下给出的动作。实验输出写入 `data/`，由仓库 `.gitignore` 排除。

## 数据准备

课件所列专家文件和示教数据不在 PPT 内，需要自行放到下列位置，或在命令中指定其他路径：

```text
cs285/policies/experts/Ant.pkl
cs285/policies/experts/HalfCheetah.pkl
cs285/policies/experts/Hopper.pkl
cs285/policies/experts/Walker2d.pkl
cs285/expert_data/expert_data_Ant-v4.pkl
cs285/expert_data/expert_data_HalfCheetah-v4.pkl
cs285/expert_data/expert_data_Hopper-v4.pkl
cs285/expert_data/expert_data_Walker2d-v4.pkl
```

专家数据需为 CS285 的非空轨迹列表，每条轨迹至少有 `observation`、`action`、`reward` 数组。专家策略 `.pkl` 需为 CS285 的 `GaussianPolicy` 格式。不要把不可信 pickle 文件作为输入。

## 后续运行方式

以下仅作使用说明，**本次没有执行**。在 `Week5/imitation_learning` 下运行，将 `Ant` 换成其他课件任务名即可：

```bash
python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/Ant.pkl \
  --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
  --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
  --video_log_freq -1 --save_params

python cs285/scripts/run_hw1.py \
  --expert_policy_file cs285/policies/experts/Ant.pkl \
  --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
  --env_name Ant-v4 --exp_name dagger_ant --do_dagger --n_iter 10 \
  --video_log_freq -1 --save_params
```

`requirements.txt` 记录课件中的参考版本，没有安装。若以后启用视频记录，可将 `--video_log_freq` 设为正数，并让环境支持 `rgb_array` 渲染。输出的 TensorBoard 标量包括训练损失、评估回报均值与标准差、轨迹长度等。

本练习的结构和输入格式依据 [Berkeley CS285 Homework 1](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw1)；这里的实现已补全课件所列 TODO，并保留 [原项目 MIT 许可](https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/LICENSE)的来源说明。
