# Week8 GPU training results

The three Unity Linux players were run under Ubuntu 22.04 on WSL 2 with an RTX 4060 Laptop GPU. Python dependencies were installed with uv from the Aliyun mirror. PyTorch reported `2.5.1+cu124` and `torch.cuda.is_available() == True`.

| Experiment | Goal | Status | Last logged training reward |
| --- | ---: | --- | ---: |
| SnowballTarget PPO | 200,000 steps | Complete; ONNX exported | 25.074 at 200,000 steps |
| Pyramids PPO + RND | 1,000,000 steps | Training in progress | See WSL run directory |
| SoccerTwos MA-POCA | 5,000,000 steps | Training in progress | See WSL run directory |

The reward is the trainer's mean cumulative reward over recent completed episodes, not an independent test score. The [training curves](./overview.png) and [scalar data](./metrics.json) are generated from the TensorBoard event files by [build_results.py](./build_results.py). Regenerate them with:

```bash
uv run --project Week6 --frozen --no-sync --offline python results/week8/full/build_results.py
```

The completed SnowballTarget run contains its trainer configuration, TensorBoard events, PyTorch checkpoint, exported ONNX model. The final checkpoint records 200,112 steps and a reward of 25.818. [verify_models.py](./verify_models.py) passed `onnx.checker.check_model` on the exported model. The three Unity archives were checked against the SHA256 values in [setup-wsl.sh](../../../Week8/setup-wsl.sh) before extraction. Each experiment also completed a 20,000-step CUDA smoke run and exported ONNX before formal training started.

Run locations in WSL:

```text
/root/Week8/results/snowball-gpu-200k
/root/Week8/results/pyramids-gpu-1m
/root/Week8/results/soccer-gpu-5m
```

To resume an interrupted formal run, use the same `--run-id` with `--resume` in [train.py](../../../Week8/train.py). Copy completed run directories into this folder, then rerun [build_results.py](./build_results.py).
