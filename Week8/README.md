# Week8 多智能体强化学习实践

按[课件](./第8周-多智能体强化学习实践.pptx)使用已有 Unity 构建训练三个行为。训练器参数保存在 `config/`。

| 实验 | 方法 | 正式训练步数 |
| --- | --- | ---: |
| [SnowballTarget](./config/ppo/SnowballTarget.yaml) | PPO | 200,000 |
| [Pyramids](./config/ppo/PyramidsRND.yaml) | PPO + RND | 1,000,000 |
| [SoccerTwos](./config/poca/SoccerTwos.yaml) | MA-POCA + 自博弈 | 5,000,000 |

## Windows WSL 2 快速配置

在 Intel 或 AMD 的 Windows 电脑上，用 PowerShell 执行 `wsl -l -v` 确认 Ubuntu 使用 WSL 2。在 Ubuntu 中运行 `uname -m`，应输出 `x86_64`。把 Week8 目录放到 WSL 的 Linux 文件系统，例如 `~/Week8`。若使用准备好的 ZIP，从 WSL 将其解压到主目录即可。

```bash
cd ~/Week8
bash setup-wsl.sh
```

脚本会安装 Unity 系统依赖，校验并解压三个 Unity 构建，创建 Python 3.10.12 的 CUDA 环境并检查训练器。Python 包和 uv 使用阿里云镜像。已有下载会复用。首次安装 Python 与依赖需要联网，安装系统包需要 sudo。配置成功后，在同一目录运行短训：

```bash
.venv/bin/python train.py snowball --env builds/linux/SnowballTarget/SnowballTarget.x86_64 --run-id snowball-wsl-smoke --smoke --torch-device cuda
.venv/bin/python train.py pyramids --env builds/linux/Pyramids/Pyramids --run-id pyramids-wsl-smoke --smoke --torch-device cuda
```

日志出现 `Connected to Unity environment` 且训练步数持续增长表示连通。若重跑，换一个 `--run-id`。

## 配置训练器

```bash
cd Week8
uv python install 3.10.12
uv sync --locked
uv run --locked mlagents-learn --help
for experiment in snowball pyramids soccer; do
  uv run --locked train.py "$experiment" --check || break
done
```

本项目锁定 ML-Agents 1.1.0 和 PyTorch 2.5.1。Apple Silicon 使用上游较新版本允许的 grpcio 1.53.2 wheel。旧版训练器还依赖 `pkg_resources`，因此保留包含该模块的 Setuptools 版本。

## 准备 Linux x86-64 构建

将可执行文件和同目录的数据、共享库一起保留。SnowballTarget 和 Pyramids 使用课件提供的构建。SoccerTwos 使用课件配套的完整构建，或从 [Unity ML-Agents release_22 的 SoccerTwos 场景](https://github.com/Unity-Technologies/ml-agents/blob/release_22/Project/Assets/ML-Agents/Examples/Soccer/Scenes/SoccerTwos.unity)构建 Linux x86-64 Player。构建时将 Behavior Type 设为 `Default`，并保留行为名 `SoccerTwos`。

```bash
mkdir -p builds/archives builds/linux results
curl -fL --retry 3 -o builds/archives/SnowballTarget.zip \
  https://media.githubusercontent.com/media/huggingface/Snowball-Target/1bc2647d32c4f83dbfacf53af57cb5e44896386a/SnowballTarget.zip
curl -fL --retry 3 -o builds/archives/Pyramids.zip \
  https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/main/Pyramids.zip
unzip -o builds/archives/SnowballTarget.zip -d builds/linux
unzip -o builds/archives/Pyramids.zip -d builds/linux
chmod +x builds/linux/SnowballTarget/SnowballTarget.x86_64 builds/linux/Pyramids/Pyramids
```

也可以下载完整的 SoccerTwos Linux x86-64 训练压缩包：

```bash
curl -fL --retry 3 -o builds/archives/SoccerTwos.zip \
  'https://drive.usercontent.google.com/download?id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL&export=download&confirm=t'
mkdir -p builds/linux/SoccerTwos
unzip -o builds/archives/SoccerTwos.zip -d builds/linux/SoccerTwos
chmod +x builds/linux/SoccerTwos/SoccerTwos.x86_64
```

## 短训与正式训练

在 Linux x86-64 环境中执行以下命令。`--smoke` 将训练目标设为 20,000 步，并每 2,000 步打印进度。已提交的采样批次可能让最终检查点略超目标步数。正式训练时去掉 `--smoke` 并换一个 `--run-id`。运行多个实验时，使用不同 `--base-port`。

```bash
uv run --locked train.py snowball --env builds/linux/SnowballTarget/SnowballTarget.x86_64 --run-id snowball-smoke --smoke --torch-device cuda
uv run --locked train.py pyramids --env builds/linux/Pyramids/Pyramids --run-id pyramids-smoke --smoke --torch-device cuda
uv run --locked train.py soccer --env builds/linux/SoccerTwos/SoccerTwos.x86_64 --run-id soccer-smoke --smoke --torch-device cuda
uv run --locked tensorboard --logdir results
```

### macOS 原生 SoccerTwos

```bash
mkdir -p builds/archives builds/macos
curl -fL --retry 3 -o builds/archives/SoccerTwos-macos.zip \
  'https://drive.usercontent.google.com/download?id=1PEF4snwfzj__bS3GBemLN8MmNooXX20o&export=download&confirm=t'
ditto -x -k builds/archives/SoccerTwos-macos.zip builds/macos
uv run --locked train.py soccer --env builds/macos/SoccerTwos.app --run-id soccer-macos-smoke --smoke
```

### Linux x86-64 容器

在原生 Linux x86-64 上准备 Unity 构建后，可用下列命令构建镜像并训练。Apple Silicon 上尝试通过 Rosetta 和 QEMU 运行 Linux Unity 构建时，遇到了 Mono 初始化崩溃或握手超时。

```bash
docker build --platform linux/amd64 -t week8-mlagents .
docker run --rm --platform linux/amd64 \
  -v "$PWD/builds/linux:/lab/builds/linux:ro" -v "$PWD/results:/lab/results" \
  week8-mlagents pyramids --env builds/linux/Pyramids/Pyramids --run-id pyramids-smoke --smoke
```

先确认 Unity 已连接且训练步数持续增长，再检查 `results/<run-id>/` 下的 TensorBoard 日志和 ONNX。Pyramids 另看 RND 的内在奖励，SoccerTwos 另看自博弈指标。`--num-envs` 控制 Unity 进程数，建议连通后再增加。

本机实测三个配置均可由 ML-Agents 1.1.0 加载。SoccerTwos 使用 Apple Silicon 原生构建跑过 20,000 步并导出 ONNX。SnowballTarget 和 Pyramids 的 Linux 构建已下载，仍需在原生 Linux x86-64 上确认 Unity 握手与短训。
