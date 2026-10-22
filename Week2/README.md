# Week 2：深度学习基本原理实践

## 环境：只使用 uv

在 `Week2` 目录下执行：

```bash
cd Week2
uv sync
```

`uv` 会依据 `.python-version` 使用 Python 3.11，并在 `Week2/.venv` 创建虚拟环境。项目依赖写在 `pyproject.toml`；无需 conda，也无需手工安装 PyTorch。`uv sync` 第一次运行需要下载依赖。

## 按 PPT 顺序复现

| PPT 页码 | 内容 | 命令 |
| --- | --- | --- |
| 4–8 | 张量创建、索引、变形、自动微分、网络层、Dataset/DataLoader | `uv run python basics.py` |
| 9–13 | 拟合 `sin(x/2) + exp(-0.1x²)cos(5x)` | `uv run python function_fit.py` |
| 14–15 | 下载并查看 MNIST 数据与第 42 张图片 | `uv run python inspect_mnist.py` |
| 16–17 | MLP 分类器 | `uv run python mnist_models.py --model mlp` |
| 18 | CNN 分类器 | `uv run python mnist_models.py --model cnn` |
| 19 | LSTM 分类器 | `uv run python mnist_models.py --model lstm` |
| 扩展 | GRU 分类器 | `uv run python mnist_models.py --model gru` |

一次运行四个 MNIST 模型：

```bash
uv run python mnist_models.py --model all
```

## 默认配置与结果文件

函数拟合遵照 PPT：20,000 个动态采样点、批量 512、Adam 学习率 `1e-3`、200 轮。输出模型 `outputs/function_fit/function_fit.pt` 和曲线 `outputs/function_fit/function_fit.png`。PPT 的训练目标会加入 `[0, 0.1)` 的正向噪声，代码保留这一设定；图中“真实函数”仍为无噪声版本。

MNIST 使用 PPT 指向的 [PyTorch 教程数据文件](https://github.com/pytorch/tutorials/blob/main/_static/mnist.pkl.gz)，包含 50,000 张训练图、10,000 张验证图和 10,000 张测试图。脚本自动下载到 `data/mnist/`；训练使用前两部分，与 PPT 一致。测试集暂不参与训练或选型。数据文件采用 pickle 格式，代码只指向 PyTorch 官方仓库；请不要用来历不明的同名文件替换它。

MNIST 默认批量 64、验证批量 1000、Adam 学习率 `1e-3`、训练 3 轮。模型权重和每轮损失、验证准确率保存在 `outputs/mnist/`。PPT 截图中的具体损失值依赖随机初始化与运行环境，复现时无需期待逐位相同。

默认设备参数是 `--device auto`：优先 CUDA，其次 Apple MPS，最后 CPU。若希望只用 CPU，可以对训练命令追加 `--device cpu`。例如先缩短函数拟合训练作流程检查：

```bash
uv run python function_fit.py --epochs 2 --samples 1000 --device cpu
```

网络下载只发生在首次执行 MNIST 脚本时。离线运行可先把官方 `mnist.pkl.gz` 放到 `data/mnist/`，然后添加 `--no-download`。

## 文件结构

```text
Week2/
├── .python-version       # Python 3.11
├── pyproject.toml        # uv 项目和依赖
├── basics.py             # 张量、autograd、网络层和数据加载
├── function_fit.py       # 非线性函数拟合
├── mnist_data.py         # 下载并加载 PPT 使用的 MNIST 数据
├── inspect_mnist.py      # 数据探索和样本可视化
├── mnist_models.py       # MLP、CNN、LSTM、GRU
├── data/                 # 首次运行后生成
└── outputs/              # 训练后生成
```
