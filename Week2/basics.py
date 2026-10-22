"""PPT 第 4–8 页：张量、自动微分、网络层、Dataset 和 DataLoader。"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def tensor_examples() -> None:
    print("\n=== 张量创建与属性 ===")
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    x_np = torch.from_numpy(np.array(data))
    print("列表创建:", x_data)
    print("NumPy 创建:", x_np)

    shape = (2, 3)
    random_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print("随机 / 全一 / 全零:", random_tensor, ones_tensor, zeros_tensor)
    print("形状 / 类型 / 设备:", x_data.shape, x_data.dtype, x_data.device)

    print("\n=== 索引、变形与拼接 ===")
    print("第一行:", random_tensor[0])
    print("第二列:", random_tensor[:, 1])
    print("转置:", random_tensor.transpose(1, 0))
    print("按行拼接:", torch.cat((random_tensor, ones_tensor)))
    print("按列拼接:", torch.cat((random_tensor, ones_tensor), dim=1))
    print("堆叠形状:", torch.stack((random_tensor, ones_tensor)).shape)
    print("增加维度:", random_tensor.unsqueeze(0).shape)
    print("展平:", random_tensor.flatten())
    print("float32 / bool:", ones_tensor.float().dtype, ones_tensor.bool().dtype)
    if torch.cuda.is_available():
        print("CUDA 张量设备:", ones_tensor.to("cuda:0").device)


def autograd_example() -> None:
    print("\n=== 自动微分 ===")
    x = torch.linspace(0.0, 2.0, steps=25, requires_grad=True)
    w = torch.rand_like(x)
    y = torch.sum(x * w)
    y.backward()
    print("x.grad:", x.grad)
    print("是否等于 w:", torch.allclose(x.grad, w))


class Network(nn.Module):
    """PPT 中的 1 -> 128 -> 256 -> 128 -> 1 全连接网络。"""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def network_examples() -> None:
    print("\n=== 神经网络层 ===")
    linear_layer = nn.Linear(32, 256)
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    leaky_relu = nn.LeakyReLU()
    gelu = nn.GELU()
    classifier = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    print("线性层:", linear_layer)
    print("激活函数:", relu, sigmoid, leaky_relu, gelu)
    print("分类器:", classifier)
    print("函数拟合网络的参数形状:")
    for parameter in Network().parameters():
        print(parameter.shape)


class FunctionDataset(Dataset):
    """按 PPT 示例，在每次索引时随机产生一个函数样本。"""

    def __init__(self, func, x_range=(-10.0, 10.0), n_samples=10_000) -> None:
        self.func = func
        self.x_range = x_range
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_val = (self.x_range[1] - self.x_range[0]) * np.random.rand() + self.x_range[0]
        y_val = self.func(x_val)
        return (
            torch.tensor([x_val], dtype=torch.float32),
            torch.tensor([y_val], dtype=torch.float32),
        )


def dataloader_example() -> None:
    print("\n=== Dataset / DataLoader ===")
    target_function = lambda x: x**2 + 2 * x + 1
    dataset = FunctionDataset(target_function, n_samples=20_000)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print("数据集长度:", len(dataset))
    print("一批数据形状:", x_batch.shape, y_batch.shape)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    tensor_examples()
    autograd_example()
    network_examples()
    dataloader_example()
