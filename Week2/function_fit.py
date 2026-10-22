"""PPT 第 9–13 页：用 MLP 拟合 sin(x/2)+exp(-0.1x²)cos(5x)。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent


def target_function(x):
    return np.sin(x / 2) + np.exp(-0.1 * x**2) * np.cos(5 * x)


class FunctionDataset(Dataset):
    def __init__(self, x_range=(-10.0, 10.0), n_samples=20_000) -> None:
        self.x_range = x_range
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (self.x_range[1] - self.x_range[0]) * np.random.rand() + self.x_range[0]
        # PPT 中的噪声是 [0, 0.1) 的正向扰动，照原示例保留。
        y = target_function(x) + 0.1 * np.random.rand()
        return (
            torch.tensor([x], dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


class Network(nn.Module):
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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "function_fit")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.samples) <= 0 or args.learning_rate <= 0:
        parser.error("epochs、batch-size、samples 和 learning-rate 必须为正数")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    loader = DataLoader(FunctionDataset(n_samples=args.samples), batch_size=args.batch_size, shuffle=True)
    model = Network(hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {total_loss / args.samples:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "function_fit.pt")

    model.eval()
    x_test = torch.linspace(-10, 10, 1000).view(-1, 1).to(device)
    with torch.no_grad():
        y_pred = model(x_test).cpu().numpy().reshape(-1)
    x_np = x_test.cpu().numpy().reshape(-1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_np, target_function(x_np), label="True Function", linewidth=2)
    plt.plot(x_np, y_pred, label="Model Prediction", linestyle="--", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Fitting with Neural Network")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_dir / "function_fit.png", dpi=150)
    plt.close()
    print(f"模型和曲线已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
