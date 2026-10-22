"""PPT 第 16–19 页：相同 MNIST 数据上的 MLP、CNN 和 LSTM；另附 GRU 扩展。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_data import DEFAULT_DATA_DIR, load_mnist


ROOT = Path(__file__).resolve().parent


class MNISTMLP(nn.Module):
    """PPT 第 17 页：784 -> 64 -> 10。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 64)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.reshape(-1, 784)
        return self.linear2(self.activation(self.linear1(x)))


class MNISTCNN(nn.Module):
    """PPT 第 18 页：两层 5x5 卷积及池化，接 800 -> 500 -> 10。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images.reshape(-1, 1, 28, 28))


class MNISTLSTM(nn.Module):
    """PPT 第 19 页：把 28 行当作长度 28 的序列。"""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence = images.reshape(-1, 28, 28)
        output, _ = self.lstm(sequence)
        return self.linear(output[:, -1, :])


class MNISTGRU(nn.Module):
    """PPT 标题提到 GRU；这一实现是沿 LSTM 示例补充的扩展。"""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence = images.reshape(-1, 28, 28)
        output, _ = self.gru(sequence)
        return self.linear(output[:, -1, :])


MODELS = {"mlp": MNISTMLP, "cnn": MNISTCNN, "lstm": MNISTLSTM, "gru": MNISTGRU}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(data_dir: Path, batch_size: int, validation_batch_size: int, download: bool):
    (x_train, y_train), (x_val, y_val), _ = load_mnist(data_dir, download=download)
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    validation_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(validation_data, batch_size=validation_batch_size),
    )


def train_model(
    name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
) -> list[dict]:
    model = MODELS[name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = []
    print(f"\n=== Running {name.upper()} ===")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)

        model.eval()
        validation_loss_sum = 0.0
        correct = 0
        validation_count = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                validation_loss_sum += loss_function(logits, labels).item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                validation_count += images.size(0)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss_sum / validation_count,
            "validation_accuracy": correct / validation_count,
        }
        history.append(metrics)
        print(
            f"Epoch {epoch}/{epochs}, Validation Loss: {metrics['validation_loss']:.4f}, "
            f"Validation Accuracy: {metrics['validation_accuracy']:.2%}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"mnist_{name}.pt")
    (output_dir / f"mnist_{name}_metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mnist")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validation_batch_size) <= 0 or args.learning_rate <= 0:
        parser.error("epochs、batch-size、validation-batch-size 和 learning-rate 必须为正数")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    train_loader, validation_loader = make_loaders(
        args.data_dir, args.batch_size, args.validation_batch_size, not args.no_download
    )

    selected = list(MODELS) if args.model == "all" else [args.model]
    for name in selected:
        train_model(
            name,
            train_loader,
            validation_loader,
            args.epochs,
            args.learning_rate,
            device,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
