"""PPT 第 15 页：查看 MNIST 数据形状、范围和索引 42 的样本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_data import DEFAULT_DATA_DIR, load_mnist


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mnist_sample_42.png")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(
        args.data_dir, download=not args.no_download
    )
    print(f"training inputs: shape {x_train.shape}, dtype {x_train.dtype}")
    print(f"training outputs: shape {y_train.shape}, dtype {y_train.dtype}")
    print(f"validation / test: {x_val.shape} / {x_test.shape}")
    print(f"input range: ({x_train.min()}, {x_train.max()})")
    print(f"label range: ({y_train.min()}, {y_train.max()})")
    print(f"label of training example 42: {y_train[42]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(x_train[42].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y_train[42]}")
    plt.axis("off")
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"样本图已保存到 {args.output}")


if __name__ == "__main__":
    main()
