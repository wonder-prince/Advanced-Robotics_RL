"""PPT 第 14 页使用的 PyTorch 教程 MNIST 数据文件。"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "mnist"
DATA_URL = "https://raw.githubusercontent.com/pytorch/tutorials/main/_static/mnist.pkl.gz"
FILENAME = "mnist.pkl.gz"


def download_mnist(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """从 PyTorch 官方教程仓库下载一次，避免保存失败的半成品。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / FILENAME
    if destination.is_file():
        return destination

    temporary = data_dir / f"{FILENAME}.download"
    try:
        with requests.get(DATA_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        with gzip.open(temporary, "rb") as file:
            if file.read(2) == b"":
                raise ValueError("MNIST 下载文件为空")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_mnist(data_dir: Path = DEFAULT_DATA_DIR, download: bool = True):
    path = download_mnist(data_dir) if download else Path(data_dir) / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}；请先允许下载数据，或手动放置官方 mnist.pkl.gz")

    # 该文件使用 pickle 格式。只读取上方 PyTorch 官方教程来源，
    # 不要将本地文件替换为不可信的同名文件。
    with gzip.open(path, "rb") as file:
        (train, validation, test) = pickle.load(file, encoding="latin-1")

    def convert(split):
        images, labels = split
        return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    return convert(train), convert(validation), convert(test)
