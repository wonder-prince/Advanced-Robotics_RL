"""TensorBoard scalar and rollout video logging."""

from __future__ import annotations

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, value: float, name: str, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_paths_as_videos(self, paths: list[dict], step: int, fps: int = 30) -> None:
        frames = [path["image_obs"] for path in paths if len(path["image_obs"])]
        if not frames:
            return
        length = max(map(len, frames))
        padded = [np.concatenate((video, np.repeat(video[-1:], length - len(video), axis=0))) for video in frames]
        batch = np.stack(padded).transpose(0, 1, 4, 2, 3)
        self.writer.add_video("eval_rollouts", batch, step, fps=fps)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
