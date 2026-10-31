"""Transition replay buffer for the growing DAgger dataset."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int = 1_000_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.obs: np.ndarray | None = None
        self.acs: np.ndarray | None = None

    def __len__(self) -> int:
        return 0 if self.obs is None else len(self.obs)

    def add_rollouts(self, paths: list[dict]) -> None:
        if not paths:
            raise ValueError("paths cannot be empty")
        obs = np.concatenate([np.asarray(p["observation"], dtype=np.float32) for p in paths])
        acs = np.concatenate([np.asarray(p["action"], dtype=np.float32) for p in paths])
        if len(obs) != len(acs):
            raise ValueError("observations and expert actions must align")
        self.obs = obs[-self.max_size :] if self.obs is None else np.concatenate((self.obs, obs))[-self.max_size :]
        self.acs = acs[-self.max_size :] if self.acs is None else np.concatenate((self.acs, acs))[-self.max_size :]

    def sample(self, batch_size: int, rng: np.random.Generator):
        if not len(self):
            raise ValueError("buffer is empty")
        indices = rng.choice(len(self), size=min(batch_size, len(self)), replace=False)
        return self.obs[indices], self.acs[indices]
