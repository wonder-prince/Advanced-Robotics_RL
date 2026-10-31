"""Online normalization for RND exploration bonuses."""

import numpy as np


class ZFilter:
    """Exponentially weighted z-score of scalar RND errors."""

    def __init__(self, decay: float = 0.99, epsilon: float = 1e-8):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.epsilon = epsilon
        self.mean = None
        self.variance = None

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = float(values.mean())
        batch_variance = float(values.var())
        if self.mean is None:
            self.mean, self.variance = batch_mean, batch_variance
        else:
            delta = batch_mean - self.mean
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.variance = (
                self.decay * self.variance
                + (1 - self.decay) * batch_variance
                + self.decay * (1 - self.decay) * delta**2
            )
        return ((values - self.mean) / np.sqrt(self.variance + self.epsilon)).astype(np.float32)
