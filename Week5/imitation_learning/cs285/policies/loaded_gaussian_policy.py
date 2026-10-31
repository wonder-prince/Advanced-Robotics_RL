"""Read the Berkeley CS285 Gaussian expert pickle used by the Week 5 slides."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _linear(layer: dict) -> nn.Linear:
    params = layer["AffineLayer"]
    weights = np.asarray(params["W"], dtype=np.float32)
    bias = np.asarray(params["b"], dtype=np.float32).reshape(-1)
    # The expert pickle stores each weight matrix as [input, output].
    linear = nn.Linear(weights.shape[0], weights.shape[1])
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(weights.T))
        linear.bias.copy_(torch.from_numpy(bias))
    return linear


class LoadedGaussianPolicy(nn.Module):
    def __init__(self, filename: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        with open(filename, "rb") as stream:
            data = pickle.load(stream)
        params = data["GaussianPolicy"]
        norm = params["obsnorm"]["Standardizer"]
        mean = np.asarray(norm["mean_1_D"], dtype=np.float32)
        mean_sq = np.asarray(norm["meansq_1_D"], dtype=np.float32)
        self.register_buffer("obs_mean", torch.from_numpy(mean))
        self.register_buffer("obs_std", torch.from_numpy(np.sqrt(np.maximum(0, mean_sq - mean**2))))
        nonlinearity = {"tanh": nn.Tanh, "lrelu": lambda: nn.LeakyReLU(0.01)}[data["nonlin_type"]]
        hidden = params["hidden"]["FeedforwardNet"]
        self.hidden_layers = nn.ModuleList(_linear(hidden[name]) for name in sorted(hidden))
        self.activation = nonlinearity()
        self.output_layer = _linear(params["out"])
        self.to(self.device)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Original CS285 loader pads/truncates an observation if its width differs.
        width = self.obs_mean.shape[-1]
        if observation.shape[-1] < width:
            observation = nn.functional.pad(observation, (0, width - observation.shape[-1]))
        else:
            observation = observation[..., :width]
        hidden = (observation - self.obs_mean) / (self.obs_std + 1e-6)
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        return self.output_layer(hidden)

    @torch.no_grad()
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.forward(tensor).cpu().numpy()
