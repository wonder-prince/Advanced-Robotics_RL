"""Continuous-action policy used for behavior cloning and DAgger."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(input_size: int, output_size: int, n_layers: int = 2, size: int = 64) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_size
    for _ in range(n_layers):
        layers.extend((nn.Linear(width, size), nn.Tanh()))
        width = size
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class MLPPolicySL(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        n_layers: int = 2,
        size: int = 64,
        learning_rate: float = 5e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.mean_net = build_mlp(ob_dim, ac_dim, n_layers, size)
        self.logstd = nn.Parameter(torch.zeros(ac_dim))
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Sample a differentiable action from N(mean_net(observation), exp(logstd))."""
        mean = self.mean_net(observation)
        std = torch.exp(self.logstd)
        return Normal(mean, std).rsample()

    def update(self, observations: np.ndarray, actions: np.ndarray) -> dict[str, float]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        predicted = self.forward(obs)
        loss = self.loss(predicted, target)
        loss.backward()
        self.optimizer.step()
        return {"Training Loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = self.mean_net(tensor) if deterministic else self.forward(tensor)
        return action.cpu().numpy()

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)
