from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        if cql_temperature <= 0:
            raise ValueError("cql_temperature must be positive")
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        qa_values = variables["qa_values"]
        q_values = variables["q_values"]
        penalty = (
            self.cql_temperature * torch.logsumexp(qa_values / self.cql_temperature, dim=-1)
            - q_values
        ).mean()
        loss = loss + self.cql_alpha * penalty
        metrics["cql_penalty"] = penalty.detach().item()
        metrics["critic_loss"] = loss.detach().item()

        return loss, metrics, variables
