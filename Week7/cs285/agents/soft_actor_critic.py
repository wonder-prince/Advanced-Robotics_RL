"""Soft actor-critic used by the Week 7 MBPO experiment."""

from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

from cs285.infrastructure import pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        make_actor: Callable,
        make_actor_optimizer: Callable,
        make_actor_schedule: Callable,
        make_critic: Callable,
        make_critic_optimizer: Callable,
        make_critic_schedule: Callable,
        discount: float,
        actor_gradient_type: str = "reparametrize",
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        num_critic_networks: int = 2,
        target_critic_backup_type: str = "min",
        use_entropy_bonus: bool = True,
        backup_entropy: bool = True,
        temperature: float = 0.1,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
    ):
        super().__init__()
        if actor_gradient_type not in ("reinforce", "reparametrize"):
            raise ValueError("Invalid actor gradient type")
        if target_critic_backup_type not in ("mean", "min", "doubleq"):
            raise ValueError("Invalid target critic backup type")
        if num_critic_networks < 1 or num_actor_samples < 1 or num_critic_updates < 1:
            raise ValueError("Network, sample and update counts must be positive")
        if target_update_period is None and soft_target_update_rate is None:
            raise ValueError("Choose a hard or soft target update schedule")
        if soft_target_update_rate is not None and not 0 < soft_target_update_rate <= 1:
            raise ValueError("Soft target update rate must be in (0, 1]")

        self.actor = make_actor(observation_shape, action_dim)
        self.critics = nn.ModuleList(
            [make_critic(observation_shape, action_dim) for _ in range(num_critic_networks)]
        )
        self.target_critics = deepcopy(self.critics)
        for param in self.target_critics.parameters():
            param.requires_grad_(False)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.actor_schedule = make_actor_schedule(self.actor_optimizer)
        self.critic_schedule = make_critic_schedule(self.critic_optimizer)
        self.discount = discount
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.target_critic_backup_type = target_critic_backup_type
        self.use_entropy_bonus = use_entropy_bonus
        self.backup_entropy = backup_entropy
        self.temperature = temperature
        self.target_update_period = target_update_period
        self.soft_target_update_rate = soft_target_update_rate

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        batched_obs = ptu.from_numpy(np.asarray(obs, dtype=np.float32)[None])
        return ptu.to_numpy(self.actor(batched_obs).sample()[0])

    def critic(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(obs, actions) for critic in self.critics])

    @torch.no_grad()
    def target_critic(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(obs, actions) for critic in self.target_critics])

    def _backup_value(self, values: torch.Tensor) -> torch.Tensor:
        if self.target_critic_backup_type == "min":
            return values.min(dim=0).values
        if self.target_critic_backup_type == "mean":
            return values.mean(dim=0)
        if values.shape[0] < 2:
            raise ValueError("Double-Q backup requires at least two critics")
        return values[:2].min(dim=0).values

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        step: int,
    ) -> dict:
        obs = ptu.from_numpy(observations)
        acs = ptu.from_numpy(actions)
        rew = ptu.from_numpy(rewards).reshape(-1)
        next_obs = ptu.from_numpy(next_observations)
        done = ptu.from_numpy(dones).reshape(-1)

        for _ in range(self.num_critic_updates):
            with torch.no_grad():
                next_distribution = self.actor(next_obs)
                next_actions = next_distribution.sample()
                next_values = self._backup_value(self.target_critic(next_obs, next_actions))
                if self.use_entropy_bonus and self.backup_entropy:
                    next_values -= self.temperature * next_distribution.log_prob(next_actions)
                target = rew + self.discount * (1 - done) * next_values
            critic_loss = (self.critic(obs, acs) - target.unsqueeze(0)).square().mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        for param in self.critics.parameters():
            param.requires_grad_(False)
        distribution = self.actor(obs)
        actor_terms = []
        if self.actor_gradient_type == "reparametrize":
            for sample in distribution.rsample((self.num_actor_samples,)):
                q_value = self.critic(obs, sample).min(dim=0).values
                entropy_term = self.temperature * distribution.log_prob(sample) if self.use_entropy_bonus else 0
                actor_terms.append((entropy_term - q_value).mean())
        else:
            for _ in range(self.num_actor_samples):
                sample = distribution.sample()
                log_prob = distribution.log_prob(sample)
                q_value = self.critic(obs, sample).min(dim=0).values.detach()
                entropy_term = self.temperature * log_prob if self.use_entropy_bonus else 0
                actor_terms.append((log_prob * (entropy_term.detach() - q_value) + entropy_term).mean())
        actor_loss = torch.stack(actor_terms).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for param in self.critics.parameters():
            param.requires_grad_(True)

        if self.soft_target_update_rate is not None:
            rate = self.soft_target_update_rate
            with torch.no_grad():
                for target_param, param in zip(self.target_critics.parameters(), self.critics.parameters()):
                    target_param.mul_(1 - rate).add_(param, alpha=rate)
        elif step % self.target_update_period == 0:
            self.target_critics.load_state_dict(self.critics.state_dict())

        self.actor_schedule.step()
        self.critic_schedule.step()
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}
