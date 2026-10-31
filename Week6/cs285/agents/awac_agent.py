from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
import numpy as np


from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure import pytorch_util as ptu


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))
        obs = ptu.from_numpy(np.asarray(observation, dtype=np.float32))[None]
        with torch.no_grad():
            return int(self.actor(obs).probs.argmax(dim=-1).item())

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            next_distribution = self.actor(next_observations)
            next_qa_values = self.target_critic(next_observations)
            next_qs = (next_distribution.probs * next_qa_values).sum(dim=-1)
            target_values = rewards.float().reshape(-1) + self.discount * (1 - dones.float().reshape(-1)) * next_qs

        qa_values = self.critic(observations)
        q_values = qa_values.gather(-1, actions.long().reshape(-1, 1)).squeeze(-1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        with torch.no_grad():
            qa_values = self.critic(observations)
            q_values = qa_values.gather(-1, actions.long().reshape(-1, 1)).squeeze(-1)
            if action_dist is None:
                action_dist = self.actor(observations)
            values = (action_dist.probs * qa_values).sum(dim=-1)
            return q_values - values

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        distribution = self.actor(observations)
        advantages = self.compute_advantage(observations, actions, distribution)
        weights = torch.exp((advantages / self.temperature).clamp(max=10.0)).clamp(max=100.0)
        loss = -(distribution.log_prob(actions.long().reshape(-1)) * weights).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
