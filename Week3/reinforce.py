"""课件中的 REINFORCE：CartPole-v1 与 FlappyBird-v0。"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class Experiment:
    env_id: str
    hidden_size: int
    hidden_layers: int
    train_episodes: int
    eval_episodes: int
    max_steps: int
    gamma: float
    learning_rate: float


PRESETS = {
    "cartpole": Experiment("CartPole-v1", 16, 1, 1000, 10, 1000, 1.0, 1e-2),
    "flappy": Experiment("FlappyBird-v0", 64, 2, 5000, 10, 1000, 0.99, 1e-4),
}


class Policy(nn.Module):
    """输入环境状态，输出各离散动作的概率。"""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = obs_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_features, hidden_size), nn.ReLU()))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: np.ndarray) -> Categorical:
        state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return Categorical(logits=self.network(state))


def make_env(name: str) -> gym.Env:
    if name == "flappy":
        import flappy_bird_gymnasium  # noqa: F401 - 导入时注册 FlappyBird-v0

    return gym.make(PRESETS[name].env_id)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    running_return = 0.0
    returns = []
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    values = torch.tensor(returns, dtype=torch.float32)
    # 单步回合的标准差为 0；保留零中心化后的结果即可。
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


def train(name: str, episodes: int, seed: int, output: Path) -> None:
    config = PRESETS[name]
    torch.manual_seed(seed)
    env = make_env(name)
    try:
        env.action_space.seed(seed)
        policy = Policy(
            env.observation_space.shape[0],
            env.action_space.n,
            config.hidden_size,
            config.hidden_layers,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        recent_scores: deque[float] = deque(maxlen=100)

        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode - 1)
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []

            for _ in range(config.max_steps):
                distribution = policy(observation)
                action = distribution.sample()
                observation, reward, terminated, truncated, _ = env.step(int(action.item()))
                log_probs.append(distribution.log_prob(action).squeeze())
                rewards.append(float(reward))
                if terminated or truncated:
                    break

            returns = discounted_returns(rewards, config.gamma)
            loss = -torch.stack(log_probs).dot(returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recent_scores.append(sum(rewards))
            if episode % 100 == 0 or episode == episodes:
                print(f"episode={episode:5d} mean_reward_100={np.mean(recent_scores):8.2f}")

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "environment": name,
                "config": asdict(config),
            },
            output,
        )
        print(f"模型已保存：{output}")
    finally:
        env.close()


def evaluate(name: str, episodes: int, seed: int, checkpoint: Path) -> None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["environment"] != name:
        raise ValueError(f"模型属于 {saved['environment']}，当前选择的是 {name}")
    config = Experiment(**saved["config"])
    env = make_env(name)
    try:
        policy = Policy(
            env.observation_space.shape[0],
            env.action_space.n,
            config.hidden_size,
            config.hidden_layers,
        )
        policy.load_state_dict(saved["state_dict"])
        policy.eval()
        scores = []
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                total_reward = 0.0
                for _ in range(config.max_steps):
                    action = policy(observation).sample().item()
                    observation, reward, terminated, truncated, _ = env.step(int(action))
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                scores.append(total_reward)
        print(f"{config.env_id}: {np.mean(scores):.2f} ± {np.std(scores):.2f} ({episodes} 回合)")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--env", choices=tuple(PRESETS), default="cartpole")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = PRESETS[args.env]
    checkpoint = args.checkpoint or Path("outputs/reinforce") / f"{args.env}.pt"
    episodes = args.episodes if args.episodes is not None else (
        config.train_episodes if args.command == "train" else config.eval_episodes
    )
    if episodes <= 0:
        parser.error("--episodes 必须大于 0")
    if args.command == "train":
        train(args.env, episodes, args.seed, checkpoint)
    else:
        evaluate(args.env, episodes, args.seed, checkpoint)


if __name__ == "__main__":
    main()
