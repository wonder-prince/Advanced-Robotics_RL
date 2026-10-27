"""LunarLander-v3 PPO：Actor/Critic、GAE、裁剪目标与价值损失。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


ENV_ID = "LunarLander-v3"
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "outputs" / "lunar_ppo.pt"


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 500_000
    num_envs: int = 4
    rollout_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 1

    def validate(self) -> None:
        batch_size = self.num_envs * self.rollout_steps
        if min(
            self.total_timesteps,
            self.num_envs,
            self.rollout_steps,
            self.num_minibatches,
            self.update_epochs,
        ) <= 0:
            raise ValueError("步数、环境数、minibatch 数和更新轮数都必须大于 0")
        if batch_size % self.num_minibatches:
            raise ValueError("num_envs × rollout_steps 必须能被 num_minibatches 整除")
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gamma 和 gae_lambda 必须在 [0, 1] 内")


def init_layer(layer: nn.Linear, std: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """课件中的两个独立 8→64→64 网络。"""

    def __init__(self, obs_dim: int = 8, action_dim: int = 4):
        super().__init__()
        self.actor = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def get_action_and_value(
        self, observation: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = Categorical(logits=self.actor(observation))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.get_value(observation),
        )


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 rollout 末端向前递推，张量形状均为 [步数, 环境数]。"""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        following_value = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * following_value * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def inspect_environment(seed: int) -> None:
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(seed=seed)
        print(f"环境：{ENV_ID}")
        print(f"观测空间：{env.observation_space}；初始状态：{observation}")
        print(f"动作空间：{env.action_space}；0=不点火，1=左，2=主引擎，3=右")
    finally:
        env.close()


def train(config: PPOConfig, model_path: Path, log_dir: Path, device: torch.device) -> None:
    from torch.utils.tensorboard import SummaryWriter

    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    envs = [gym.make(ENV_ID) for _ in range(config.num_envs)]
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for index, env in enumerate(envs):
            env.action_space.seed(config.seed + index)
        observations = np.stack(
            [env.reset(seed=config.seed + index)[0] for index, env in enumerate(envs)]
        )
        if observations.shape[1:] != (8,) or envs[0].action_space.n != 4:
            raise ValueError("本实现需要 8 维观测和 4 个离散动作")

        agent = ActorCritic().to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        shape = (config.rollout_steps, config.num_envs)
        obs_buf = torch.zeros((*shape, 8), device=device)
        action_buf = torch.zeros(shape, dtype=torch.long, device=device)
        logprob_buf = torch.zeros(shape, device=device)
        reward_buf = torch.zeros(shape, device=device)
        done_buf = torch.zeros(shape, device=device)
        value_buf = torch.zeros(shape, device=device)
        episode_returns = np.zeros(config.num_envs, dtype=np.float64)
        recent_returns: deque[float] = deque(maxlen=100)

        batch_size = config.rollout_steps * config.num_envs
        minibatch_size = batch_size // config.num_minibatches
        updates = math.ceil(config.total_timesteps / batch_size)
        global_steps = 0

        for update in range(1, updates + 1):
            # 课件中的线性学习率退火。
            fraction = 1.0 - (update - 1) / updates
            optimizer.param_groups[0]["lr"] = config.learning_rate * fraction

            for step in range(config.rollout_steps):
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                obs_buf[step] = obs_tensor
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(obs_tensor)
                action_buf[step] = actions
                logprob_buf[step] = logprobs
                value_buf[step] = values

                next_observations = []
                for index, (env, action) in enumerate(zip(envs, actions.cpu().tolist())):
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    episode_returns[index] += float(reward)
                    training_reward = float(reward)
                    if truncated and not terminated:
                        # 时间限制不是 MDP 终止：在重置前用最终状态价值 bootstrap。
                        with torch.no_grad():
                            final_obs = torch.as_tensor(
                                next_obs, dtype=torch.float32, device=device
                            ).unsqueeze(0)
                            training_reward += config.gamma * agent.get_value(final_obs).item()
                    reward_buf[step, index] = training_reward
                    done_buf[step, index] = float(terminated or truncated)
                    if terminated or truncated:
                        recent_returns.append(float(episode_returns[index]))
                        writer.add_scalar("charts/episodic_return", episode_returns[index], global_steps + index + 1)
                        episode_returns[index] = 0.0
                        next_obs, _ = env.reset()
                    next_observations.append(next_obs)
                observations = np.stack(next_observations)
                global_steps += config.num_envs

            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                next_value = agent.get_value(next_obs_tensor)
                advantages, returns = compute_gae(
                    reward_buf, done_buf, value_buf, next_value, config.gamma, config.gae_lambda
                )

            flat_obs = obs_buf.reshape(batch_size, 8)
            flat_actions = action_buf.reshape(batch_size)
            flat_old_logprobs = logprob_buf.reshape(batch_size)
            flat_old_values = value_buf.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)

            for _ in range(config.update_epochs):
                for indices in np.random.permutation(batch_size).reshape(
                    config.num_minibatches, minibatch_size
                ):
                    mb = torch.as_tensor(indices, dtype=torch.long, device=device)
                    _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                        flat_obs[mb], flat_actions[mb]
                    )
                    ratio = (new_logprobs - flat_old_logprobs[mb]).exp()
                    mb_advantages = flat_advantages[mb]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                    policy_loss = torch.maximum(
                        -mb_advantages * ratio,
                        -mb_advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef),
                    ).mean()
                    value_loss_unclipped = (new_values - flat_returns[mb]).square()
                    clipped_values = flat_old_values[mb] + (
                        new_values - flat_old_values[mb]
                    ).clamp(-config.clip_coef, config.clip_coef)
                    value_loss_clipped = (clipped_values - flat_returns[mb]).square()
                    value_loss = 0.5 * torch.maximum(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_mean
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()

            writer.add_scalar("losses/policy", policy_loss.item(), global_steps)
            writer.add_scalar("losses/value", value_loss.item(), global_steps)
            writer.add_scalar("losses/entropy", entropy_mean.item(), global_steps)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_steps)
            if update % 10 == 0 or update == updates:
                mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
                print(
                    f"update={update}/{updates} steps={global_steps} "
                    f"recent_return={mean_return:.2f} loss={loss.item():.4f}"
                )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": agent.state_dict(),
                "config": asdict(config),
                "environment": ENV_ID,
                "global_steps": global_steps,
            },
            model_path,
        )
        print(f"模型已保存：{model_path}；实际交互步数：{global_steps}")
    finally:
        writer.close()
        for env in envs:
            env.close()


def load_agent(model_path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if checkpoint["environment"] != ENV_ID:
        raise ValueError(f"模型环境与 {ENV_ID} 不匹配")
    agent = ActorCritic().to(device)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.eval()
    return agent


def play_episode(env: gym.Env, agent: ActorCritic, seed: int, device: torch.device) -> float:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(agent.actor(obs_tensor).argmax(dim=-1).item())
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            return total_reward


def evaluate(model_path: Path, episodes: int, seed: int, device: torch.device) -> None:
    agent = load_agent(model_path, device)
    env = gym.make(ENV_ID)
    try:
        scores = [play_episode(env, agent, seed + index, device) for index in range(episodes)]
        print(
            json.dumps(
                {
                    "environment": ENV_ID,
                    "episodes": episodes,
                    "mean_reward": float(np.mean(scores)),
                    "std_reward": float(np.std(scores)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        env.close()


def record(model_path: Path, video_dir: Path, seed: int, device: torch.device) -> None:
    agent = load_agent(model_path, device)
    env = gym.wrappers.RecordVideo(
        gym.make(ENV_ID, render_mode="rgb_array"),
        video_folder=str(video_dir),
        episode_trigger=lambda episode: episode == 0,
        name_prefix="stage1-lunar-ppo",
    )
    try:
        score = play_episode(env, agent, seed, device)
        print(f"视频目录：{video_dir}；本局奖励：{score:.2f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--seed", type=int, default=1)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--total-timesteps", type=int, default=500_000)
    train_parser.add_argument("--num-envs", type=int, default=4)
    train_parser.add_argument("--rollout-steps", type=int, default=128)
    train_parser.add_argument("--num-minibatches", type=int, default=4)
    train_parser.add_argument("--update-epochs", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--log-dir", type=Path, default=HERE / "outputs" / "tensorboard")

    eval_parser = subparsers.add_parser("evaluate")
    record_parser = subparsers.add_parser("record")
    for name in ("train", "evaluate", "record"):
        subparsers.choices[name].add_argument("--model", type=Path, default=DEFAULT_MODEL)
        subparsers.choices[name].add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=123)
    record_parser.add_argument("--video-dir", type=Path, default=HERE / "outputs" / "videos")
    record_parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    if args.command == "inspect":
        inspect_environment(args.seed)
        return
    device = choose_device(args.device)
    if args.command == "train":
        config = PPOConfig(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        train(config, args.model, args.log_dir, device)
    elif args.command == "evaluate":
        if args.episodes <= 0:
            parser.error("--episodes 必须大于 0")
        evaluate(args.model, args.episodes, args.seed, device)
    else:
        record(args.model, args.video_dir, args.seed, device)


if __name__ == "__main__":
    main()
