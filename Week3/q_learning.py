"""课件中的表格 Q-Learning：FrozenLake-v1 与 Taxi-v3。"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Experiment:
    env_id: str
    train_episodes: int
    eval_episodes: int
    max_steps: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float


PRESETS = {
    "frozenlake": Experiment("FrozenLake-v1", 10000, 100, 99, 0.7, 0.95, 1.0, 0.05, 0.0005),
    "taxi": Experiment("Taxi-v3", 25000, 100, 99, 0.7, 0.95, 1.0, 0.05, 0.005),
}


def make_env(name: str) -> gym.Env:
    if name == "frozenlake":
        return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    return gym.make("Taxi-v3")


def train(name: str, episodes: int, seed: int, output_dir: Path) -> None:
    config = PRESETS[name]
    rng = np.random.default_rng(seed)
    env = make_env(name)
    try:
        env.action_space.seed(seed)
        q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
        recent_scores: deque[float] = deque(maxlen=100)

        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            score = 0.0
            epsilon = config.epsilon_min + (
                config.epsilon_start - config.epsilon_min
            ) * np.exp(-config.epsilon_decay * episode)

            for _ in range(config.max_steps):
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[state]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                # 真正的终止状态没有后续回报；时间截断仍可估计下一状态价值。
                next_value = 0.0 if terminated else float(np.max(q_table[next_state]))
                td_target = float(reward) + config.gamma * next_value
                q_table[state, action] += config.learning_rate * (
                    td_target - q_table[state, action]
                )
                state = next_state
                score += float(reward)
                if terminated or truncated:
                    break

            recent_scores.append(score)
            if (episode + 1) % 1000 == 0 or episode + 1 == episodes:
                print(
                    f"episode={episode + 1:5d} epsilon={epsilon:.3f} "
                    f"mean_reward_100={np.mean(recent_scores):8.2f}"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "q_table.npy", q_table)
        (output_dir / "config.json").write_text(
            json.dumps(
                {"environment": name, "seed": seed, "config": asdict(config)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Q 表已保存到：{output_dir}")
    finally:
        env.close()


def evaluate(name: str, episodes: int, seed: int, output_dir: Path) -> None:
    metadata = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if metadata["environment"] != name:
        raise ValueError(f"Q 表属于 {metadata['environment']}，当前选择的是 {name}")
    config = Experiment(**metadata["config"])
    q_table = np.load(output_dir / "q_table.npy", allow_pickle=False)
    env = make_env(name)
    try:
        expected_shape = (env.observation_space.n, env.action_space.n)
        if q_table.shape != expected_shape:
            raise ValueError(f"Q 表维度应为 {expected_shape}，实际为 {q_table.shape}")
        scores = []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            score = 0.0
            for _ in range(config.max_steps):
                action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                if terminated or truncated:
                    break
            scores.append(score)
        print(f"{config.env_id}: {np.mean(scores):.2f} ± {np.std(scores):.2f} ({episodes} 回合)")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--env", choices=tuple(PRESETS), default="frozenlake")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = PRESETS[args.env]
    output_dir = args.output_dir or Path("outputs/q_learning") / args.env
    episodes = args.episodes if args.episodes is not None else (
        config.train_episodes if args.command == "train" else config.eval_episodes
    )
    if episodes <= 0:
        parser.error("--episodes 必须大于 0")
    if args.command == "train":
        train(args.env, episodes, args.seed, output_dir)
    else:
        evaluate(args.env, episodes, args.seed, output_dir)


if __name__ == "__main__":
    main()
