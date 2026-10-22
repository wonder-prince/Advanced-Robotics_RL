"""课件中的 PandaReachDense-v3：四环境 A2C 与 VecNormalize。"""

from __future__ import annotations

import argparse
from pathlib import Path

import panda_gym  # noqa: F401 - 导入时注册 PandaReachDense-v3
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize


ENV_ID = "PandaReachDense-v3"
DEFAULT_OUTPUT = Path("outputs/panda_a2c")


def train(timesteps: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(ENV_ID, n_envs=4, seed=seed)
    normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    try:
        model = A2C("MultiInputPolicy", normalized_env, verbose=1, seed=seed)
        model.learn(total_timesteps=timesteps)
        model.save(output_dir / "model")
        normalized_env.save(str(output_dir / "vecnormalize.pkl"))
        print(f"模型和归一化统计已保存到：{output_dir}")
    finally:
        normalized_env.close()


def evaluate(episodes: int, seed: int, output_dir: Path) -> None:
    model_path = output_dir / "model.zip"
    stats_path = output_dir / "vecnormalize.pkl"
    if not model_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("请先训练，确保 model.zip 和 vecnormalize.pkl 都存在")

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed)
    normalized_env = VecNormalize.load(str(stats_path), eval_env)
    normalized_env.training = False
    normalized_env.norm_reward = False
    try:
        model = A2C.load(model_path, env=normalized_env, device="auto")
        mean_reward, std_reward = evaluate_policy(
            model, normalized_env, n_eval_episodes=episodes, deterministic=True
        )
        print(f"{ENV_ID}: {mean_reward:.2f} ± {std_reward:.2f} ({episodes} 回合)")
    finally:
        normalized_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.episodes <= 0:
        parser.error("--timesteps 和 --episodes 必须大于 0")
    if args.command == "train":
        train(args.timesteps, args.seed, args.output_dir)
    else:
        evaluate(args.episodes, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
