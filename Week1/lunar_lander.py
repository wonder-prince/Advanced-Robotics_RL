"""Hugging Face Deep RL Course Unit 1 的本地 LunarLander PPO 复现。"""

from __future__ import annotations

if __name__ == "__main__":
    print("正在加载 Gymnasium、PyTorch 和训练库，请稍候…", flush=True)

import argparse
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


ENV_ID = "LunarLander-v3"
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "outputs" / "ppo-LunarLander-v3.zip"


def inspect_environment(seed: int) -> None:
    """对应课程的观察状态、动作与交互环节。"""
    env = gym.make(ENV_ID)
    try:
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        print(f"环境: {ENV_ID}")
        print(f"观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        print(f"初始 8 维状态: {observation}")
        print("动作: 0=不点火, 1=左侧引擎, 2=主引擎, 3=右侧引擎")
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        print(f"随机动作 {action} -> 奖励 {reward:.3f}, 结束 {terminated or truncated}")
        print(f"下一状态: {next_observation}")
    finally:
        env.close()


def train(args: argparse.Namespace) -> None:
    if args.timesteps <= 0 or args.n_envs <= 0 or args.rollout_steps <= 0:
        raise ValueError("timesteps、n-envs 和 rollout-steps 必须是正整数")
    if args.batch_size <= 0 or args.batch_size > args.n_envs * args.rollout_steps:
        raise ValueError("batch-size 必须在 1 和 n-envs × rollout-steps 之间")

    print(f"依赖已加载，正在创建 {args.n_envs} 个环境…", flush=True)
    env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=args.seed)
    try:
        # 与课程解答一致；MlpPolicy 接收 LunarLander 的 8 维向量观测。
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        print("环境和 PPO 模型已就绪，开始收集经验并训练…", flush=True)
        model.learn(total_timesteps=args.timesteps)
        model_path = Path(args.model).expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"模型已保存: {model_path}")
        print(f"实际训练步数: {model.num_timesteps}（PPO 按完整 rollout 收集数据）")
    finally:
        env.close()


def load_model(path: str) -> PPO:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型: {model_path}；请先运行 train")
    return PPO.load(model_path, device="cpu")


def evaluate(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes 必须是正整数")
    model = load_model(args.model)
    eval_env = Monitor(gym.make(ENV_ID))
    try:
        eval_env.reset(seed=args.seed)
        mean, std = evaluate_policy(
            model, eval_env, n_eval_episodes=args.episodes, deterministic=True
        )
        result = {
            "environment": ENV_ID,
            "episodes": args.episodes,
            "mean_reward": float(mean),
            "std_reward": float(std),
            "mean_minus_std": float(mean - std),
            "seed": args.seed,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        eval_env.close()


def record(args: argparse.Namespace) -> None:
    """生成一局完整的 MP4 回放，不依赖 Colab 的虚拟屏幕。"""
    model = load_model(args.model)
    video_dir = Path(args.video_dir).expanduser().resolve()
    env = gym.wrappers.RecordVideo(
        gym.make(ENV_ID, render_mode="rgb_array"),
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix="ppo-lunarlander-v3",
    )
    try:
        observation, info = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        print(f"回放目录: {video_dir}")
        print(f"本局奖励: {total_reward:.2f}")
    finally:
        env.close()


def upload(args: argparse.Namespace) -> None:
    """显式调用才上传；需要先用 hf auth login 登录。"""
    if "/" not in args.repo_id:
        raise ValueError("repo-id 应为 用户名/仓库名")
    from huggingface_sb3 import package_to_hub

    model = load_model(args.model)
    # huggingface_sb3 用向量化环境评估并录制视频。
    eval_env = DummyVecEnv(
        [lambda: Monitor(gym.make(ENV_ID, render_mode="rgb_array"))]
    )
    try:
        package_to_hub(
            model=model,
            model_name=Path(args.model).stem,
            model_architecture="PPO",
            env_id=ENV_ID,
            eval_env=eval_env,
            repo_id=args.repo_id,
            commit_message="Upload PPO LunarLander-v3 agent",
        )
    finally:
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="查看状态、动作及一步交互")
    inspect_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="训练并保存 PPO 模型")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000)
    train_parser.add_argument("--n-envs", type=int, default=16)
    train_parser.add_argument("--rollout-steps", type=int, default=1024)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL))

    eval_parser = subparsers.add_parser("evaluate", help="独立环境中评估 10 局")
    eval_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=123)

    record_parser = subparsers.add_parser("record", help="录制一局 MP4")
    record_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    record_parser.add_argument("--video-dir", default=str(HERE / "outputs" / "videos"))
    record_parser.add_argument("--seed", type=int, default=123)

    upload_parser = subparsers.add_parser("upload", help="上传模型、评估与回放到 Hugging Face Hub")
    upload_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    upload_parser.add_argument("--repo-id", required=True)

    args = parser.parse_args()
    {
        "inspect": lambda: inspect_environment(args.seed),
        "train": lambda: train(args),
        "evaluate": lambda: evaluate(args),
        "record": lambda: record(args),
        "upload": lambda: upload(args),
    }[args.command]()


if __name__ == "__main__":
    main()
