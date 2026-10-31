"""Train behavior cloning or DAgger on a continuous-control MuJoCo task."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import gym
import numpy as np
import torch

# Allow `python cs285/scripts/run_hw1.py` from the exercise directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure import utils
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy


ENV_NAMES = ("Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert_policy_file", type=Path)
    parser.add_argument("--expert_data", type=Path, required=True)
    parser.add_argument("--env_name", choices=ENV_NAMES, required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--do_dagger", action="store_true")
    parser.add_argument("--ep_len", type=int)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=5000)
    parser.add_argument("--train_batch_size", type=int, default=100)
    parser.add_argument("--num_agent_train_steps_per_iter", type=int, default=1000)
    parser.add_argument("--max_replay_buffer_size", type=int, default=1_000_000)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--save_params", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.do_dagger and args.n_iter < 2:
        parser.error("DAgger requires --n_iter >= 2")
    if not args.do_dagger and args.n_iter != 1:
        parser.error("behavior cloning uses --n_iter 1; add --do_dagger for more iterations")
    if args.do_dagger and args.expert_policy_file is None:
        parser.error("DAgger requires --expert_policy_file")
    if args.video_log_freq == 0 or args.scalar_log_freq <= 0:
        parser.error("logging frequencies must be positive, or use --video_log_freq -1")
    for name in ("batch_size", "eval_batch_size", "train_batch_size", "num_agent_train_steps_per_iter", "max_replay_buffer_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.ep_len is not None and args.ep_len <= 0:
        parser.error("--ep_len must be positive")
    return args


def load_expert_paths(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        paths = pickle.load(stream)
    if not isinstance(paths, list) or not paths:
        raise ValueError("expert_data must be a nonempty list of CS285 trajectory dictionaries")
    for trajectory in paths:
        if not {"observation", "action", "reward"} <= trajectory.keys():
            raise ValueError("each expert trajectory needs observation, action and reward")
    return paths


def run_training_loop(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = f"cuda:{args.which_gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("this exercise requires a continuous Box action space")
    episode_length = args.ep_len or env.spec.max_episode_steps
    if episode_length is None:
        raise ValueError("set --ep_len when the environment has no episode limit")

    actor = MLPPolicySL(
        ac_dim=int(np.prod(env.action_space.shape)),
        ob_dim=int(np.prod(env.observation_space.shape)),
        n_layers=args.n_layers,
        size=args.size,
        learning_rate=args.learning_rate,
        device=device,
    )
    expert = LoadedGaussianPolicy(args.expert_policy_file, device=device) if args.do_dagger else None
    buffer = ReplayBuffer(args.max_replay_buffer_size)
    logdir = args.output_dir / f"{'dagger' if args.do_dagger else 'bc'}_{args.exp_name}_{args.env_name}"
    logdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(logdir))
    total_envsteps = 0
    started = time.monotonic()

    try:
        for iteration in range(args.n_iter):
            if iteration == 0:
                paths = load_expert_paths(args.expert_data)
            else:
                # DAgger: visit states with the current student policy.
                paths, steps = utils.sample_trajectories(env, actor, args.batch_size, episode_length)
                total_envsteps += steps
                # Replace executed student actions with expert labels at *those* states.
                for trajectory in paths:
                    trajectory["action"] = expert.get_action(trajectory["observation"]).astype(np.float32)
            buffer.add_rollouts(paths)

            training_logs = []
            for _ in range(args.num_agent_train_steps_per_iter):
                ob_batch, ac_batch = buffer.sample(args.train_batch_size, rng)
                training_logs.append(actor.update(ob_batch, ac_batch))

            if iteration % args.scalar_log_freq == 0:
                eval_paths, _ = utils.sample_trajectories(env, actor, args.eval_batch_size, episode_length)
                metrics = utils.compute_metrics(paths, eval_paths)
                metrics.update(training_logs[-1])
                metrics["Train_EnvstepsSoFar"] = total_envsteps
                metrics["TimeSinceStart"] = time.monotonic() - started
                for key, value in metrics.items():
                    print(f"{iteration} {key}: {value:.4f}")
                    logger.log_scalar(value, key, iteration)
                logger.flush()

            if args.video_log_freq > 0 and iteration % args.video_log_freq == 0:
                video_env = gym.make(args.env_name, render_mode="rgb_array")
                try:
                    videos = utils.sample_n_trajectories(video_env, actor, 2, episode_length, render=True)
                    logger.log_paths_as_videos(videos, iteration, fps=video_env.metadata.get("render_fps", 30))
                finally:
                    video_env.close()

            if args.save_params:
                actor.save(str(logdir / f"policy_itr_{iteration}.pt"))
    finally:
        env.close()
        logger.close()
    return logdir


if __name__ == "__main__":
    print(f"Logs saved to {run_training_loop(parse_args())}")
