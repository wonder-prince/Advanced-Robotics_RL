"""Offline pretraining followed by online fine-tuning in Pointmass."""

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gym
import numpy as np
import torch
import tqdm

from cs285.agents import agents as agent_types
from cs285.envs import Pointmass  # noqa: F401 - registers Pointmass environments
from cs285.infrastructure import pytorch_util as ptu, utils
from cs285.infrastructure.replay_buffer import ReplayBuffer
from scripting_utils import make_config, make_logger
from run_hw5_explore import visualize


def _copy_dataset(source: ReplayBuffer, target: ReplayBuffer) -> None:
    """Keep the original transition order when copying a possibly circular buffer."""
    count = min(source.size, source.max_size)
    for offset in range(count):
        index = (source.size - count + offset) % source.max_size
        target.insert(
            observation=source.observations[index],
            action=source.actions[index],
            reward=source.rewards[index],
            next_observation=source.next_observations[index],
            done=source.dones[index],
        )


def _log_evaluation(logger, env, agent, step: int, episodes: int, episode_length: int) -> None:
    trajectories = utils.sample_n_trajectories(env, agent, episodes, episode_length)
    returns = np.asarray([path["episode_statistics"]["r"] for path in trajectories])
    lengths = np.asarray([path["episode_statistics"]["l"] for path in trajectories])
    logger.log_scalar(float(returns.mean()), "eval_return", step)
    logger.log_scalar(float(lengths.mean()), "eval_ep_len", step)
    if len(returns) > 1:
        logger.log_scalar(float(returns.std()), "eval/return_std", step)


def run_training_loop(config: dict, logger, args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env = config["make_env"]()
    eval_env = config["make_env"]()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("Pointmass fine-tuning requires discrete actions")
    agent = agent_types[config["agent"]](
        env.observation_space.shape, env.action_space.n, **config["agent_kwargs"]
    )
    episode_length = env.spec.max_episode_steps or env.unwrapped.max_episode_steps
    offline_steps = int(config["offline_steps"])
    total_steps = int(config["total_steps"])
    if not 0 < offline_steps < total_steps:
        raise ValueError("require 0 < offline_steps < total_steps")

    dataset_file = Path(args.dataset_dir) / f"{config['dataset_name']}.pkl"
    with dataset_file.open("rb") as stream:
        offline_dataset = pickle.load(stream)
    if len(offline_dataset) == 0:
        raise ValueError("offline dataset is empty")

    retain = bool(config.get("retain_offline_buffer", False) or args.retain_offline_buffer)
    online_steps = total_steps - offline_steps
    replay_buffer = None
    observation = None
    recent_observations = []
    exploration_schedule = config.get("exploration_schedule")

    try:
        for step in tqdm.trange(total_steps, dynamic_ncols=True):
            epsilon = None
            if step < offline_steps:
                batch = offline_dataset.sample(config["batch_size"])
            else:
                if step == offline_steps:
                    # Start a new online buffer. The ablation flag retains offline data.
                    old_count = min(offline_dataset.size, offline_dataset.max_size) if retain else 0
                    replay_buffer = ReplayBuffer(capacity=online_steps + old_count)
                    if retain:
                        _copy_dataset(offline_dataset, replay_buffer)
                    observation = env.reset()
                    torch.save(agent.state_dict(), os.path.join(logger._log_dir, "offline_agent.pt"))

                epsilon = exploration_schedule.value(step) if exploration_schedule else 0.0
                action = agent.get_action(observation, epsilon)
                next_observation, reward, done, info = env.step(action)
                truncated = info.get("TimeLimit.truncated", False)
                replay_buffer.insert(
                    observation=np.asarray(observation),
                    action=action,
                    reward=reward,
                    next_observation=np.asarray(next_observation),
                    done=done and not truncated,
                )
                recent_observations.append(observation)
                if done:
                    if "episode" in info:
                        logger.log_scalar(info["episode"]["r"], "train_return", step)
                        logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
                    observation = env.reset()
                else:
                    observation = next_observation

                if len(replay_buffer) < config["batch_size"]:
                    continue
                batch = replay_buffer.sample(config["batch_size"])

            batch = ptu.from_numpy(batch)
            metrics = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
                step,
            )
            metrics["phase"] = 0 if step < offline_steps else 1
            if epsilon is not None:
                metrics["epsilon"] = epsilon

            if step % args.log_interval == 0:
                for key, value in metrics.items():
                    logger.log_scalar(value, key, step)
                logger.flush()
            if step % args.eval_interval == 0:
                _log_evaluation(logger, eval_env, agent, step, args.num_eval_trajectories, episode_length)
            if recent_observations and step % args.visualize_interval == 0:
                logger.log_figure(
                    visualize(env.unwrapped, agent, np.asarray(recent_observations)),
                    "online_state_coverage", step, "train"
                )
                recent_observations.clear()

        torch.save(agent.state_dict(), os.path.join(logger._log_dir, "finetuned_agent.pt"))
        output = Path(args.dataset_dir) / f"{config['dataset_name']}_online.pkl"
        with output.open("wb") as stream:
            pickle.dump(replay_buffer, stream)
    finally:
        env.close()
        eval_env.close()
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", "-cfg", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--retain_offline_buffer", action="store_true", help="Ablation: keep offline samples online")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--visualize_interval", type=int, default=1000)
    parser.add_argument("--num_eval_trajectories", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    config = make_config(args.config_file, args.set)
    logger = make_logger("hw5_finetune_", config)
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
