import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib import pyplot as plt

from cs285.agents.model_based_agent import ModelBasedAgent
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config

import argparse

from cs285.envs import register_envs

register_envs()


def collect_mbpo_rollout(
    env: gym.Env,
    mb_agent: ModelBasedAgent,
    sac_agent: SoftActorCritic,
    ob: np.ndarray,
    rollout_len: int = 1,
):
    obs, acs, rewards, next_obs, dones = [], [], [], [], []
    for _ in range(rollout_len):
        ac = sac_agent.get_action(ob)
        predictions = [
            mb_agent.get_dynamics_predictions(i, ob[None], ac[None])[0]
            for i in range(mb_agent.ensemble_size)
        ]
        next_ob = np.mean(predictions, axis=0).astype(np.float32)
        reward, done = env.get_reward(next_ob[None], ac[None])
        rew = float(np.asarray(reward).reshape(-1)[0])
        terminal = bool(np.asarray(done).reshape(-1)[0])

        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(terminal)

        ob = next_ob
        if terminal:
            break

    return {
        "observation": np.array(obs),
        "action": np.array(acs),
        "reward": np.array(rewards),
        "next_observation": np.array(next_obs),
        "done": np.array(dones),
    }


def run_training_loop(
    config: dict, logger: Logger, args: argparse.Namespace, sac_config: Optional[dict]
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True) if args.num_render_trajectories > 0 else None
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our MPC implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 2

    # initialize agent
    train_dynamics = sac_config is None or sac_config["mbpo_rollout_length"] > 0
    mb_agent = ModelBasedAgent(env, **config["agent_kwargs"]) if train_dynamics else None
    actor_agent = mb_agent

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # if doing MBPO, initialize SAC and make that our main agent that we use to
    # collect data and evaluate
    if sac_config is not None:
        sac_agent = SoftActorCritic(
            env.observation_space.shape,
            env.action_space.shape[0],
            **sac_config["agent_kwargs"],
        )
        sac_replay_buffer = ReplayBuffer(sac_config["replay_buffer_capacity"])
        actor_agent = sac_agent

    total_envsteps = 0
    sac_train_steps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env, utils.RandomPolicy(env), config["initial_batch_size"], ep_len
            )
        else:
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env, actor_agent, config["batch_size"], ep_len
            )

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # if doing MBPO, add the collected data to the SAC replay buffer as well
        if sac_config is not None:
            for traj in trajs:
                sac_replay_buffer.batched_insert(
                    observations=traj["observation"],
                    actions=traj["action"],
                    rewards=traj["reward"],
                    next_observations=traj["next_observation"],
                    dones=traj["done"],
                )

        if train_dynamics:
            mb_agent.update_statistics(
                obs=replay_buffer.observations[: min(len(replay_buffer), replay_buffer.max_size)],
                acs=replay_buffer.actions[: min(len(replay_buffer), replay_buffer.max_size)],
                next_obs=replay_buffer.next_observations[: min(len(replay_buffer), replay_buffer.max_size)],
            )

            print("Training dynamics models...")
            all_losses = []
            for _ in tqdm.trange(config["num_agent_train_steps_per_iter"], dynamic_ncols=True):
                step_losses = []
                for model_index in range(mb_agent.ensemble_size):
                    batch = replay_buffer.sample(config["train_batch_size"])
                    step_losses.append(
                        mb_agent.update(
                            model_index, batch["observations"],
                            batch["actions"], batch["next_observations"],
                        )
                    )
                all_losses.append(np.mean(step_losses))

            if itr == 0:
                plt.plot(all_losses)
                plt.title("Iteration 0: Dynamics Model Training Loss")
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))
                plt.close()
            logger.log_scalar(np.mean(all_losses), "dynamics_loss", itr)

        # for MBPO: now we need to train the SAC agent
        if sac_config is not None:
            print("Training SAC agent...")
            for i in tqdm.trange(
                sac_config["num_agent_train_steps_per_iter"], dynamic_ncols=True
            ):
                if sac_config["mbpo_rollout_length"] > 0:
                    # collect a rollout using the dynamics model
                    rollout = collect_mbpo_rollout(
                        env,
                        mb_agent,
                        sac_agent,
                        # sample one observation from the "real" replay buffer
                        replay_buffer.sample(1)["observations"][0],
                        sac_config["mbpo_rollout_length"],
                    )
                    # insert it into the SAC replay buffer only
                    sac_replay_buffer.batched_insert(
                        observations=rollout["observation"],
                        actions=rollout["action"],
                        rewards=rollout["reward"],
                        next_observations=rollout["next_observation"],
                        dones=rollout["done"],
                    )
                # train SAC
                batch = sac_replay_buffer.sample(sac_config["batch_size"])
                sac_metrics = sac_agent.update(
                    batch["observations"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_observations"],
                    batch["dones"],
                    sac_train_steps,
                )
                sac_train_steps += 1
            logger.log_scalar(sac_metrics["actor_loss"], "sac_actor_loss", itr)
            logger.log_scalar(sac_metrics["critic_loss"], "sac_critic_loss", itr)

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)

        if args.num_render_trajectories > 0:
            video_trajectories = utils.sample_n_trajectories(
                render_env, actor_agent, args.num_render_trajectories, ep_len, render=True
            )
            logger.log_paths_as_videos(
                video_trajectories, itr, fps=fps,
                max_videos_to_save=args.num_render_trajectories,
                video_title="eval_rollouts",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--sac_config_file", type=str, default=None)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    config = make_config(args.config_file)
    logger = make_logger(config)

    if args.sac_config_file is not None:
        sac_config = make_config(args.sac_config_file)
    else:
        sac_config = None

    run_training_loop(config, logger, args, sac_config)


if __name__ == "__main__":
    main()
