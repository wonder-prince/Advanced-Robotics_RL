"""Trajectory sampling compatible with the four- and five-value Gym step APIs."""

from __future__ import annotations

import numpy as np


def _reset(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_ob, reward, terminated, truncated, _ = result
        return next_ob, reward, bool(terminated or truncated)
    next_ob, reward, done, _ = result
    return next_ob, reward, bool(done)


def sample_trajectory(env, policy, max_path_length: int, render: bool = False) -> dict:
    if max_path_length <= 0:
        raise ValueError("max_path_length must be positive")
    ob = _reset(env)
    observations, actions, rewards, next_observations, terminals, images = [], [], [], [], [], []
    for step in range(max_path_length):
        if render:
            image = env.render()
            if image is None:
                raise RuntimeError("Create the environment with render_mode='rgb_array' to record video")
            images.append(np.asarray(image, dtype=np.uint8))

        action = np.asarray(policy.get_action(ob), dtype=np.float32).reshape(-1)
        if hasattr(env.action_space, "low"):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        next_ob, reward, done = _step(env, action)
        rollout_done = done or step + 1 >= max_path_length
        observations.append(ob)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_ob)
        terminals.append(rollout_done)
        ob = next_ob
        if rollout_done:
            break

    return {
        "observation": np.asarray(observations, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "next_observation": np.asarray(next_observations, dtype=np.float32),
        "terminal": np.asarray(terminals, dtype=np.float32),
        "image_obs": np.asarray(images, dtype=np.uint8),
    }


def sample_trajectories(env, policy, min_timesteps_per_batch: int, max_path_length: int, render: bool = False):
    paths = []
    steps = 0
    while steps < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        steps += len(path["reward"])
    return paths, steps


def sample_n_trajectories(env, policy, ntraj: int, max_path_length: int, render: bool = False):
    return [sample_trajectory(env, policy, max_path_length, render) for _ in range(ntraj)]


def compute_metrics(train_paths: list[dict], eval_paths: list[dict]) -> dict[str, float]:
    result = {}
    for prefix, paths in (("Train", train_paths), ("Eval", eval_paths)):
        returns = np.asarray([np.sum(p["reward"]) for p in paths], dtype=np.float32)
        lengths = np.asarray([len(p["reward"]) for p in paths], dtype=np.float32)
        result[f"{prefix}_AverageReturn"] = float(returns.mean())
        result[f"{prefix}_StdReturn"] = float(returns.std())
        result[f"{prefix}_MaxReturn"] = float(returns.max())
        result[f"{prefix}_MinReturn"] = float(returns.min())
        result[f"{prefix}_AverageEpLen"] = float(lengths.mean())
    return result
