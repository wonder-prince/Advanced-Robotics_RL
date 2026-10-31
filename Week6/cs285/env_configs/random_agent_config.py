import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from typing import Optional

def random_agent_config(
    env_name: str,
    exp_name: Optional[str] = None,
    **kwargs
):
    log_string = "{env_name}_random".format(
        env_name=env_name
    )

    def make_env():
        env = gym.make(env_name)
        return RecordEpisodeStatistics(env)

    return {
        "agent": "random",
        "agent_kwargs": {},
        "make_env": make_env,
        "env_name": env_name,
        "log_name": log_string,
        "batch_size": 0,
        **kwargs,
    }
