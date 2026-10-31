import yaml
import os
import time

import cs285.env_configs
from cs285.infrastructure.logger import Logger

def make_config(config_file: str, overrides=None) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"expected KEY=VALUE after --set, got {override!r}")
        key, value = override.split("=", 1)
        if key not in config_kwargs and key not in {
            "rnd_bonus_normalization", "retain_offline_buffer", "cql_temperature"
        }:
            raise KeyError(f"unknown config key: {key}")
        config_kwargs[key] = yaml.safe_load(value)

    base_config_name = config_kwargs.pop("base_config")
    return cs285.env_configs.configs[base_config_name](**config_kwargs)

def make_logger(logdir_prefix: str, config: dict) -> Logger:
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix + config["log_name"] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    return Logger(logdir)
