"""用 Sample Factory 的 ViZDoom 集成训练 Health Gathering Supreme。"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENV_ID = "doom_health_gathering_supreme"


def build_command(args: argparse.Namespace) -> list[str]:
    sampling_size = args.workers * args.envs_per_worker * args.rollout
    dataset_size = args.batch_size * args.batches_per_epoch
    if min(
        args.timesteps,
        args.workers,
        args.envs_per_worker,
        args.rollout,
        args.batch_size,
        args.batches_per_epoch,
        args.epochs,
    ) <= 0:
        raise ValueError("训练步数和批量参数必须大于 0")
    if args.envs_per_worker % 2:
        raise ValueError("默认双缓冲采样要求每个 worker 的环境数为偶数")
    if sampling_size != dataset_size:
        raise ValueError(
            f"同步 PPO 需要采样量与训练集相等："
            f"{sampling_size=}，{dataset_size=}"
        )

    return [
        sys.executable,
        "-m",
        "sf_examples.vizdoom.train_vizdoom",
        f"--env={ENV_ID}",
        "--algo=APPO",  # SF 的算法标签；async_rl=False 时按同步 PPO 收集数据
        "--async_rl=False",
        "--with_vtrace=False",
        f"--experiment={args.experiment}",
        f"--train_dir={args.train_dir.resolve()}",
        f"--device={args.device}",
        f"--seed={args.seed}",
        f"--train_for_env_steps={args.timesteps}",
        f"--num_workers={args.workers}",
        f"--num_envs_per_worker={args.envs_per_worker}",
        f"--rollout={args.rollout}",
        f"--batch_size={args.batch_size}",
        f"--num_batches_per_epoch={args.batches_per_epoch}",
        f"--num_epochs={args.epochs}",
        "--shuffle_minibatches=True",
        "--optimizer=adam",
        "--learning_rate=0.0001",
        "--lr_schedule=constant",
        "--gamma=0.99",
        "--gae_lambda=0.95",
        "--ppo_clip_ratio=0.2",
        "--ppo_clip_value=0.2",
        "--value_loss_coeff=0.5",
        "--exploration_loss=entropy",
        "--exploration_loss_coeff=0.01",
        "--max_grad_norm=0.5",
        "--normalize_input=False",
        "--normalize_returns=False",
        "--obs_scale=255.0",
        "--pixel_format=CHW",
        "--res_w=128",
        "--res_h=72",
        "--encoder_conv_architecture=convnet_simple",
        "--encoder_conv_mlp_layers",
        "512",
        "--nonlinearity=relu",
        "--use_rnn=False",
        "--actor_critic_share_weights=True",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches-per-epoch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--experiment", default="default_experiment")
    parser.add_argument("--train-dir", type=Path, default=HERE / "train_dir")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不启动训练")
    args = parser.parse_args()
    try:
        command = build_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
