"""Validate or run a Week8 ML-Agents experiment."""

import argparse
from pathlib import Path

from mlagents.trainers.learn import parse_command_line, run_cli


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "snowball": "ppo/SnowballTarget.yaml",
    "pyramids": "ppo/PyramidsRND.yaml",
    "soccer": "poca/SoccerTwos.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=CONFIGS)
    parser.add_argument("--check", action="store_true", help="Validate trainer settings without Unity")
    parser.add_argument("--smoke", action="store_true", help="Limit to 20,000 trainer steps")
    parser.add_argument("--env", type=Path, help="Path to the Unity build")
    parser.add_argument("--run-id", help="Unique name for the output directory")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    cli = [str(ROOT / "config" / CONFIGS[args.experiment])]
    if not args.check:
        if args.env is None or not args.env.exists():
            parser.error("--env must point to an existing Unity build")
        if not args.run_id:
            parser.error("--run-id is required for training")
        if args.num_envs < 1:
            parser.error("--num-envs must be positive")
        cli.extend([
            "--env", str(args.env.resolve()), "--run-id", args.run_id,
            "--num-envs", str(args.num_envs), "--base-port", str(args.base_port),
            "--results-dir", str(ROOT / "results"), "--torch-device", args.torch_device,
        ])
        if not args.graphics:
            cli.append("--no-graphics")
        if args.resume:
            cli.append("--resume")

    options = parse_command_line(cli)
    for behavior, settings in options.behaviors.items():
        if args.smoke:
            settings.max_steps = 20_000
            settings.summary_freq = 2_000
            settings.checkpoint_interval = 20_000
        print(f"{behavior}: {settings.trainer_type}, max_steps={settings.max_steps}", flush=True)

    if not args.check:
        run_cli(options)


if __name__ == "__main__":
    main()
