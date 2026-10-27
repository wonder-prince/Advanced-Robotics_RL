"""评估 ViZDoom 智能体 10 局，并保存一局视频。"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def build_command(args: argparse.Namespace) -> list[str]:
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0")
    return [
        sys.executable,
        "-m",
        "sf_examples.vizdoom.enjoy_vizdoom",
        "--env=doom_health_gathering_supreme",
        "--algo=APPO",
        f"--experiment={args.experiment}",
        f"--train_dir={args.train_dir.resolve()}",
        f"--device={args.device}",
        f"--max_num_episodes={args.episodes}",
        "--eval_deterministic=True",
        "--save_video",
        "--no_render",
        "--video_frames=-1",  # 视频录制一整局；评估仍统计指定的局数
        f"--video_name={args.video_name}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--experiment", default="default_experiment")
    parser.add_argument("--train-dir", type=Path, default=HERE / "train_dir")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--video-name", default="replay")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不加载模型")
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
