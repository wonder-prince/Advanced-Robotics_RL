"""评估 ViZDoom 智能体 10 局，并保存一局视频。"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def build_command(args: argparse.Namespace) -> list[str]:
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0")
    return [
        sys.executable,
        "-c",
        """\
import sys
import numpy as np
from numpy.core.multiarray import scalar
from torch.serialization import safe_globals
from sf_examples.vizdoom.enjoy_vizdoom import main

# Sample Factory serializes scalar NumPy rewards in its checkpoints.
with safe_globals([
    (scalar, 'numpy.core.multiarray.scalar'), np.dtype,
    type(np.dtype('float32')), type(np.dtype('float64')), type(np.dtype('int64')),
]):
    status, _ = main()
sys.exit(status)
""",
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
    if args.dry_run:
        return

    video_name = args.video_name if args.video_name.endswith(".mp4") else f"{args.video_name}.mp4"
    video = args.train_dir / args.experiment / video_name
    previous_video = video.stat() if video.is_file() else None

    if shutil.which("ffmpeg"):
        subprocess.run(command, check=True, cwd=HERE)
    else:
        from imageio_ffmpeg import get_ffmpeg_exe

        with TemporaryDirectory(prefix="doom-ffmpeg-") as directory:
            ffmpeg = Path(directory) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            shutil.copy2(get_ffmpeg_exe(), ffmpeg)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ.get("PATH", ""))
            subprocess.run(command, check=True, cwd=HERE, env=env)

    current_video = video.stat() if video.is_file() else None
    if current_video is None or not current_video.st_size or current_video == previous_video:
        raise RuntimeError(f"未生成回放视频：{video}")


if __name__ == "__main__":
    main()
