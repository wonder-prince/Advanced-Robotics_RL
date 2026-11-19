"""Summarize completed Week 8 Unity GPU runs and draw their training curves.

Run from the repository root with:
    uv run --project Week6 --frozen --no-sync --offline python results/week8/full/build_results.py
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboardX.proto.event_pb2 import Event


OUT = Path(__file__).resolve().parent
RUNS = {
    "snowball-gpu-200k": ("SnowballTarget", 200_000),
    "pyramids-gpu-1m": ("Pyramids", 1_000_000),
    "soccer-gpu-5m": ("SoccerTwos", 5_000_000),
}


def read_scalars(run: Path, behavior: str) -> dict[str, list[dict[str, float | int]]]:
    history: dict[str, list[dict[str, float | int]]] = {}
    for event_file in (run / behavior).glob("events*"):
        with event_file.open("rb") as stream:
            while length_bytes := stream.read(8):
                length = struct.unpack("<Q", length_bytes)[0]
                stream.read(4)
                event = Event.FromString(stream.read(length))
                stream.read(4)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        number = value.simple_value
                    elif value.HasField("tensor") and value.tensor.float_val:
                        number = value.tensor.float_val[0]
                    else:
                        continue
                    history.setdefault(value.tag, []).append(
                        {"step": int(event.step), "value": float(number)}
                    )
    for points in history.values():
        points.sort(key=lambda point: point["step"])
    return history


def main() -> None:
    metrics: dict[str, dict] = {}
    completed = [
        (name, behavior, target)
        for name, (behavior, target) in RUNS.items()
        if (OUT / name / f"{behavior}.onnx").is_file()
    ]
    if not completed:
        raise RuntimeError("No completed runs found")
    fig, axes = plt.subplots(
        1, len(completed), figsize=(5 * len(completed), 4.5), constrained_layout=True
    )
    if len(completed) == 1:
        axes = [axes]
    for axis, (run_name, behavior, target) in zip(axes, completed):
        run = OUT / run_name
        history = read_scalars(run, behavior)
        reward_tag = next(
            (tag for tag in history if "Cumulative Reward" in tag), None
        )
        if reward_tag is None:
            raise RuntimeError(f"No cumulative reward in {run}: {list(history)}")
        points = history[reward_tag]
        status = json.loads((run / "run_logs" / "training_status.json").read_text(encoding="utf-8"))
        final_checkpoint = status[behavior]["final_checkpoint"]
        if final_checkpoint["steps"] < target:
            raise RuntimeError(f"Incomplete checkpoint for {run_name}")
        model = run / f"{behavior}.onnx"
        axis.plot([p["step"] for p in points], [p["value"] for p in points])
        axis.set(title=behavior, xlabel="Trainer steps", ylabel="Mean cumulative reward")
        axis.grid(alpha=0.3)
        metrics[run_name] = {
            "behavior": behavior,
            "target_steps": target,
            "reported_steps": points[-1]["step"],
            "final_mean_reward": points[-1]["value"],
            "checkpoint_steps": final_checkpoint["steps"],
            "checkpoint_reward": final_checkpoint["reward"],
            "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "reward_tag": reward_tag,
            "all_tags": list(history),
            "reward_history": points,
        }
    fig.savefig(OUT / "overview.png", dpi=180)
    plt.close(fig)
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, data in metrics.items():
        print(name, data["reported_steps"], data["final_mean_reward"])


if __name__ == "__main__":
    main()
