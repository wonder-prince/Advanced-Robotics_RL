"""Export verified Week 7 HalfCheetah GPU MPC runs.

Run from the repository root with:
    uv run --project Week6 --frozen --no-sync --offline python results/week7/gpu/build_results.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboardX.proto.event_pb2 import Event


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = {
    "random_mpc": ("Week7/data/cheetah-cs285-v0_cheetah_multi_*", 15),
    "cem_mpc": ("Week7/data/cheetah-cs285-v0_cheetah_cem_*", 5),
}


def scalars(folder: Path) -> dict[str, list[dict[str, float | int]]]:
    history: dict[str, list[dict[str, float | int]]] = {}
    for event_file in folder.glob("events*"):
        with event_file.open("rb") as stream:
            while length_bytes := stream.read(8):
                length = struct.unpack("<Q", length_bytes)[0]
                stream.read(4)
                event = Event.FromString(stream.read(length))
                stream.read(4)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        history.setdefault(value.tag, []).append(
                            {"step": int(event.step), "value": float(value.simple_value)}
                        )
    for values in history.values():
        values.sort(key=lambda item: item["step"])
    return history


def checkpoint_info(source: Path, name: str) -> dict[str, str | int]:
    state = torch.load(source, map_location="cpu", weights_only=False)
    tensors = list(state.values()) if isinstance(state, dict) else []
    tensors = [value for value in tensors if isinstance(value, torch.Tensor)]
    if not tensors or not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"Missing or non-finite checkpoint tensors: {source}")
    destination = OUT / "checkpoints" / f"{name}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "file": destination.relative_to(OUT).as_posix(),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "tensors": len(tensors),
    }


def main() -> None:
    result: dict[str, dict] = {}
    for key, (pattern, expected) in RUNS.items():
        folders = [path for path in ROOT.glob(pattern) if (path / "final_dynamics.pt").is_file()]
        if not folders:
            continue
        if len(folders) != 1:
            raise RuntimeError(f"Expected one completed {key} run: {folders}")
        folder = folders[0]
        values = scalars(folder).get("eval_return", [])
        if len(values) != expected:
            raise RuntimeError(f"Incomplete {key} evaluation: {len(values)} != {expected}")
        result[key] = {
            "source": folder.relative_to(ROOT).as_posix(),
            "eval_return": values,
            "final_eval_return": values[-1]["value"],
            "best_eval_return": max(point["value"] for point in values),
            "checkpoint": checkpoint_info(folder / "final_dynamics.pt", key),
        }
    if not result:
        raise RuntimeError("No completed HalfCheetah GPU runs found")
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key, data in result.items():
        values = data["eval_return"]
        axis.plot(
            [point["step"] for point in values],
            [point["value"] for point in values],
            marker="o", label=key.replace("_", " ").upper(),
        )
    axis.set(title="HalfCheetah GPU MPC", xlabel="Training iteration", ylabel="Mean evaluation return")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(OUT / "overview.png", dpi=180)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, data in result.items():
        print(key, data["final_eval_return"], data["best_eval_return"])


if __name__ == "__main__":
    main()
