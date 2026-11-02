"""Export the Week 5–7 pilot TensorBoard scalars, figure, and checkpoints.

Run from the repository root with:
    uv run --project Week6 --frozen --offline python results/week5-7/build_results.py
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboardX.proto.event_pb2 import Event


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def latest_log(pattern: str, required: str) -> Path:
    matches = [p for p in ROOT.glob(pattern) if p.is_dir() and any(p.glob(required))]
    if not matches:
        raise FileNotFoundError(f"No completed run: {pattern} / {required}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def scalars(folder: Path) -> dict[str, list[dict[str, float | int]]]:
    history: dict[str, list[dict[str, float | int]]] = {}
    for event_file in folder.glob("events*"):
        with event_file.open("rb") as stream:
            while length_bytes := stream.read(8):
                length = struct.unpack("<Q", length_bytes)[0]
                stream.read(4)  # TFRecord length CRC
                payload = stream.read(length)
                stream.read(4)  # TFRecord payload CRC
                event = Event.FromString(payload)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        history.setdefault(value.tag, []).append(
                            {"step": event.step, "value": value.simple_value}
                        )
    for values in history.values():
        values.sort(key=lambda item: item["step"])
    return history


def checked_copy(source: Path, name: str) -> dict[str, str | int]:
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Empty or invalid checkpoint: {source}")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if not tensors or not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"Non-finite checkpoint: {source}")
    destination = OUT / "checkpoints" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"file": str(destination.relative_to(OUT)), "tensors": len(tensors), "bytes": destination.stat().st_size}


def series(ax, run: dict, tag: str, label: str) -> None:
    values = run["scalars"][tag]
    ax.plot([item["step"] for item in values], [item["value"] for item in values], marker="o", label=label)


def main() -> None:
    runs = {
        "week5_bc": (ROOT / "Week5/imitation_learning/data/bc_pilot_hopper_Hopper-v4", ["policy_itr_0.pt"]),
        "week5_dagger": (ROOT / "Week5/imitation_learning/data/dagger_pilot_hopper_Hopper-v4", ["policy_itr_2.pt"]),
        "week6_rnd": (latest_log("Week6/data/hw5_explore_PointmassEasy-v0_rnd1.0_zfilter_*", "events*"), []),
        "week6_dqn": (latest_log("Week6/data/hw5_offline_pilot_easy_dqn_*", "final_agent.pt"), ["final_agent.pt"]),
        "week6_cql": (latest_log("Week6/data/hw5_offline_PointmassEasy-v0_cql0.1_*", "final_agent.pt"), ["final_agent.pt"]),
        "week6_awac": (latest_log("Week6/data/hw5_offline_PointmassEasy-v0_awac10.0_*", "final_agent.pt"), ["final_agent.pt"]),
        "week6_iql": (latest_log("Week6/data/hw5_offline_PointmassEasy-v0_iql0.7_temp10.0_*", "final_agent.pt"), ["final_agent.pt"]),
        "week6_finetune": (latest_log("Week6/data/hw5_finetune_pilot_easy_dqn_finetune_*", "finetuned_agent.pt"), ["offline_agent.pt", "finetuned_agent.pt"]),
        "week7_obstacles": (latest_log("Week7/data/obstacles-cs285-v0_obstacles_pilot_*", "final_dynamics.pt"), ["final_dynamics.pt"]),
        "week7_mbpo": (latest_log("Week7/data/cheetah-cs285-v0_cheetah_pilot_*", "final_dynamics.pt"), ["final_dynamics.pt", "final_sac.pt"]),
    }
    model_free = [p for p in ROOT.glob("Week7/data/cheetah-cs285-v0_cheetah_pilot_*")
                  if (p / "final_sac.pt").is_file() and not (p / "final_dynamics.pt").exists()]
    if not model_free:
        raise FileNotFoundError("Week 7 model-free SAC checkpoint")
    runs["week7_model_free"] = (max(model_free, key=lambda p: p.stat().st_mtime), ["final_sac.pt"])

    result = {"runs": {}}
    for key, (folder, checkpoints) in runs.items():
        result["runs"][key] = {
            "log_dir": str(folder.relative_to(ROOT)).replace("\\", "/"),
            "scalars": scalars(folder),
            "checkpoints": [checked_copy(folder / filename, f"{key}_{filename}") for filename in checkpoints],
        }

    with (OUT / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = grid.ravel()
    series(axes[0], result["runs"]["week5_bc"], "Eval_AverageReturn", "BC")
    series(axes[0], result["runs"]["week5_dagger"], "Eval_AverageReturn", "DAgger")
    axes[0].set(title="Week 5: Hopper evaluation", xlabel="Iteration", ylabel="Episode return")
    for key in ("dqn", "cql", "awac", "iql", "finetune"):
        series(axes[1], result["runs"][f"week6_{key}"], "eval_return", key.upper())
    axes[1].set(title="Week 6: Pointmass Easy", xlabel="Update step", ylabel="Episode return")
    series(axes[2], result["runs"]["week7_obstacles"], "eval_return", "MPC")
    axes[2].set(title="Week 7: Obstacles MPC", xlabel="Iteration", ylabel="Episode return")
    for key, label in (("mbpo", "MBPO"), ("model_free", "Model-free SAC")):
        series(axes[3], result["runs"][f"week7_{key}"], "eval_return", label)
    axes[3].set(title="Week 7: HalfCheetah", xlabel="Iteration", ylabel="Episode return")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)
    print("Wrote", OUT / "metrics.json", "and", OUT / "overview.png")


if __name__ == "__main__":
    main()
