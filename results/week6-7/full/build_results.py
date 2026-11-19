"""Export verified Week 6 and Week 7 full-run results.

Run from the repository root with:
    uv run --project Week6 --frozen --no-sync --offline python results/week6-7/full/build_results.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboardX.proto.event_pb2 import Event


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def run(pattern: str, required: str) -> Path:
    matches = [p for p in ROOT.glob(pattern) if p.is_dir() and (p / required).is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completed run for {pattern}: {matches}")
    return matches[0]


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
                            {"step": event.step, "value": value.simple_value}
                        )
    for values in history.values():
        values.sort(key=lambda item: item["step"])
    return history


def export_file(source: Path, subdir: str, name: str) -> dict[str, str | int]:
    destination = OUT / subdir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "file": destination.relative_to(OUT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def export_checkpoint(source: Path, name: str) -> dict[str, str | int]:
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid checkpoint: {source}")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if not tensors or not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"Non-finite checkpoint: {source}")
    result = export_file(source, "checkpoints", name)
    result["tensors"] = len(tensors)
    return result


def export_dataset(source: Path, expected_steps: int) -> dict[str, str | int]:
    sys.path.insert(0, str(ROOT / "Week6"))
    with source.open("rb") as stream:
        dataset = pickle.load(stream)
    if len(dataset) != expected_steps:
        raise ValueError(f"Incomplete dataset {source}: {len(dataset)} != {expected_steps}")
    result = export_file(source, "datasets", source.name)
    result["transitions"] = len(dataset)
    return result


def plot_series(ax, entry: dict, label: str) -> None:
    values = entry["eval_return"]
    ax.plot([v["step"] for v in values], [v["value"] for v in values], marker="o", markersize=3, label=label)


def main() -> None:
    runs = {
        "week6_hard_dqn": ("Week6/data/hw5_offline_pointmass_hard_dqn_PointmassHard-v0_*", ["final_agent.pt"]),
        "week6_hard_cql": ("Week6/data/hw5_offline_PointmassHard-v0_cql0.1_*", ["final_agent.pt"]),
        "week6_hard_awac": ("Week6/data/hw5_offline_PointmassHard-v0_awac10.0_*", ["final_agent.pt"]),
        "week6_hard_iql": ("Week6/data/hw5_offline_PointmassHard-v0_iql0.99_*", ["final_agent.pt"]),
        "week6_hard_cql_finetune": ("Week6/data/hw5_finetune_PointmassHard-v0_cql1.0_*", ["offline_agent.pt", "finetuned_agent.pt"]),
        "week7_obstacles_mpc": ("Week7/data/obstacles-cs285-v0_obstacles_multi_*", ["final_dynamics.pt"]),
    }
    result: dict = {"runs": {}, "datasets": {}}
    for key, (pattern, checkpoints) in runs.items():
        folder = run(pattern, checkpoints[-1])
        evaluation = scalars(folder).get("eval_return", [])
        expected = 12 if key == "week7_obstacles_mpc" else (19 if key.endswith("finetune") else 20)
        if len(evaluation) != expected:
            raise ValueError(f"Incomplete evaluation series for {key}: {len(evaluation)} != {expected}")
        result["runs"][key] = {
            "source": folder.relative_to(ROOT).as_posix(),
            "eval_return": evaluation,
            "final_eval_return": evaluation[-1]["value"],
            "best_eval_return": max(v["value"] for v in evaluation),
            "checkpoints": [export_checkpoint(folder / filename, f"{key}_{filename}") for filename in checkpoints],
        }

    for difficulty in ("easy", "medium", "hard"):
        for method in ("random", "rnd"):
            key = f"week6_{difficulty}_{method}"
            candidates = [p for p in ROOT.glob(f"Week6/data/hw5_explore_Pointmass{difficulty.title()}-v0_{method}*") if p.is_dir() and any(p.glob("events*"))]
            complete = [(p, values) for p in candidates if (values := scalars(p).get("eval_return", [])) and values[-1]["step"] == 5000]
            if len(complete) != 1:
                raise RuntimeError(f"Expected one 10k-step exploration run for {key}: {complete}")
            folder, evaluation = complete[0]
            result["runs"][key] = {
                "source": folder.relative_to(ROOT).as_posix(),
                "eval_return": evaluation,
                "final_eval_return": evaluation[-1]["value"],
            }
            source = ROOT / "Week6/datasets" / f"pointmass_{difficulty}_{method}.pkl"
            result["datasets"][f"pointmass_{difficulty}_{method}"] = export_dataset(source, 10_000)
    online = ROOT / "Week6/datasets/pointmass_hard_rnd_online.pkl"
    result["datasets"]["pointmass_hard_rnd_online"] = export_dataset(online, 100_000)

    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for difficulty in ("easy", "medium", "hard"):
        for method in ("random", "rnd"):
            key = f"week6_{difficulty}_{method}"
            plot_series(axes[0], result["runs"][key], f"{difficulty.title()} {method.upper()}")
    axes[0].set(title="Week 6: exploration", xlabel="Environment steps", ylabel="Eval return")
    for method in ("dqn", "cql", "awac", "iql"):
        plot_series(axes[1], result["runs"][f"week6_hard_{method}"], method.upper())
    plot_series(axes[1], result["runs"]["week6_hard_cql_finetune"], "CQL fine-tune")
    axes[1].axvline(100_000, color="gray", linestyle="--", linewidth=1)
    axes[1].set(title="Week 6: Pointmass Hard", xlabel="Update / environment step", ylabel="Eval return")
    plot_series(axes[2], result["runs"]["week7_obstacles_mpc"], "MPC")
    axes[2].set(title="Week 7: Obstacles MPC", xlabel="Iteration", ylabel="Eval return")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)
    print(f"Exported {len(result['runs'])} runs and {len(result['datasets'])} datasets to {OUT}")


if __name__ == "__main__":
    main()
