"""Export the Week 3 Atari DQN evaluation and trained policies.

Run from the repository root with:
    uv run --project Week6 --frozen --no-sync --offline python results/week3/gpu/build_results.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "Week3/logs/course-gpu/dqn/ALE-SpaceInvaders-v5_1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / "evaluations.npz"
    shutil.copy2(SOURCE / "evaluations.npz", snapshot)
    with np.load(snapshot) as data:
        steps = data["timesteps"].astype(int)
        rewards = data["results"].astype(float)
        lengths = data["ep_lengths"].astype(float)
    if steps.ndim != 1 or rewards.shape != lengths.shape or len(steps) != len(rewards):
        raise ValueError("Unexpected evaluation array shapes")
    if not np.all(np.isfinite(rewards)) or not np.all(np.diff(steps) > 0):
        raise ValueError("Non-finite or unordered evaluation results")
    means = rewards.mean(axis=1)
    stds = rewards.std(axis=1)
    best_index = int(np.argmax(means))
    model = OUT / "best_model.zip"
    shutil.copy2(SOURCE / "best_model.zip", model)
    with zipfile.ZipFile(model) as archive:
        if bad_file := archive.testzip():
            raise ValueError(f"Corrupt checkpoint entry: {bad_file}")
    final_source = SOURCE / "ALE-SpaceInvaders-v5.zip"
    completed = final_source.is_file() and int(steps[-1]) >= 10_000_000
    final_model = OUT / "final_model.zip"
    if completed:
        shutil.copy2(final_source, final_model)
        with zipfile.ZipFile(final_model) as archive:
            if bad_file := archive.testzip():
                raise ValueError(f"Corrupt final checkpoint entry: {bad_file}")
    result = {
        "environment": "ALE/SpaceInvaders-v5",
        "target_train_steps": 10_000_000,
        "status": "complete" if completed else "in_progress",
        "last_eval_step": int(steps[-1]),
        "eval_episodes_per_point": int(rewards.shape[1]),
        "best_eval_step": int(steps[best_index]),
        "best_eval_mean_reward": float(means[best_index]),
        "last_eval_mean_reward": float(means[-1]),
        "best_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "final_model_sha256": hashlib.sha256(final_model.read_bytes()).hexdigest() if completed else None,
        "evaluations": [
            {"step": int(step), "mean_reward": float(mean), "std_reward": float(std)}
            for step, mean, std in zip(steps, means, stds)
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(steps, means, label="Mean of 5 eval episodes")
    axis.fill_between(steps, means - stds, means + stds, alpha=0.2, label="1 episode SD")
    axis.set(
        title=f"Atari Space Invaders DQN (CUDA, {result['status']})",
        xlabel="Training steps", ylabel="Evaluation reward",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(OUT / "overview.png", dpi=180)
    plt.close(fig)
    print(result["last_eval_step"], result["best_eval_mean_reward"])


if __name__ == "__main__":
    main()
