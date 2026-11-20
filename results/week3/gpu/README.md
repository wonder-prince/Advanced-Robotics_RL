# Week3 Atari DQN GPU training results

The Space Invaders DQN run completed 10,000,000 training steps on an RTX 4060 Laptop GPU. RL Baselines3 Zoo logged `Using cuda device`. The ALE environment is `ALE/SpaceInvaders-v5` with one frame skip and zero sticky-action probability; see [dqn.yml](../../../Week3/dqn.yml) for hyperparameters.

Periodic evaluation used five episodes per point. The highest mean reward was **1,581** at 3,675,000 steps; the final evaluation mean was **969** at 10,000,000 steps. The final policy is therefore different from the best evaluated policy. These are simulator evaluation scores, not a held-out benchmark.

The [curve](./overview.png), [numeric summary](./metrics.json), and [evaluation arrays](./evaluations.npz) record the full run. Both the [best policy](./best_model.zip) and [final policy](./final_model.zip) passed ZIP integrity checks and loaded with Stable Baselines3 `DQN.load(..., device="cpu")`; their reported training steps are 3,675,000 and 10,000,000 respectively.

[build_results.py](./build_results.py) validates the arrays and checkpoints, then regenerates the plot and metrics:

```bash
uv run --project Week6 --frozen --no-sync --offline python results/week3/gpu/build_results.py
```
