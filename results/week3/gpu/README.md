# Week3 Atari DQN GPU training snapshot

The 10,000,000-step Space Invaders DQN run is still training on an RTX 4060 Laptop GPU. RL Baselines3 Zoo logged `Using cuda device`. The current ALE environment is `ALE/SpaceInvaders-v5` with one frame skip and zero sticky-action probability, matching the old `SpaceInvadersNoFrameskip-v4` behavior. The complete hyperparameters are in [dqn.yml](../../../Week3/dqn.yml).

At this snapshot, periodic five-episode evaluation had reached 750,000 training steps. The best mean evaluation reward was 780.0 at 750,000 steps. The [curve](./overview.png), [numeric summary](./metrics.json), [evaluation arrays](./evaluations.npz), and [best policy](./best_model.zip) are an **interim** result; no 10-million-step outcome is claimed.

[build_results.py](./build_results.py) validates array shapes and finite rewards, checks the ZIP archive, and regenerates the plot. The copied policy was also loaded successfully with Stable Baselines3 `DQN.load(..., device="cpu")`, which reported 750,000 training steps. Refresh this snapshot after further training with:

```bash
uv run --project Week6 --frozen --no-sync --offline python results/week3/gpu/build_results.py
```
