# Week7 HalfCheetah GPU MPC results

The full 15-iteration random MPC run completed on the RTX 4060 Laptop GPU. The trainer printed `Using GPU id 0` and evaluated ten rollouts after each iteration. Mean evaluation return increased from 122.56 in iteration 0 to 416.52 in iteration 14; the final value was also the best observed value. The CEM run is training separately and will be added after completion.

The [training curve](./overview.png) and [evaluation values](./metrics.json) come from TensorBoard events. [build_results.py](./build_results.py) requires the expected number of evaluations, checks that the final dynamics checkpoint contains finite tensors, copies it to [checkpoints/](./checkpoints), and regenerates the plot:

```bash
uv run --project Week6 --frozen --no-sync --offline python results/week7/gpu/build_results.py
```

The evaluation return is based on ten simulator rollouts per iteration. It varies with the random seed and is not a separate held-out environment score.
