# Week7 HalfCheetah GPU MPC results

Both HalfCheetah model-based MPC runs completed on the RTX 4060 Laptop GPU. The trainer printed `Using GPU id 0` and evaluated ten rollouts after each iteration.

| Planner | Completed iterations | Final mean evaluation return | Best mean evaluation return |
| --- | ---: | ---: | ---: |
| Random MPC | 15 | 416.52 | 416.52 |
| CEM MPC | 5 | 1028.64 | 1028.64 |

The [training curves](./overview.png) and [evaluation values](./metrics.json) come from TensorBoard events. [build_results.py](./build_results.py) requires the expected number of evaluations, checks that each final dynamics checkpoint contains finite tensors, copies them into [checkpoints/](./checkpoints), and regenerates the plot:

```bash
uv run --project Week6 --frozen --no-sync --offline python results/week7/gpu/build_results.py
```

The return is based on ten simulator rollouts per iteration. These runs use different planners and iteration counts, and the values vary with the random seed; they are not independent held-out scores.
