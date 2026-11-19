#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
  echo 'Usage: bash setup-gpu.sh {week3|week6|week7|all}'
}

if (( $# != 1 )); then
  usage >&2
  exit 2
fi

case "$1" in
  week3) projects=(Week3) ;;
  week6) projects=(Week6) ;;
  week7) projects=(Week7) ;;
  all) projects=(Week3 Week6 Week7) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'Use x86-64 Linux or Ubuntu on x86-64 WSL 2.' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi

for project in "${projects[@]}"; do
  if [[ $project == Week3 ]]; then
    uv sync --project "$project" --locked --extra atari
  else
    uv sync --project "$project" --locked
  fi
  uv run --project "$project" --no-sync python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA unavailable (PyTorch CUDA runtime: {torch.version.cuda}); check the NVIDIA driver.")
x = torch.ones((8, 8), device="cuda")
assert (x @ x)[0, 0].item() == 8.0
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
'
done
