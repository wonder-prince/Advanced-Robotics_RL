"""Validate exported Week 8 ONNX models with the WSL ML-Agents environment."""

from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parent


def main() -> None:
    models = sorted(ROOT.glob("*-gpu-*/[A-Za-z]*.onnx"))
    if not models:
        raise RuntimeError("No exported models found")
    for path in models:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        inputs = [item.name for item in model.graph.input]
        outputs = [item.name for item in model.graph.output]
        print(f"{path.parent.name}: ONNX valid; inputs={inputs}; outputs={outputs}")


if __name__ == "__main__":
    main()
