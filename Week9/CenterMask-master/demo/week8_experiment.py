"""Run the Week9 pretrained CenterMask demo and confidence-threshold experiment."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import shlex
import statistics
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK9_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torchvision

from maskrcnn_benchmark import _C
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("confidence thresholds must be in [0, 1]")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file", type=Path,
        default=PROJECT_ROOT / "configs/centermask/centermask_M_v2_FPN_lite_res600_ms_bs32_1x.yaml",
    )
    parser.add_argument("--weights", type=Path, default=WEEK9_ROOT / "weights/centermask-lite-M-v2-ms-bs32-1x.pth")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "demo/images")
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=WEEK9_ROOT / "results/experiment")
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--conf_th", type=probability, help="run a single confidence threshold")
    thresholds.add_argument("--thresholds", type=probability, nargs="+", help="default: 0.1 0.5 0.9")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=3, help="maximum images; 0 means all")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="optional YACS KEY VALUE pairs")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("limit/warmup must be nonnegative; repeats/threads must be positive")
    if args.opts and len(args.opts) % 2:
        parser.error("YACS overrides require KEY VALUE pairs")
    args.thresholds = list(dict.fromkeys(
        args.thresholds if args.thresholds is not None
        else [args.conf_th] if args.conf_th is not None else [0.1, 0.5, 0.9]
    ))
    return args


def find_images(source, limit):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [source] if source.is_file() else sorted(source.iterdir()) if source.is_dir() else []
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in extensions]
    if not paths:
        raise ValueError(f"No supported images found: {source}")
    return paths[:limit] if limit else paths


def render(demo, image, predictions):
    result = image.copy()
    colors = demo.compute_colors_for_labels(predictions.get_field("labels")).tolist()
    masks = predictions.get_field("mask").numpy() if predictions.has_field("mask") else None
    for index, (box, color) in enumerate(zip(predictions.bbox.tolist(), colors)):
        if masks is not None:
            region = masks[index, 0].astype(bool)
            result[region] = (0.55 * result[region] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = int(predictions.get_field("labels")[index])
        score = float(predictions.get_field("scores")[index])
        text = f"{demo.CATEGORIES[label]} {score:.2f}"
        cv2.putText(result, text, (max(x1, 0), max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return result


def serialize_instances(demo, predictions):
    instances = []
    for index, box in enumerate(predictions.bbox.tolist()):
        label = int(predictions.get_field("labels")[index])
        record = {
            "bbox_xyxy": box, "class_index": label, "class_name": demo.CATEGORIES[label],
            "score": float(predictions.get_field("scores")[index]),
        }
        if predictions.has_field("mask_scores"):
            record["mask_score"] = float(predictions.get_field("mask_scores")[index])
        if predictions.has_field("mask"):
            mask = predictions.get_field("mask")[index, 0].numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            record["segmentation"] = rle
        instances.append(record)
    return instances


def write_gallery(output_dir, rows, metadata, log):
    cards = []
    for row in rows:
        cards.append(
            '<article><img src="' + html.escape(row["output_image"]) + '"><p>'
            + html.escape(f'{row["image"]} | threshold={row["threshold"]:g} | '
                          f'{row["detections"]} detections | {row["inference_ms"]:.2f} ms')
            + '</p></article>'
        )
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Week9 CenterMask 实验结果</title><style>
body{margin:32px;background:#f4f6fa;color:#17243a;font:16px system-ui}
h1{font-size:28px}pre{padding:20px;background:#152238;color:#d8f3e8;white-space:pre-wrap;word-break:break-all;border-radius:8px}
section{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
article{margin:0;background:white;padding:10px;border-radius:8px}img{width:100%}p{font-size:13px}
</style><h1>Week9 · CenterMask 实例分割与阈值实验</h1>
<p>原始 CenterMask-Lite / MobileNetV2；以下命令与耗时来自本次真实运行日志。</p>'''
    text += '<pre>' + html.escape(log) + '</pre><section>' + ''.join(cards) + '</section>'
    text += '<p>' + html.escape(f'Checkpoint SHA256: {metadata["checkpoint_sha256"]}') + '</p></html>'
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def check_finite_head_outputs(module, inputs, outputs):
    for pyramid in outputs:
        for tensor in pyramid:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Nonfinite FCOS outputs; check the checkpoint and normalization epsilon")


def run(args):
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {args.weights}. Run Week9/download_weights.py first.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    paths = find_images(args.input, args.limit)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    config = cfg.clone()
    config.merge_from_file(str(args.config_file))
    config.merge_from_list(args.opts)
    config.MODEL.DEVICE = args.device
    config.MODEL.WEIGHT = str(args.weights.resolve())
    config.OUTPUT_DIR = str(args.output_dir.resolve())
    if not config.MODEL.FCOS_ON or not config.MODEL.FCOS_MASK or not config.MODEL.MASK_ON:
        raise ValueError("This experiment requires a CenterMask configuration with FCOS and masks enabled")
    if config.MODEL.FCOS.INFERENCE_TH > min(args.thresholds):
        raise ValueError("MODEL.FCOS.INFERENCE_TH must not exceed the lowest display threshold")
    config.freeze()
    print("COMMAND:", shlex.join([sys.executable, *sys.argv]), flush=True)
    print(f"ENV: Python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__}")
    backend = "torchvision inference fallback" if getattr(_C, "IS_TORCHVISION_FALLBACK", False) else "native C++/CUDA"
    print(f"DEVICE: {args.device} | operators: {backend} | threads: {args.threads}")
    print(f"MODEL: {args.config_file.name} | pretrained: {args.weights.name}")
    load_start = perf_counter()
    demo = COCODemo(config, confidence_threshold=min(args.thresholds), strict_weights=True)
    demo.model.rpn.head.register_forward_hook(check_finite_head_outputs)
    model_load_ms = (perf_counter() - load_start) * 1000
    print(f"WEIGHTS: strict=True, all model parameters matched; load={model_load_ms:.2f} ms")
    with args.weights.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": args.device, "operators": backend, "threads": args.threads,
        "checkpoint": str(args.weights.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "finite_head_outputs_checked": True,
        "config": str(args.config_file.resolve()), "thresholds": args.thresholds,
        "warmup": args.warmup, "repeats": args.repeats, "model_load_ms": model_load_ms,
        "timing_scope": "preprocess + model forward + transfer to CPU + mask projection; excludes image I/O, rendering and saving",
        "selection": "score > threshold (mask_scores when present, otherwise scores)",
        "images": [str(path.resolve()) for path in paths],
    }
    (args.output_dir / "resolved_config.yaml").write_text(config.dump(), encoding="utf-8")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Cannot decode image: {paths[0]}")
    with torch.inference_mode():
        for _ in range(args.warmup):
            demo.compute_prediction(first)
        if args.device == "cuda":
            torch.cuda.synchronize()
        print(f"WARMUP: {args.warmup}; repeats per image/threshold: {args.repeats}")
        rows = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode image: {path}")
            for threshold in args.thresholds:
                durations = []
                for _ in range(args.repeats):
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    start = perf_counter()
                    predictions = demo.compute_prediction(image)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    durations.append((perf_counter() - start) * 1000)
                demo.confidence_threshold = threshold
                selected = demo.select_top_predictions(predictions)
                folder = args.output_dir / f"threshold_{threshold:g}"
                folder.mkdir(parents=True, exist_ok=True)
                output_image = folder / f"{path.name}_result.png"
                if not cv2.imwrite(str(output_image), render(demo, image, selected)):
                    raise OSError(f"Could not save image: {output_image}")
                instances = serialize_instances(demo, selected)
                (folder / f"{path.name}_predictions.json").write_text(json.dumps(
                    {"image": str(path.resolve()), "threshold": threshold,
                     "inference_ms": durations, "instances": instances}, indent=2
                ), encoding="utf-8")
                class_counts = dict(Counter(record["class_name"] for record in instances))
                row = {"image": path.name, "threshold": threshold, "detections": len(selected),
                       "inference_ms": statistics.mean(durations),
                       "class_counts": json.dumps(class_counts, sort_keys=True),
                       "output_image": str(output_image.relative_to(args.output_dir))}
                rows.append(row)
                print(f'{path.name} | conf_th={threshold:g} | detections={len(selected):2d} | '
                      f'{row["inference_ms"]:.2f} ms | {row["class_counts"]}')
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("NUMERICS: all FCOS classification, regression and centerness outputs were finite")
    print(f"SAVED: {len(rows)} visualizations, predictions JSON, metrics.csv, environment.json and resolved_config.yaml")
    return rows, metadata


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            rows, metadata = run(args)
    write_gallery(args.output_dir, rows, metadata, (args.output_dir / "run.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
