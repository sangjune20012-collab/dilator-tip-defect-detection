from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on a custom dilator tip dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="YOLO pretrained weight path or model name.")
    parser.add_argument("--project", type=Path, default=Path("outputs/yolo"), help="Output project directory.")
    parser.add_argument("--name", type=str, default="yolo11s_dilator", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", type=str, default="0", help="GPU id such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {args.data}")
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
