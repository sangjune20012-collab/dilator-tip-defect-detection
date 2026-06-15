from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR on a COCO-format dilator tip dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="COCO dataset root containing train/valid/test folders.")
    parser.add_argument("--pretrain-weights", type=Path, required=True, help="RF-DETR pretrained .pth weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rfdetr/rfdetr_dilator"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--allow-kmp-duplicate", action="store_true", help="Set KMP_DUPLICATE_LIB_OK=TRUE if needed on Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_kmp_duplicate:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if not args.pretrain_weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {args.pretrain_weights}")

    from rfdetr import RFDETRSmall

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = RFDETRSmall(pretrain_weights=str(args.pretrain_weights), num_classes=args.num_classes)
    model.train(
        dataset_dir=str(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=str(args.output_dir),
        save_val_results=True,
        plot_confusion_matrix=True,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
