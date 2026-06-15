from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def collect_images(split_dir: Path, image_extensions: Sequence[str]) -> list[Path]:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_extensions}
    return sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def convert_split(src_root: Path, dst_root: Path, split: str, class_names: Sequence[str], image_extensions: Sequence[str]) -> None:
    src_split = src_root / split
    dst_split = dst_root / split
    if not src_split.exists():
        print(f"[WARN] Split directory does not exist: {src_split}")
        return
    if dst_split.exists():
        shutil.rmtree(dst_split)
    dst_split.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(class_names)],
    }
    ann_id = 0
    image_files = collect_images(src_split, image_extensions)
    print(f"\n===== {split.upper()} =====")
    print(f"[INFO] Images: {len(image_files)}")

    for img_id, image_path in enumerate(tqdm(image_files, desc=f"Converting {split}")):
        image = read_image(image_path)
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue
        height, width = image.shape[:2]
        shutil.copy2(image_path, dst_split / image_path.name)
        coco["images"].append({"id": img_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[WARN] Invalid label format: {label_path}:{line_number}")
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            abs_w = bw * width
            abs_h = bh * height
            x_min = cx * width - abs_w / 2.0
            y_min = cy * height - abs_h / 2.0
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1
    with (dst_split / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Annotations: {len(coco['annotations'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO-format dataset to RF-DETR/COCO-style dataset.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--class-names", nargs="+", default=["impurity", "short_shot"])
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    args.dst_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        convert_split(args.src_root, args.dst_root, split, args.class_names, args.image_exts)
    print("\n[DONE] YOLO to COCO conversion completed.")


if __name__ == "__main__":
    main()
