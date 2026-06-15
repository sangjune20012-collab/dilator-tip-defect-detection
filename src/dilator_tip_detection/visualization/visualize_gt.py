from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_class_names(items: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in items:
        idx, name = item.split("=", 1)
        out[int(idx)] = name
    return out


def parse_class_colors(items: Sequence[str]) -> Dict[int, Tuple[int, int, int]]:
    out: Dict[int, Tuple[int, int, int]] = {}
    for item in items:
        idx, color = item.split("=", 1)
        parts = tuple(int(v) for v in color.split(","))
        if len(parts) != 3:
            raise ValueError(f"Invalid BGR color: {item}")
        out[int(idx)] = parts
    return out


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def put_text(cv_img, text: str, x: int, y: int, color_bgr: Tuple[int, int, int], font_path: Path | None, font_size: int):
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=color_rgb)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        cx *= img_w; cy *= img_h; bw *= img_w; bh *= img_h
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        x2 = min(img_w - 1, int(round(cx + bw / 2)))
        y2 = min(img_h - 1, int(round(cy + bh / 2)))
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def visualize_image(img_path: Path, label_path: Path, save_path: Path, class_names: Dict[int, str], class_colors: Dict[int, Tuple[int, int, int]], font_path: Path | None, font_size: int, skip_empty: bool) -> bool:
    image = read_image(img_path)
    if image is None:
        return False
    height, width = image.shape[:2]
    boxes = load_yolo_labels(label_path, width, height)
    if skip_empty and not boxes:
        return False
    for cls_id, x1, y1, x2, y2 in boxes:
        name = class_names.get(cls_id, f"class_{cls_id}")
        color = class_colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        image = put_text(image, name, x1, max(y1 - font_size - 4, 0), color, font_path, font_size)
    return write_image(save_path, image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO ground-truth boxes.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--class-names", nargs="+", default=["0=미고착파티클", "1=미성형"])
    parser.add_argument("--class-colors", nargs="+", default=["0=255,0,0", "1=0,0,255"])
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=48)
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    parser.add_argument("--include-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = parse_class_names(args.class_names)
    class_colors = parse_class_colors(args.class_colors)
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.image_exts}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        split_dir = args.base_dir / split
        out_split_dir = args.out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        if not split_dir.exists():
            print(f"[WARN] Split directory does not exist: {split_dir}")
            continue
        image_paths = sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        saved = 0
        for img_path in image_paths:
            if visualize_image(img_path, split_dir / f"{img_path.stem}.txt", out_split_dir / img_path.name, class_names, class_colors, args.font_path, args.font_size, skip_empty=not args.include_empty):
                saved += 1
        print(f"[INFO] {split}: saved {saved} visualizations")


if __name__ == "__main__":
    main()
