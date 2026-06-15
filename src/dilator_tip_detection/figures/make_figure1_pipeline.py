from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_class_names(items: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in items:
        idx, name = item.split("=", 1)
        out[int(idx)] = name
    return out


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int, class_names: Dict[int, str]) -> list[dict]:
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
        boxes.append({"class_id": cls_id, "class_name": class_names.get(cls_id, f"class_{cls_id}"), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1})
    return sorted(boxes, key=lambda b: b["class_id"])


def crop_with_margin(img, x1: int, y1: int, x2: int, y2: int, margin_scale: float, min_margin: int):
    height, width = img.shape[:2]
    margin = int(max(min_margin, max(x2 - x1, y2 - y1) * margin_scale))
    nx1 = max(0, x1 - margin); ny1 = max(0, y1 - margin)
    nx2 = min(width, x2 + margin); ny2 = min(height, y2 + margin)
    return img[ny1:ny2, nx1:nx2].copy(), nx1, ny1


def draw_paper_style(image_path: Path, label_path: Path, save_path: Path, class_names: Dict[int, str], font_family: str, margin_scale: float, min_margin: int) -> bool:
    image = read_image(image_path)
    if image is None:
        return False
    height, width = image.shape[:2]
    boxes = load_yolo_labels(label_path, width, height, class_names)
    if not boxes:
        return False
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["axes.unicode_minus"] = False
    n = len(boxes)
    fig = plt.figure(figsize=(12, max(5, 3 * n)))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.2, 1.0])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(image)
    ax_main.axis("off")
    for i, box in enumerate(boxes):
        color = "#1f77b4" if box["class_id"] == 0 else "#d62728"
        ax_main.add_patch(Rectangle((box["x1"], box["y1"]), box["w"], box["h"], fill=False, edgecolor=color, linewidth=2))
        crop, ox, oy = crop_with_margin(image, box["x1"], box["y1"], box["x2"], box["y2"], margin_scale, min_margin)
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(crop)
        ax.axis("off")
        ax.set_title(f"{box['class_name']} ({box['w']}x{box['h']})", fontsize=12)
        ax.add_patch(Rectangle((box["x1"] - ox, box["y1"] - oy), box["w"], box["h"], fill=False, edgecolor=color, linewidth=2))
        con = ConnectionPatch(xyA=(box["x2"], box["y1"]), xyB=(0, 0), coordsA="data", coordsB="axes fraction", axesA=ax_main, axesB=ax, color=color, linewidth=1)
        ax.add_artist(con)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style GT visualizations with zoomed insets.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--class-names", nargs="+", default=["0=미고착파티클", "1=미성형"])
    parser.add_argument("--font-family", type=str, default="Malgun Gothic")
    parser.add_argument("--margin-scale", type=float, default=3.0)
    parser.add_argument("--min-margin", type=int, default=18)
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = parse_class_names(args.class_names)
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.image_exts}
    for split in args.splits:
        split_dir = args.base_dir / split
        out_split_dir = args.out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        if not split_dir.exists():
            continue
        saved = 0
        for image_path in sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]):
            if draw_paper_style(image_path, split_dir / f"{image_path.stem}.txt", out_split_dir / f"{image_path.stem}_paper_style.png", class_names, args.font_family, args.margin_scale, args.min_margin):
                saved += 1
        print(f"[INFO] {split}: saved {saved} paper-style visualizations")


if __name__ == "__main__":
    main()
