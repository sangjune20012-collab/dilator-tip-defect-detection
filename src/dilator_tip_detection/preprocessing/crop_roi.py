from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class RoiBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def clip_to_image(self, image_width: int, image_height: int) -> Optional["RoiBox"]:
        x1 = max(0, min(self.x1, image_width))
        y1 = max(0, min(self.y1, image_height))
        x2 = max(0, min(self.x2, image_width))
        y2 = max(0, min(self.y2, image_height))
        if x2 <= x1 or y2 <= y1:
            return None
        return RoiBox(x1, y1, x2, y2)


@dataclass
class SplitStats:
    split: str
    total_images: int = 0
    saved_images: int = 0
    empty_labels: int = 0
    failed_images: int = 0
    input_boxes: int = 0
    output_boxes: int = 0


def read_image(image_path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def write_image(image_path: Path, image: np.ndarray) -> bool:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        return False
    encoded.tofile(str(image_path))
    return True


def yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return center_x - box_width / 2, center_y - box_height / 2, center_x + box_width / 2, center_y + box_height / 2


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, image_width: int, image_height: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(xmin, image_width))
    ymin = max(0.0, min(ymin, image_height))
    xmax = max(0.0, min(xmax, image_width))
    ymax = max(0.0, min(ymax, image_height))
    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width <= 0 or box_height <= 0:
        return None
    return (xmin + xmax) / 2.0 / image_width, (ymin + ymax) / 2.0 / image_height, box_width / image_width, box_height / image_height


def clip_box_to_roi(xmin: float, ymin: float, xmax: float, ymax: float, roi: RoiBox) -> Optional[Tuple[float, float, float, float]]:
    clipped = max(xmin, roi.x1), max(ymin, roi.y1), min(xmax, roi.x2), min(ymax, roi.y2)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return boxes
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"[WARN] Invalid label format: {label_path}:{line_number}")
            continue
        try:
            boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            print(f"[WARN] Invalid numeric value: {label_path}:{line_number}")
    return boxes


def convert_boxes_to_roi_yolo(boxes: Sequence[Tuple[int, float, float, float, float]], original_width: int, original_height: int, roi: RoiBox, min_box_width: float, min_box_height: float) -> List[str]:
    new_label_lines: List[str] = []
    for class_id, x_center, y_center, width, height in boxes:
        xmin, ymin, xmax, ymax = yolo_to_xyxy(x_center, y_center, width, height, original_width, original_height)
        clipped = clip_box_to_roi(xmin, ymin, xmax, ymax, roi)
        if clipped is None:
            continue
        local_xmin, local_ymin, local_xmax, local_ymax = clipped[0] - roi.x1, clipped[1] - roi.y1, clipped[2] - roi.x1, clipped[3] - roi.y1
        if local_xmax - local_xmin < min_box_width or local_ymax - local_ymin < min_box_height:
            continue
        yolo_box = xyxy_to_yolo(local_xmin, local_ymin, local_xmax, local_ymax, roi.width, roi.height)
        if yolo_box is None:
            continue
        new_label_lines.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
    return new_label_lines


def collect_image_paths(split_dir: Path, image_extensions: Sequence[str]) -> List[Path]:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_extensions}
    return sorted([path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in exts])


def process_split(src_root: Path, dst_root: Path, split: str, roi: RoiBox, min_box_width: float, min_box_height: float, image_extensions: Sequence[str]) -> SplitStats:
    src_split_dir = src_root / split
    dst_split_dir = dst_root / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    stats = SplitStats(split=split)
    if not src_split_dir.exists():
        print(f"[WARN] Split directory does not exist: {src_split_dir}")
        return stats
    image_paths = collect_image_paths(src_split_dir, image_extensions)
    stats.total_images = len(image_paths)
    print(f"\n===== {split.upper()} =====")
    print(f"[INFO] Images: {stats.total_images}")
    for image_path in image_paths:
        label_path = src_split_dir / f"{image_path.stem}.txt"
        image = read_image(image_path)
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            stats.failed_images += 1
            continue
        image_height, image_width = image.shape[:2]
        clipped_roi = roi.clip_to_image(image_width=image_width, image_height=image_height)
        if clipped_roi is None:
            print(f"[WARN] Invalid ROI for image: {image_path.name}")
            stats.failed_images += 1
            continue
        cropped_image = image[clipped_roi.y1:clipped_roi.y2, clipped_roi.x1:clipped_roi.x2]
        boxes = parse_yolo_label(label_path)
        stats.input_boxes += len(boxes)
        new_label_lines = convert_boxes_to_roi_yolo(boxes, image_width, image_height, clipped_roi, min_box_width, min_box_height)
        stats.output_boxes += len(new_label_lines)
        output_image_path = dst_split_dir / image_path.name
        output_label_path = dst_split_dir / f"{image_path.stem}.txt"
        if not write_image(output_image_path, cropped_image):
            print(f"[WARN] Failed to write image: {output_image_path}")
            stats.failed_images += 1
            continue
        label_text = "\n".join(new_label_lines)
        output_label_path.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        stats.saved_images += 1
        if not new_label_lines:
            stats.empty_labels += 1
    print(f"[INFO] Saved images: {stats.saved_images}")
    print(f"[INFO] Empty labels: {stats.empty_labels}")
    print(f"[INFO] Failed images: {stats.failed_images}")
    print(f"[INFO] Input boxes: {stats.input_boxes}")
    print(f"[INFO] Output boxes: {stats.output_boxes}")
    return stats


def write_data_yaml(dst_root: Path, splits: Sequence[str], class_names: Sequence[str]) -> None:
    train_split = "train" if "train" in splits else splits[0]
    val_split = "valid" if "valid" in splits else train_split
    test_split = "test" if "test" in splits else val_split
    lines = [f"train: {train_split}", f"val: {val_split}", f"test: {test_split}", "", "names:"]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    (dst_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply ROI cropping to a YOLO-format dataset and convert labels to ROI-local YOLO coordinates.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--roi", nargs=4, type=int, default=[0, 1000, 2400, 2000], metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--min-box-w", type=float, default=1.0)
    parser.add_argument("--min-box-h", type=float, default=1.0)
    parser.add_argument("--class-names", nargs="+", default=["impurity", "short_shot"])
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {args.src_root}")
    args.dst_root.mkdir(parents=True, exist_ok=True)
    roi = RoiBox(*args.roi)
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError(f"Invalid ROI coordinates: {args.roi}")
    all_stats = [process_split(args.src_root, args.dst_root, split, roi, args.min_box_w, args.min_box_h, args.image_exts) for split in args.splits]
    write_data_yaml(args.dst_root, args.splits, args.class_names)
    print("\n===== SUMMARY =====")
    print(f"[INFO] Source root: {args.src_root}")
    print(f"[INFO] Destination root: {args.dst_root}")
    print(f"[INFO] ROI: ({roi.x1}, {roi.y1}, {roi.x2}, {roi.y2})")
    print(f"[INFO] Total saved images: {sum(s.saved_images for s in all_stats)}")
    print(f"[INFO] Total failed images: {sum(s.failed_images for s in all_stats)}")
    print(f"[INFO] Total input boxes: {sum(s.input_boxes for s in all_stats)}")
    print(f"[INFO] Total output boxes: {sum(s.output_boxes for s in all_stats)}")
    print("[DONE] ROI-cropped YOLO dataset generation completed.")


if __name__ == "__main__":
    main()
