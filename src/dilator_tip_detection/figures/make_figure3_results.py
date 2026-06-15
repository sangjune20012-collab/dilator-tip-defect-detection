from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def parse_label_map(items: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid label map entry: {item}. Use english=korean.")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_color_map(items: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    result: Dict[str, Tuple[int, int, int]] = {}
    for item in items:
        key, value = item.split("=", 1)
        bgr = tuple(int(v) for v in value.split(","))
        if len(bgr) != 3:
            raise ValueError(f"Invalid BGR color: {item}")
        result[key] = bgr
    return result


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


def get_patch_positions(roi_w: int, roi_h: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    y_steps = list(range(0, max(1, roi_h - patch_size + 1), stride))
    x_steps = list(range(0, max(1, roi_w - patch_size + 1), stride))
    if roi_h > patch_size:
        y_steps.append(roi_h - patch_size)
    if roi_w > patch_size:
        x_steps.append(roi_w - patch_size)
    return [(sx, sy) for sy in sorted(set(y_steps)) for sx in sorted(set(x_steps))]


def apply_nms(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float) -> List[int]:
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores_np.argsort()[::-1]
    keep: List[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def put_text_centered(image: np.ndarray, text: str, box_x1: int, box_x2: int, y: int, color_bgr: Tuple[int, int, int], font_path: Path | None, font_size: int) -> np.ndarray:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    tx = max(0, (box_x1 + box_x2) // 2 - text_w // 2)
    bbox = draw.textbbox((tx, y), text, font=font)
    draw.rectangle(bbox, fill=color_rgb)
    draw.text((tx, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_detections(image: np.ndarray, detections: Sequence[Tuple[int, int, int, int, str, float]], label_map: Dict[str, str], color_map: Dict[str, Tuple[int, int, int]], font_path: Path | None, font_size: int) -> np.ndarray:
    for x1, y1, x2, y2, label_en, conf in detections:
        color = color_map.get(label_en, (0, 255, 0))
        text = f"{label_map.get(label_en, label_en)} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        image = put_text_centered(image, text, x1, x2, max(0, y1 - font_size - 20), color, font_path, font_size)
    return image


def crop_roi(image: np.ndarray, roi_x: int, roi_y: int, roi_w: int, roi_h: int, crop_pad_y: int) -> np.ndarray:
    height, width = image.shape[:2]
    y1 = max(0, roi_y - crop_pad_y)
    y2 = min(height, roi_y + roi_h + crop_pad_y)
    x1 = max(0, roi_x)
    x2 = min(width, roi_x + roi_w)
    return image[y1:y2, x1:x2]


def infer_before(model: YOLO, image_path: Path, conf: float, label_map: Dict[str, str], color_map: Dict[str, Tuple[int, int, int]], font_path: Path | None, font_size: int, roi: Sequence[int], crop_pad_y: int) -> np.ndarray:
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    result = model(str(image_path), conf=conf, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append((x1, y1, x2, y2, result.names[int(box.cls[0])], float(box.conf[0])))
    image = draw_detections(image, detections, label_map, color_map, font_path, font_size)
    return crop_roi(image, roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1], crop_pad_y)


def infer_after(model: YOLO, raw_image_path: Path, patch_dir: Path, conf: float, patch_size: int, overlap: int, roi: Sequence[int], nms_iou: float, label_map: Dict[str, str], color_map: Dict[str, Tuple[int, int, int]], font_path: Path | None, font_size: int, crop_pad_y: int) -> np.ndarray:
    image = read_image(raw_image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {raw_image_path}")
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    stride = patch_size - overlap
    positions = get_patch_positions(roi_x2 - roi_x1, roi_y2 - roi_y1, patch_size, stride)
    cls_boxes: dict[int, list] = defaultdict(list)
    cls_scores: dict[int, list] = defaultdict(list)
    cls_labels: dict[int, list] = defaultdict(list)
    stem = raw_image_path.stem
    for idx, (px, py) in enumerate(positions):
        patch_path = patch_dir / f"{stem}_p{idx}.jpg"
        if not patch_path.exists():
            continue
        result = model(str(patch_path), conf=conf, verbose=False)[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            cls_boxes[cls].append([x1 + px + roi_x1, y1 + py + roi_y1, x2 + px + roi_x1, y2 + py + roi_y1])
            cls_scores[cls].append(float(box.conf[0]))
            cls_labels[cls].append(result.names[cls])
    detections = []
    for cls, boxes in cls_boxes.items():
        for keep_idx in apply_nms(boxes, cls_scores[cls], nms_iou):
            x1, y1, x2, y2 = [int(v) for v in boxes[keep_idx]]
            detections.append((x1, y1, x2, y2, cls_labels[cls][keep_idx], cls_scores[cls][keep_idx]))
    image = draw_detections(image, detections, label_map, color_map, font_path, font_size)
    return crop_roi(image, roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1, crop_pad_y)


def stack_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    width = max(top.shape[1], bottom.shape[1])
    def pad(img):
        diff = width - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(50, 50, 50)) if diff > 0 else img
    divider = np.full((6, width, 3), (100, 100, 100), dtype=np.uint8)
    return np.vstack([pad(top), divider, pad(bottom)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create before/after YOLO detection comparison figures.")
    parser.add_argument("--model-before", type=Path, required=True)
    parser.add_argument("--model-after", type=Path, required=True)
    parser.add_argument("--raw-test-dir", type=Path, required=True)
    parser.add_argument("--patch-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, default=[0, 1000, 2400, 2000])
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--crop-pad-y", type=int, default=30)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=80)
    parser.add_argument("--label-map", nargs="+", default=["impurity=미고착파티클", "short_shot=미성형"])
    parser.add_argument("--color-map", nargs="+", default=["impurity=180,119,31", "short_shot=40,39,214"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_before = YOLO(str(args.model_before))
    model_after = YOLO(str(args.model_after))
    label_map = parse_label_map(args.label_map)
    color_map = parse_color_map(args.color_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_images = sorted(args.raw_test_dir.glob("*.jpg"))
    print(f"[INFO] Images: {len(raw_images)}")
    for image_path in raw_images:
        before = infer_before(model_before, image_path, args.conf, label_map, color_map, args.font_path, args.font_size, args.roi, args.crop_pad_y)
        after = infer_after(model_after, image_path, args.patch_test_dir, args.conf, args.patch_size, args.overlap, args.roi, args.nms_iou, label_map, color_map, args.font_path, args.font_size, args.crop_pad_y)
        combined = stack_vertical(before, after)
        out_path = args.output_dir / f"compare_{image_path.name}"
        write_image(out_path, combined)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
