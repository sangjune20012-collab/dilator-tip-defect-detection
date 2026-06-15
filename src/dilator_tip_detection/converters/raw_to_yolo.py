from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_class_map(items: Sequence[str]) -> Dict[str, int]:
    class_map: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid class map entry: {item}. Use class_name=id.")
        name, value = item.split("=", 1)
        class_map[name] = int(value)
    return class_map


def get_aabb_from_rbox(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated_corners = np.dot(corners, rot_matrix.T) + np.array([cx, cy], dtype=np.float32)
    xmin, ymin = np.min(rotated_corners, axis=0)
    xmax, ymax = np.max(rotated_corners, axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(xmin, img_w))
    ymin = max(0.0, min(ymin, img_h))
    xmax = max(0.0, min(xmax, img_w))
    ymax = max(0.0, min(ymax, img_h))
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_all_annotation_candidates(obj: Any) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            keys = set(x.keys())
            has_class = any(key in keys for key in ("class_name", "label_name", "name"))
            has_ann = any(key in keys for key in ("annotation", "annotation_type", "coord"))
            if has_class and has_ann:
                results.append(x)
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return results


def extract_class_name(item: Dict[str, Any]) -> Optional[str]:
    for key in ("class_name", "label_name", "name"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def parse_one_object_to_xyxy(item: Dict[str, Any]) -> Optional[Tuple[str, float, float, float, float]]:
    class_name = extract_class_name(item)
    if class_name is None:
        return None
    annotation_type = item.get("annotation_type")
    annotation = item.get("annotation", {})
    coord = annotation.get("coord", item.get("coord", {}))

    if annotation_type == "rbox":
        cx = coord.get("cx")
        cy = coord.get("cy")
        w = coord.get("width")
        h = coord.get("height")
        angle = coord.get("angle", coord.get("rotation", 0))
        if None in (cx, cy, w, h):
            return None
        xmin, ymin, xmax, ymax = get_aabb_from_rbox(float(cx), float(cy), float(w), float(h), float(angle))
        return class_name, xmin, ymin, xmax, ymax

    if annotation_type == "bbox":
        x = coord.get("x", coord.get("left"))
        y = coord.get("y", coord.get("top"))
        w = coord.get("width")
        h = coord.get("height")
        if None in (x, y, w, h):
            return None
        return class_name, float(x), float(y), float(x) + float(w), float(y) + float(h)

    return None


def extract_yolo_lines_from_label_json(label_json: Any, img_w: int, img_h: int, class_map: Dict[str, int]) -> List[str]:
    lines: List[str] = []
    for item in find_all_annotation_candidates(label_json):
        parsed = parse_one_object_to_xyxy(item)
        if parsed is None:
            continue
        class_name, xmin, ymin, xmax, ymax = parsed
        if class_name not in class_map:
            continue
        yolo_box = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if yolo_box is None:
            continue
        x_c, y_c, w, h = yolo_box
        lines.append(f"{class_map[class_name]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines


def resolve_label_json_path(meta: Dict[str, Any], raw_label_dir: Path) -> Optional[Path]:
    label_path_list = meta.get("label_path", [])
    if isinstance(label_path_list, list) and label_path_list:
        cand = raw_label_dir / Path(label_path_list[0]).name
        if cand.exists():
            return cand
    label_id = meta.get("label_id")
    if isinstance(label_id, str):
        cand = raw_label_dir / f"{label_id}.json"
        if cand.exists():
            return cand
    return None


def find_image_path(raw_image_dir: Path, image_name: str, fallback_path: Path) -> Path:
    candidate = raw_image_dir / image_name
    return candidate if candidate.exists() else fallback_path


def process_split(raw_split_root: Path, raw_image_dir: Path, raw_label_dir: Path, out_root: Path, split: str, class_map: Dict[str, int], image_extensions: Sequence[str]) -> None:
    in_split_dir = raw_split_root / split
    out_split_dir = out_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    if not in_split_dir.exists():
        print(f"[WARN] Split directory does not exist: {in_split_dir}")
        return
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_extensions}
    img_files = sorted([p for p in in_split_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    print(f"\n===== {split.upper()} =====")
    print(f"[INFO] Input raw images: {len(img_files)}")
    num_with_obj = 0
    num_without_obj = 0
    num_missing_meta = 0
    num_missing_label = 0

    for img_path in img_files:
        img_name = img_path.name
        stem = img_path.stem
        src_img_path = find_image_path(raw_image_dir, img_name, img_path)
        meta_path = raw_image_dir / f"{img_name}.json"
        shutil.copy2(src_img_path, out_split_dir / img_name)
        out_txt = out_split_dir / f"{stem}.txt"

        if not meta_path.exists():
            num_missing_meta += 1
            num_without_obj += 1
            out_txt.write_text("", encoding="utf-8")
            continue
        meta = load_json(meta_path)
        image_info = meta.get("image_info", {})
        img_w = image_info.get("width")
        img_h = image_info.get("height")
        if img_w is None or img_h is None:
            num_missing_meta += 1
            num_without_obj += 1
            out_txt.write_text("", encoding="utf-8")
            continue
        label_json_path = resolve_label_json_path(meta, raw_label_dir)
        if label_json_path is None:
            num_missing_label += 1
            num_without_obj += 1
            out_txt.write_text("", encoding="utf-8")
            continue
        yolo_lines = extract_yolo_lines_from_label_json(load_json(label_json_path), int(img_w), int(img_h), class_map)
        out_txt.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        if yolo_lines:
            num_with_obj += 1
        else:
            num_without_obj += 1
    print(f"[INFO] Images with objects: {num_with_obj}")
    print(f"[INFO] Images without objects: {num_without_obj}")
    print(f"[INFO] Missing meta: {num_missing_meta}")
    print(f"[INFO] Missing label json: {num_missing_label}")


def write_data_yaml(out_root: Path, splits: Sequence[str], class_names: Sequence[str]) -> None:
    lines = ["path: .", f"train: {'train' if 'train' in splits else splits[0]}", f"val: {'valid' if 'valid' in splits else splits[0]}", f"test: {'test' if 'test' in splits else splits[-1]}", "", "names:"]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw split images and JSON annotations to YOLO format.")
    parser.add_argument("--raw-split-root", type=Path, required=True)
    parser.add_argument("--raw-image-dir", type=Path, required=True)
    parser.add_argument("--raw-label-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--class-map", nargs="+", default=["미고착파티클=0", "미성형=1"])
    parser.add_argument("--class-names", nargs="+", default=["impurity", "short_shot"])
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = parse_class_map(args.class_map)
    for path in (args.raw_split_root, args.raw_image_dir, args.raw_label_dir):
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        process_split(args.raw_split_root, args.raw_image_dir, args.raw_label_dir, args.out_root, split, class_map, args.image_exts)
    write_data_yaml(args.out_root, args.splits, args.class_names)
    print("\n[DONE] Raw split to YOLO conversion completed.")


if __name__ == "__main__":
    main()
