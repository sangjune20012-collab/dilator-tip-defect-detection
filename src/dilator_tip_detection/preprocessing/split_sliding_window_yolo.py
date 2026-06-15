from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    name: str


def parse_class_map(items: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid class map entry: {item}. Use class_name=id.")
        name, value = item.split("=", 1)
        result[name] = int(value)
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_image(path: Path) -> Optional[np.ndarray]:
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


def get_aabb_from_rbox(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = np.dot(corners, rot_matrix.T) + [cx, cy]
    xmin, ymin = np.min(rotated, axis=0)
    xmax, ymax = np.max(rotated, axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def window_positions(width: int, height: int, patch_size: int, overlap: int) -> List[Tuple[int, int]]:
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")
    y_steps = list(range(0, max(1, height - patch_size + 1), stride))
    x_steps = list(range(0, max(1, width - patch_size + 1), stride))
    if height > patch_size:
        y_steps.append(height - patch_size)
    if width > patch_size:
        x_steps.append(width - patch_size)
    return [(sx, sy) for sy in sorted(set(y_steps)) for sx in sorted(set(x_steps))]


def create_sliding_patches(image: np.ndarray, annotations: Sequence[Dict[str, float]], roi_x1: int, roi_y1: int, patch_size: int, overlap: int, include_empty: bool) -> List[Dict[str, Any]]:
    height, width = image.shape[:2]
    patches: List[Dict[str, Any]] = []
    for sx, sy in window_positions(width, height, patch_size, overlap):
        patch_img = image[sy: sy + patch_size, sx: sx + patch_size].copy()
        if patch_img.shape[:2] != (patch_size, patch_size):
            continue
        patch_labels: List[str] = []
        for ann in annotations:
            xmin, ymin, xmax, ymax = get_aabb_from_rbox(ann["cx"], ann["cy"], ann["w"], ann["h"], ann["angle"])
            local_xmin = xmin - roi_x1 - sx
            local_ymin = ymin - roi_y1 - sy
            local_xmax = xmax - roi_x1 - sx
            local_ymax = ymax - roi_y1 - sy
            center_x = (local_xmin + local_xmax) / 2.0
            center_y = (local_ymin + local_ymax) / 2.0
            if not (0 <= center_x < patch_size and 0 <= center_y < patch_size):
                continue
            local_xmin = max(0.0, local_xmin)
            local_ymin = max(0.0, local_ymin)
            local_xmax = min(float(patch_size), local_xmax)
            local_ymax = min(float(patch_size), local_ymax)
            box_w = local_xmax - local_xmin
            box_h = local_ymax - local_ymin
            if box_w <= 0 or box_h <= 0:
                continue
            final_cx = (local_xmin + local_xmax) / 2.0 / patch_size
            final_cy = (local_ymin + local_ymax) / 2.0 / patch_size
            final_w = box_w / patch_size
            final_h = box_h / patch_size
            patch_labels.append(f"{int(ann['cls'])} {final_cx:.6f} {final_cy:.6f} {final_w:.6f} {final_h:.6f}")
        if include_empty or patch_labels:
            patches.append({"image": patch_img, "labels": patch_labels})
    return patches


def resolve_image_path(image_dir: Path, meta_path: Path, image_extensions: Sequence[str]) -> Optional[Path]:
    candidate = image_dir / meta_path.stem
    if candidate.exists():
        return candidate
    for ext in image_extensions:
        candidate = image_dir / f"{meta_path.stem}{ext if ext.startswith('.') else '.' + ext}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(image_dir: Path, label_dir: Path, image_extensions: Sequence[str]) -> List[Sample]:
    samples: List[Sample] = []
    meta_files = sorted(image_dir.glob("*.json"))
    print(f"[INFO] Metadata files: {len(meta_files)}")
    for meta_path in meta_files:
        try:
            meta = read_json(meta_path)
            image_path = resolve_image_path(image_dir, meta_path, image_extensions)
            if image_path is None:
                continue
            label_name = Path(meta["label_path"][0]).name
            label_path = label_dir / label_name
            if label_path.exists():
                samples.append(Sample(image_path=image_path, label_path=label_path, name=image_path.stem))
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
    return samples


def extract_annotations(label_path: Path, class_map: Dict[str, int]) -> List[Dict[str, float]]:
    objects = read_json(label_path).get("objects", [])
    annotations: List[Dict[str, float]] = []
    for obj in objects:
        class_name = obj.get("class_name")
        if class_name not in class_map:
            continue
        coord = obj.get("annotation", {}).get("coord", {})
        required = ("cx", "cy", "width", "height")
        if not all(key in coord for key in required):
            continue
        annotations.append({
            "cls": class_map[class_name],
            "cx": float(coord["cx"]),
            "cy": float(coord["cy"]),
            "w": float(coord["width"]),
            "h": float(coord["height"]),
            "angle": float(coord.get("angle", coord.get("rotation", 0))),
        })
    return annotations


def split_samples(samples: Sequence[Sample], valid_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Sample]]:
    if not samples:
        return {"train": [], "valid": [], "test": []}
    temp_ratio = valid_ratio + test_ratio
    train_samples, temp_samples = train_test_split(list(samples), test_size=temp_ratio, random_state=seed)
    if not temp_samples:
        return {"train": train_samples, "valid": [], "test": []}
    test_share = test_ratio / temp_ratio
    valid_samples, test_samples = train_test_split(temp_samples, test_size=test_share, random_state=seed)
    return {"train": train_samples, "valid": valid_samples, "test": test_samples}


def write_data_yaml(dst_root: Path, class_names: Sequence[str]) -> None:
    lines = ["path: .", "train: train", "val: valid", "test: test", "", f"nc: {len(class_names)}", f"names: {list(class_names)}"]
    (dst_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset(image_dir: Path, label_dir: Path, dst_root: Path, class_map: Dict[str, int], class_names: Sequence[str], roi: Sequence[int], patch_size: int, overlap: int, include_empty: bool, valid_ratio: float, test_ratio: float, seed: int, image_extensions: Sequence[str], overwrite: bool) -> None:
    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(image_dir, label_dir, image_extensions)
    print(f"[INFO] Valid matched samples: {len(samples)}")
    splits = split_samples(samples, valid_ratio, test_ratio, seed)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    for split_name, split_samples_ in splits.items():
        out_dir = dst_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        patch_count = 0
        for sample in tqdm(split_samples_, desc=f"Processing {split_name}"):
            image = read_image(sample.image_path)
            if image is None:
                continue
            cropped = image[roi_y1:roi_y2, roi_x1:roi_x2]
            annotations = extract_annotations(sample.label_path, class_map)
            patches = create_sliding_patches(cropped, annotations, roi_x1, roi_y1, patch_size, overlap, include_empty)
            for idx, patch in enumerate(patches):
                patch_name = f"{sample.name}_p{idx}"
                write_image(out_dir / f"{patch_name}.jpg", patch["image"])
                (out_dir / f"{patch_name}.txt").write_text("\n".join(patch["labels"]) + ("\n" if patch["labels"] else ""), encoding="utf-8")
                patch_count += 1
        print(f"[INFO] {split_name}: {patch_count} patches")
    write_data_yaml(dst_root, class_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full sliding-window YOLO patch dataset from raw images and JSON labels.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, default=[0, 1000, 2400, 2000], metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=384)
    parser.add_argument("--class-map", nargs="+", default=["미고착파티클=0", "미성형=1"])
    parser.add_argument("--class-names", nargs="+", default=["impurity", "short_shot"])
    parser.add_argument("--valid-ratio", type=float, default=0.3)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    parser.add_argument("--drop-empty", action="store_true", help="Drop background patches. By default, sliding-window mode keeps empty patches.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.image_dir, args.label_dir, args.dst_root, parse_class_map(args.class_map), args.class_names, args.roi, args.patch_size, args.overlap, not args.drop_empty, args.valid_ratio, args.test_ratio, args.seed, args.image_exts, args.overwrite)
    print("[DONE] Patch dataset generation completed.")


if __name__ == "__main__":
    main()
