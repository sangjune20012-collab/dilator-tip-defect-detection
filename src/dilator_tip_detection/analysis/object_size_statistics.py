from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_class_name_map(items: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid class mapping: {item}. Use source_name=display_name.")
        src, dst = item.split("=", 1)
        mapping[src] = dst
    return mapping


def coco_size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def side32_bucket(width: float, height: float) -> str:
    return "both<=32" if width <= 32 and height <= 32 else "one_or_more>32"


def rotated_to_axis_aligned_bbox_size(width: float, height: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    return width * cos_t + height * sin_t, width * sin_t + height * cos_t


def find_objects(obj: Any) -> list[dict]:
    result: list[dict] = []
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "class_name" in x and "annotation" in x:
                result.append(x)
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    walk(obj)
    return result


def safe_mean(values: list[float]) -> float:
    return stats.mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return stats.median(values) if values else 0.0


def analyze_label_dir(label_dir: Path, save_dir: Path, target_classes: set[str], class_name_map: Dict[str, str]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    records_by_class: dict[str, list[dict]] = defaultdict(list)
    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in label directory: {label_dir}")
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON: {json_path}")
            continue
        for obj in find_objects(data):
            class_name = obj.get("class_name")
            if target_classes and class_name not in target_classes:
                continue
            coord = obj.get("annotation", {}).get("coord", {})
            width = coord.get("width")
            height = coord.get("height")
            if width is None or height is None:
                continue
            angle = coord.get("angle", coord.get("rotation", 0))
            aabb_w, aabb_h = rotated_to_axis_aligned_bbox_size(float(width), float(height), float(angle))
            area = aabb_w * aabb_h
            records_by_class[class_name].append({
                "file": json_path.name,
                "class_name": class_name,
                "display_name": class_name_map.get(class_name, class_name),
                "rbox_width": float(width),
                "rbox_height": float(height),
                "angle": float(angle),
                "aabb_width": aabb_w,
                "aabb_height": aabb_h,
                "area": area,
                "coco_size": coco_size_bucket(area),
                "side32_bucket": side32_bucket(aabb_w, aabb_h),
            })
    records = [record for records in records_by_class.values() for record in records]
    if not records:
        print("[WARN] No target records found.")
        return
    csv_path = save_dir / "bbox_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary_path = save_dir / "bbox_statistics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["class_name", "display_name", "count", "mean_area", "median_area", "mean_width", "median_width", "mean_height", "median_height"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, rows in sorted(records_by_class.items()):
            writer.writerow({
                "class_name": class_name,
                "display_name": class_name_map.get(class_name, class_name),
                "count": len(rows),
                "mean_area": round(safe_mean([r["area"] for r in rows]), 3),
                "median_area": round(safe_median([r["area"] for r in rows]), 3),
                "mean_width": round(safe_mean([r["aabb_width"] for r in rows]), 3),
                "median_width": round(safe_median([r["aabb_width"] for r in rows]), 3),
                "mean_height": round(safe_mean([r["aabb_height"] for r in rows]), 3),
                "median_height": round(safe_median([r["aabb_height"] for r in rows]), 3),
            })
    plot_area_distribution(records_by_class, save_dir, class_name_map)
    print(f"[INFO] Wrote: {csv_path}")
    print(f"[INFO] Wrote: {summary_path}")


def plot_area_distribution(records_by_class: dict[str, list[dict]], save_dir: Path, class_name_map: Dict[str, str]) -> None:
    for class_name, rows in records_by_class.items():
        if not rows:
            continue
        areas = [r["area"] for r in rows]
        plt.figure(figsize=(8, 5))
        plt.hist(areas, bins=30)
        plt.title(f"BBox Area Distribution - {class_name_map.get(class_name, class_name)}")
        plt.xlabel("Area (px^2)")
        plt.ylabel("Count")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        out_path = save_dir / f"area_distribution_{class_name_map.get(class_name, class_name)}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze defect bbox size statistics from JSON labels.")
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--target-classes", nargs="+", default=["미고착파티클", "미성형"])
    parser.add_argument("--class-name-map", nargs="+", default=["미고착파티클=impurity", "미성형=short_shot"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_label_dir(args.label_dir, args.save_dir, set(args.target_classes), parse_class_name_map(args.class_name_map))


if __name__ == "__main__":
    main()
