from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_class_names(items: Sequence[str]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid class name entry: {item}. Use id=name.")
        idx, name = item.split("=", 1)
        result[int(idx)] = name
    return result


def is_empty_label_file(label_path: Path) -> bool:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return True
    return label_path.read_text(encoding="utf-8", errors="ignore").strip() == ""


def analyze_split(split_dir: Path, class_names: Dict[int, str], image_extensions: Sequence[str]) -> dict:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_extensions}
    image_files = sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if split_dir.exists() else []
    txt_files = sorted(split_dir.glob("*.txt")) if split_dir.exists() else []
    stats = {
        "split": split_dir.name,
        "num_images": len(image_files),
        "normal_images": 0,
        "defect_images": 0,
        "missing_label_images": 0,
        "empty_label_files": 0,
        "class_counts": Counter(),
        "invalid_lines": 0,
        "orphan_txt_files": 0,
    }
    image_stems = {p.stem for p in image_files}
    for img_path in image_files:
        label_path = split_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats["missing_label_images"] += 1
            continue
        if is_empty_label_file(label_path):
            stats["normal_images"] += 1
            stats["empty_label_files"] += 1
            continue
        stats["defect_images"] += 1
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                stats["invalid_lines"] += 1
                continue
            try:
                cls_id = int(parts[0])
                stats["class_counts"][cls_id] += 1
            except ValueError:
                stats["invalid_lines"] += 1
    stats["orphan_txt_files"] = sum(1 for p in txt_files if p.stem not in image_stems and p.name != "data.yaml")
    return stats


def print_stats(all_stats: list[dict], class_names: Dict[int, str]) -> None:
    total = Counter()
    total_class_counts = Counter()
    for s in all_stats:
        print(f"\n===== {s['split'].upper()} =====")
        for key in ["num_images", "normal_images", "defect_images", "missing_label_images", "empty_label_files", "invalid_lines", "orphan_txt_files"]:
            print(f"{key}: {s[key]}")
            total[key] += s[key]
        print("class_counts:")
        for cls_id, count in sorted(s["class_counts"].items()):
            print(f"  {cls_id} ({class_names.get(cls_id, f'class_{cls_id}')}): {count}")
        total_class_counts.update(s["class_counts"])
    print("\n===== TOTAL =====")
    for key, value in total.items():
        print(f"{key}: {value}")
    print("class_counts:")
    for cls_id, count in sorted(total_class_counts.items()):
        print(f"  {cls_id} ({class_names.get(cls_id, f'class_{cls_id}')}): {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO-format train/valid/test dataset statistics.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--class-names", nargs="+", default=["0=impurity", "1=short_shot"])
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    class_names = parse_class_names(args.class_names)
    all_stats = [analyze_split(args.dataset_root / split, class_names, args.image_exts) for split in args.splits]
    print_stats(all_stats, class_names)


if __name__ == "__main__":
    main()
