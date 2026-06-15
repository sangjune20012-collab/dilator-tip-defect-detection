from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Sequence

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def patch_to_raw_stem(filename_stem: str, patch_suffix_pattern: str = r"_p\d+$") -> str:
    return re.sub(patch_suffix_pattern, "", filename_stem)


def collect_files_by_stem(directory: Path, extensions: Sequence[str]) -> dict[str, Path]:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts}


def process_split(patch_root: Path, raw_image_dir: Path, out_root: Path, split: str, image_extensions: Sequence[str], patch_suffix_pattern: str) -> None:
    split_dir = patch_root / split
    out_split_dir = out_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    raw_map = collect_files_by_stem(raw_image_dir, image_extensions)

    if not split_dir.exists():
        print(f"[WARN] Split directory does not exist: {split_dir}")
        return

    patch_files = []
    for ext in image_extensions:
        patch_files.extend(split_dir.glob(f"*{ext if ext.startswith('.') else '.' + ext}"))
    patch_files = sorted(set(patch_files))

    raw_stems = {patch_to_raw_stem(path.stem, patch_suffix_pattern) for path in patch_files}
    copied = 0
    missing = []

    print(f"\n===== {split.upper()} =====")
    print(f"[INFO] Patch images: {len(patch_files)}")
    print(f"[INFO] Unique raw stems: {len(raw_stems)}")

    for raw_stem in sorted(raw_stems):
        src = raw_map.get(raw_stem)
        if src is None:
            missing.append(raw_stem)
            continue
        shutil.copy2(src, out_split_dir / src.name)
        copied += 1

    print(f"[INFO] Copied raw images: {copied}")
    print(f"[INFO] Missing raw images: {len(missing)}")
    if missing:
        print("[INFO] Missing examples:")
        for stem in missing[:10]:
            print(f"  {stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create raw image splits using existing patch dataset splits.")
    parser.add_argument("--patch-root", type=Path, required=True, help="Patch dataset root containing train/valid/test folders.")
    parser.add_argument("--raw-image-dir", type=Path, required=True, help="Directory containing original raw images.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for raw images with the same split as the patch dataset.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--image-exts", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    parser.add_argument("--patch-suffix-pattern", type=str, default=r"_p\d+$")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {args.patch_root}")
    if not args.raw_image_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {args.raw_image_dir}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        process_split(args.patch_root, args.raw_image_dir, args.out_root, split, args.image_exts, args.patch_suffix_pattern)
    print("\n[DONE] Raw image split generation completed.")


if __name__ == "__main__":
    main()
