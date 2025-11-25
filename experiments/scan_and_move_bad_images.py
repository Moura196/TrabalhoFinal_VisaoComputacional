"""
Scan dataset for unreadable/corrupted images and move them to outputs/bad_images.

Usage:
  python experiments/scan_and_move_bad_images.py --data_root data/cats_dogs

This script tries to open each image with Pillow and calls `verify()`.
Corrupted or unreadable images are moved to `outputs/bad_images/<relative_path>`.
"""

import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil


def scan_and_move(root: Path, dest_root: Path):
    root = root.resolve()
    moved = []
    checked = 0
    for p in root.rglob('*'):
        if p.is_file():
            checked += 1
            try:
                with Image.open(p) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError) as e:
                # move file preserving relative path
                rel = p.relative_to(root)
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(p), str(target))
                    moved.append((p, target))
                    print(f"Moved corrupted file: {p} -> {target}")
                except Exception as me:
                    print(f"Failed to move {p}: {me}")

    return checked, moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out_dir', default='outputs/bad_images')
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"Data root not found: {root}")
        return 1

    dest_root = Path(args.out_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root} for unreadable images...")
    checked, moved = scan_and_move(root, dest_root)
    print(f"Checked {checked} files, moved {len(moved)} corrupted files to {dest_root}")
    if moved:
        print("Sample moved files:")
        for src, dst in moved[:10]:
            print(f"  {src} -> {dst}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
