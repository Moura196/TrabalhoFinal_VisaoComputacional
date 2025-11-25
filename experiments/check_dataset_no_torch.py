"""
Lightweight dataset checker that does NOT import PyTorch.

Use this to verify dataset layout, counts, and to save a resized sample image
without needing PyTorch (works when torch import fails due to DLL issues).

Usage:
  python experiments/check_dataset_no_torch.py --data_root data/cats_dogs --sample_size 224

What it prints:
  - Detected class folders and file counts
  - Total number of images
  - Saves a resized sample image to `outputs/check_dataset_no_torch/sample_0.png`
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import os


def gather_classes(root: Path):
    classes = sorted([p for p in root.iterdir() if p.is_dir()])
    info = {}
    for c in classes:
        cnt = sum(1 for _ in c.rglob('*') if _.is_file())
        info[c.name] = cnt
    return info


def gather_paths(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        for p in (root / c).rglob('*'):
            if p.is_file():
                paths.append(p)
                labels.append(class_to_idx[c])
    return paths, labels, classes


def save_sample_image(path: Path, out_path: Path, size: int = 224):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return True
    except Exception as e:
        print(f'Failed to save sample image: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=224)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f'Data root not found: {root}')
        return 1

    info = gather_classes(root)
    if not info:
        print('No class folders found under data root. Expected layout: data_root/<class_name>/images')
        return 1

    total = sum(info.values())
    print('Detected classes and counts:')
    for k, v in info.items():
        print(f'  {k}: {v} files')
    print(f'Total images: {total}')

    paths, labels, classes = gather_paths(root)
    if not paths:
        print('No image files found.')
        return 1

    # Print a few sample paths
    print('\nSample files:')
    for p in paths[:5]:
        print(' ', p)

    # Save resized sample image
    out_dir = Path('outputs/check_dataset_no_torch')
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = save_sample_image(paths[0], out_dir / 'sample_0.png', size=args.sample_size)
    if saved:
        print(f'Wrote resized sample to: {out_dir / "sample_0.png"}')

    # Print simulated tensor shape for a batch
    batch_size = min(8, len(paths))
    print(f'Example batch tensor shape (simulated): ({batch_size}, 3, {args.sample_size}, {args.sample_size})')

    print('\nDataset sanity check complete (no PyTorch required).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
