"""
Create a balanced subset (non-destructive) by copying images into a new folder.

Usage examples:
  # create balanced subset with up to 1000 images per class (downsample majority)
  python -m experiments.create_balanced_subset --data_root data/cats_dogs --out_root data/subsets/1000_per_class --per_class 1000

  # oversample minority classes by copying with suffix if they have fewer than target
  python -m experiments.create_balanced_subset --data_root data/cats_dogs --out_root data/subsets/1000_per_class_oversample --per_class 1000 --oversample

Notes:
  - This script never deletes original files.
  - It preserves class-folder layout under `out_root`.
  - If your data already has `train/val/test` subfolders, use `--preserve_splits` to keep those splits and sample inside each split.
"""

import argparse
import random
from pathlib import Path
import shutil
from collections import defaultdict


def gather_by_class(data_root: Path, preserve_splits: bool = False):
    data_root = Path(data_root)
    classes = {}
    if preserve_splits:
        # expect data_root/train/class/*.jpg etc
        for split in ('train', 'val', 'test'):
            split_path = data_root / split
            if not split_path.exists():
                continue
            for c in split_path.iterdir():
                if c.is_dir():
                    classes.setdefault((split, c.name), []).extend([p for p in c.rglob('*') if p.is_file()])
        return classes
    else:
        for c in data_root.iterdir():
            if c.is_dir():
                classes[c.name] = [p for p in c.rglob('*') if p.is_file()]
        return classes


def copy_samples(samples, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in samples:
        dst = dst_dir / src.name
        # if dst exists, append numeric suffix
        if dst.exists():
            base = src.stem
            suf = 1
            while True:
                candidate = dst_dir / f"{base}_{suf}{src.suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                suf += 1
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--per_class', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--oversample', action='store_true', help='If class has fewer files than target, oversample by copying files multiple times with suffix')
    parser.add_argument('--preserve_splits', action='store_true', help='Preserve train/val/test splits if present under data_root')
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    groups = gather_by_class(data_root, preserve_splits=args.preserve_splits)
    if not groups:
        print(f'No classes found under {data_root}')
        return

    summary = {}
    for key, files in groups.items():
        if args.preserve_splits:
            split, cls = key
            out_dir = out_root / split / cls
        else:
            cls = key
            out_dir = out_root / cls

        files = list(files)
        n = len(files)
        if n == 0:
            print(f'Warning: class {cls} has no files; skipping')
            continue

        if n >= args.per_class:
            chosen = random.sample(files, args.per_class)
            copy_samples(chosen, out_dir)
            summary[cls] = (n, args.per_class)
        else:
            if args.oversample:
                # copy all and then copy random choices until reach target
                copy_samples(files, out_dir)
                needed = args.per_class - n
                i = 0
                while i < needed:
                    pick = random.choice(files)
                    # copy with suffix to avoid name collisions
                    base = pick.stem
                    suffix = f"_dup{i}"
                    dst = out_dir / f"{base}{suffix}{pick.suffix}"
                    shutil.copy2(pick, dst)
                    i += 1
                summary[cls] = (n, args.per_class)
            else:
                # just copy all available
                copy_samples(files, out_dir)
                summary[cls] = (n, n)

    print('Summary (class: original -> copied):')
    for k, v in summary.items():
        print(f'  {k}: {v[0]} -> {v[1]}')
    print(f'Created subset at: {out_root}')


if __name__ == '__main__':
    main()
