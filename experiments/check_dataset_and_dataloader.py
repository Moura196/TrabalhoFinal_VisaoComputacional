"""
Quick dataset and dataloader checker.

Usage:
  python experiments/check_dataset_and_dataloader.py --data_root PATH_TO_DATA

What it does:
  - Inspects `data_root` for either `train/val/test` subfolders or `data_root/<class>/*` layout
  - Prints class names and counts
  - Creates `CustomImageDataset` and a DataLoader, fetches one batch and prints shapes
  - Saves a denormalized sample image to `outputs/check_dataset/sample_0.png` to visually confirm transforms
"""

import argparse
import os
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

from data_preprocessing import (
    get_preprocessing_pipeline,
    CustomImageDataset,
    create_data_loaders,
    denormalize_image,
)


def inspect_folder(root: Path):
    # detect classes
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    info = {}
    for c in classes:
        cnt = sum(1 for _ in (root / c).rglob('*') if (_.is_file()))
        info[c] = cnt
    return classes, info


def gather_all(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        for p in (root / c).rglob('*'):
            if p.is_file():
                paths.append(str(p))
                labels.append(class_to_idx[c])
    return paths, labels, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f'Data root not found: {root}')

    # If contains train/val dirs, report each
    outputs_dir = Path('outputs/check_dataset')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if (root / 'train').exists():
        print('Detected split folders: train/val/test (if present)')
        for split in ['train', 'val', 'test']:
            d = root / split
            if d.exists():
                classes, info = inspect_folder(d)
                print(f'  {split}: {len(classes)} classes')
                for k, v in info.items():
                    print(f'    {k}: {v} files')
    else:
        print('Detected class-per-folder layout under root')
        classes, info = inspect_folder(root)
        print(f'  Found {len(classes)} classes:')
        for k, v in info.items():
            print(f'    {k}: {v} files')

    # Gather all into dataset
    paths, labels, classes = gather_all(root if not (root / 'train').exists() else (root / 'train'))
    print(f'  Total images collected for dataset check: {len(paths)}')

    if len(paths) == 0:
        print('No files found to test. Exiting.')
        return

    # Build dataset and loader
    train_transform = get_preprocessing_pipeline(mode='train')
    ds = CustomImageDataset(images=paths, labels=labels, transform=train_transform)
    loaders = create_data_loaders(train_dataset=ds, batch_size=args.batch_size, num_workers=args.num_workers)

    print('\nFetching one batch from DataLoader...')
    batch = next(iter(loaders['train']))
    images, labs = batch
    print(f'  Images tensor shape: {images.shape}')
    print(f'  Labels tensor shape: {labs.shape}')

    # Save first image denormalized for visual check
    img_tensor = images[0]
    denorm = denormalize_image(img_tensor)
    save_path = outputs_dir / 'sample_0.png'
    # denorm is HWC in [0,1]
    denorm_img = Image.fromarray((denorm * 255).astype('uint8'))
    denorm_img.save(save_path)
    print(f'  Saved denormalized sample to: {save_path}')

    print('\nDataset/dataloader check passed if shapes are correct and sample image looks OK.')


if __name__ == '__main__':
    main()
