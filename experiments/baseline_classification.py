"""
Baseline classification training script using ResNet50.

Usage:
  - Place your dataset under `data/cats_dogs` with subfolders per class (e.g. `cats/`, `dogs/`), or provide a folder with train/val subfolders.
  - Run: `python experiments/baseline_classification.py --data_root data/cats_dogs --epochs 5`

This script integrates with `data_preprocessing.CustomImageDataset` and `get_preprocessing_pipeline`.
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from data_preprocessing import (
    get_preprocessing_pipeline,
    CustomImageDataset,
    create_data_loaders,
)


def discover_dataset(root: str) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Discover image files and create train/val/test splits.

    If `root` contains `train`/`val` subfolders, use them. Otherwise, assume
    `root/<class_name>/*.jpg` structure and split by 80/10/10.
    Returns tuples of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    def gather_from_split(split_dir: Path):
        paths = []
        labels = []
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for p in (split_dir / c).rglob('*'):
                if p.is_file():
                    paths.append(str(p))
                    labels.append(class_to_idx[c])
        return paths, labels, classes

    # Case 1: structured into train/val/test
    if (root / 'train').exists():
        train_paths, train_labels, classes = gather_from_split(root / 'train')
        val_paths, val_labels, _ = gather_from_split(root / 'val') if (root / 'val').exists() else ([], [])
        test_paths, test_labels, _ = gather_from_split(root / 'test') if (root / 'test').exists() else ([], [])
        # If val/test empty, we'll split train later
        if not val_paths or not test_paths:
            # fall back to splitting train
            pass
        else:
            return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    # Case 2: root/<class> structure — collect and split
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_paths = []
    all_labels = []
    for c in classes:
        for p in (root / c).rglob('*'):
            if p.is_file():
                all_paths.append(str(p))
                all_labels.append(class_to_idx[c])

    # Shuffle deterministically
    combined = list(zip(all_paths, all_labels))
    combined.sort()  # sort for deterministic behavior

    n = len(combined)
    if n == 0:
        raise ValueError('No images found in dataset root')

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = combined[:n_train]
    val = combined[n_train:n_train + n_val]
    test = combined[n_train + n_val:]

    train_paths, train_labels = zip(*train) if train else ([], [])
    val_paths, val_labels = zip(*val) if val else ([], [])
    test_paths, test_labels = zip(*test) if test else ([], [])

    return list(train_paths), list(train_labels), list(val_paths), list(val_labels), list(test_paths), list(test_labels)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(pretrained=True)
    # Replace final fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / (len(loader.dataset) if len(loader.dataset) > 0 else 1)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / (len(loader.dataset) if len(loader.dataset) > 0 else 1)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/cats_dogs', help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/baseline')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Discovering dataset...')
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = discover_dataset(args.data_root)
    classes = sorted({})
    # Infer number of classes from labels
    num_classes = len(set(train_labels + val_labels + test_labels))
    if num_classes == 0:
        raise ValueError('No classes detected')

    print(f'Found {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test images across {num_classes} classes')

    train_transform = get_preprocessing_pipeline(mode='train')
    val_transform = get_preprocessing_pipeline(mode='eval')

    train_dataset = CustomImageDataset(images=train_paths, labels=train_labels, transform=train_transform)
    val_dataset = CustomImageDataset(images=val_paths, labels=val_labels, transform=val_transform) if val_paths else None
    test_dataset = CustomImageDataset(images=test_paths, labels=test_labels, transform=val_transform) if test_paths else None

    loaders = create_data_loaders(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle_train=True)

    model = build_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, loaders['val'], criterion, device) if 'val' in loaders else (0.0, 0.0)
        scheduler.step(val_loss)
        dt = time.time() - t0
        print(f'Epoch {epoch+1}/{args.epochs}  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val acc: {val_acc:.2f}%  Time: {dt:.1f}s')
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'classes': num_classes}, os.path.join(args.output_dir, 'best.pth'))

    # Final evaluation on test set if present
    if 'test' in loaders:
        test_loss, test_acc = validate(model, loaders['test'], criterion, device)
        print(f'Test loss: {test_loss:.4f}  Test acc: {test_acc:.2f}%')
    else:
        test_acc = None

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_acc, 'test_acc': test_acc}, f, indent=2)

    print('Training complete. Outputs saved to', args.output_dir)


if __name__ == '__main__':
    main()
