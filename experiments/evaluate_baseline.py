"""
Evaluate a saved baseline checkpoint on the test set.

Usage:
  python -m experiments.evaluate_baseline --data_root data/cats_dogs --checkpoint outputs/baseline_test/best.pth

Outputs:
  - Prints overall accuracy and classification report
  - Saves `outputs/baseline_test/confusion_matrix.png` and `outputs/baseline_test/predictions.csv`
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None
import torch.multiprocessing as _mp
try:
    _mp.set_sharing_strategy('file_system')
except Exception:
    pass


# Prefer sklearn when available, but provide lightweight fallbacks so evaluation
# runs inside minimal environments (no sklearn installed).
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:  # pragma: no cover - fallback implementation
    def confusion_matrix(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if labels.size == 0:
            return np.zeros((0, 0), dtype=int)
        max_label = labels.max()
        # build matrix sized by max label (assumes contiguous small labels)
        cm = np.zeros((max_label + 1, max_label + 1), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    def classification_report(y_true, y_pred, target_names=None, digits=4):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        if target_names is None:
            target_names = [str(i) for i in range(n_classes)]

        lines = []
        header = f"{'class':<20} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}"
        lines.append(header)
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            lines.append(f"{target_names[i]:<20} {prec:9.{digits}f} {rec:9.{digits}f} {f1:9.{digits}f} {int(support):9d}")

        # overall averages
        supports = cm.sum(axis=1)
        total_support = supports.sum() if supports.size else 0
        if total_support > 0:
            precision_macro = np.mean([float(l.split()[1]) for l in lines[1:]])
        else:
            precision_macro = 0.0
        lines.append('')
        lines.append(f"{'accuracy':<20} {np.sum(np.diag(cm))/total_support if total_support>0 else 0.0:>9.{digits}f} {'':>9} {'':>9} {int(total_support):9d}")
        return '\n'.join(lines)

from data_preprocessing import (
    get_preprocessing_pipeline,
    CustomImageDataset,
    create_data_loaders,
)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    return ckpt


def evaluate(model, loader, device, progress: bool = False, limit: int = None):
    model.eval()
    preds = []
    targets = []
    seen = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            batch_preds = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            targets.extend(labels.numpy().tolist())
            seen += images.size(0)
            if progress and (batch_idx % 10 == 0):
                print(f'Processed batch {batch_idx}, images seen: {seen}')
            if limit is not None and (batch_idx + 1) >= limit:
                print(f'Limite alcançado: parando depois de {batch_idx+1} batches ({seen} imagens)')
                break
    return np.array(preds), np.array(targets)


def plot_confusion(cm, classes, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--progress', action='store_true', help='print progress every 10 batches')
    parser.add_argument('--limit', type=int, default=None, help='stop after N batches (useful for quick smoke-tests)')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the evaluation dataset (useful when sampling small subsets)')
    parser.add_argument('--sample_per_class', type=int, default=None, help='if set, evaluate on up to this many samples per class (stratified sample)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = load_checkpoint(ckpt_path, device)
    # ckpt expected to contain 'model_state_dict' and 'classes' or at least number of classes saved
    # Infer num_classes
    if 'classes' in ckpt:
        num_classes = ckpt['classes']
    else:
        # try to infer from classifier weight shape if present
        state = ckpt.get('model_state_dict', ckpt)
        for k, v in state.items():
            if k.endswith('fc.weight'):
                num_classes = v.shape[0]
                break
        else:
            raise RuntimeError('Cannot infer number of classes from checkpoint')

    print(f'Loading model for {num_classes} classes on device {device}')
    model = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Build test dataset and loader
    val_transform = get_preprocessing_pipeline(mode='eval')
    test_dataset = CustomImageDataset(images=[p for p in Path(args.data_root).rglob('*') if p.is_file() and p.parent.name in ['cat', 'dog']], labels=None, transform=val_transform)
    # Better: use explicit split if present. We'll reuse baseline split logic by reading files as in baseline, but for simplicity we assume test split exists under data_root/test
    # If not, user can pass explicit test dataset.
    # Create loader
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # NOTE: the dataset above doesn't provide labels; instead, rebuild file list with labels using directory names
    # We'll reconstruct targets in-place
    classes = sorted([d.name for d in Path(args.data_root).iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths = []
    targets = []
    # gather per-class files
    per_class_files = {c: [p for p in (Path(args.data_root) / c).rglob('*') if p.is_file()] for c in classes}
    import random
    rnd = random.Random(42)
    for c in classes:
        files = per_class_files[c]
        if args.sample_per_class is not None:
            if len(files) <= args.sample_per_class:
                chosen = list(files)
            else:
                chosen = rnd.sample(files, args.sample_per_class)
        else:
            chosen = list(files)
        # optionally shuffle chosen list so evaluation order varies
        if args.shuffle:
            rnd.shuffle(chosen)
        for p in chosen:
            paths.append(str(p))
            targets.append(class_to_idx[c])

    # Recreate dataset with labels properly
    test_dataset = CustomImageDataset(images=paths, labels=targets, transform=val_transform)
    # Create DataLoader; retry with safe settings if workers fail due to shared-memory/disk limits
    try:
        loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    except RuntimeError as e:
        print('Warning: DataLoader failed with RuntimeError, retrying with num_workers=0 and pin_memory=False. Error:', e)
        loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=args.shuffle,
            num_workers=0,
            pin_memory=False,
        )

    preds, targs = evaluate(model, loader, device, progress=args.progress, limit=args.limit)
    acc = (preds == targs).mean()
    print(f'Overall accuracy: {acc * 100:.2f}%')

    # classification report
    target_names = classes
    print('Classification report:')
    print(classification_report(targs, preds, target_names=target_names, digits=4))

    # confusion matrix
    cm = confusion_matrix(targs, preds)
    out_dir = ckpt_path.parent
    if plt is not None:
        plot_confusion(cm, target_names, out_dir / 'confusion_matrix.png')
    else:
        print('matplotlib not available; skipping confusion matrix plot. Install matplotlib to enable plots.')

    # save predictions
    import csv
    out_csv = out_dir / 'predictions.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'target', 'pred'])
        for p, t, pr in zip(paths, targs, preds):
            writer.writerow([p, t, pr])

    print(f'Saved confusion matrix to: {out_dir / "confusion_matrix.png"}')
    print(f'Saved predictions to: {out_csv}')


if __name__ == '__main__':
    main()
