"""Run the existing BioCLIP experiment with an exact total and live progress."""
import argparse
import hashlib
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm
import train_tomato_small as base


def make_total_splits(root, total, val_n, test_n, seed):
    classes = sorted(p.name for p in root.iterdir()
                     if p.is_dir() and p.name.startswith('Tomato___'))
    if len(classes) != 10:
        raise ValueError(f'Expected 10 tomato classes, found {len(classes)}')
    rng = random.Random(seed)
    seen, pools = set(), []
    for label, name in enumerate(classes):
        files = sorted(p for p in (root / name).rglob('*')
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        rng.shuffle(files)
        rows = []
        for path in tqdm(files, desc=f'Check {name}', unit='image'):
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({'path': str(path.relative_to(root)), 'label': label,
                         'class': name, 'pixel_sha256': digest})
        if len(rows) <= val_n + test_n:
            raise ValueError(f'Not enough unique images in {name}')
        pools.append(rows)
    capacities = [len(rows) - val_n - test_n for rows in pools]
    if total < len(classes) or total > sum(capacities):
        raise ValueError(f'train-total must be between {len(classes)} and {sum(capacities)}')
    # Allocate nearly equally, redistributing the shortfall of smaller classes.
    counts = [0] * len(classes)
    remaining = total
    while remaining:
        for i, capacity in enumerate(capacities):
            if remaining and counts[i] < capacity:
                counts[i] += 1
                remaining -= 1
    splits = {name: [] for name in ('train', 'val', 'test')}
    for name, rows, count in zip(classes, pools, counts):
        splits['train'].extend(rows[:count])
        splits['val'].extend(rows[count:count + val_n])
        splits['test'].extend(rows[count + val_n:count + val_n + test_n])
        print(f'{name}: train={count}, val={val_n}, test={test_n}', flush=True)
    return classes, splits


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train-total', type=int, default=7000)
    options, remaining = parser.parse_known_args()
    if any(arg.split('=')[0] == '--train-per-class' for arg in remaining):
        parser.error('Use --train-total instead of --train-per-class with this script')
    sys.argv = [sys.argv[0]] + remaining
    base.make_splits = lambda root, train_n, val_n, test_n, seed: make_total_splits(
        root, options.train_total, val_n, test_n, seed)
    original_save = base.save_json

    def save_json(path, obj):
        if path.name == 'config.json':
            obj.pop('train_per_class', None)
            obj.update(train_total=options.train_total,
                       allocation='near-balanced, limited by unique images per class')
        original_save(path, obj)

    base.save_json = save_json
    original_epoch = base.run_epoch

    def live_epoch(visual, head, loader, optimizer=None):
        with tqdm(loader, desc='Train' if optimizer is not None else 'Evaluate',
                  unit='batch', dynamic_ncols=True) as progress:
            return original_epoch(visual, head, progress, optimizer)

    base.run_epoch = live_epoch
    base.main()


if __name__ == '__main__':
    main()
