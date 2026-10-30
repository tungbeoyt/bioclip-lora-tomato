"""BioCLIP visual LoRA: 100 tomato training images and disjoint val/test sets."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def make_splits(root, train_n, val_n, test_n, seed):
    classes = sorted(p.name for p in root.iterdir()
                     if p.is_dir() and p.name.startswith('Tomato___'))
    if len(classes) != 10:
        raise ValueError(f'Expected 10 tomato classes, found {len(classes)}: {classes}')
    rng = random.Random(seed)
    splits = {k: [] for k in ('train', 'val', 'test')}
    seen = set()
    for label, name in enumerate(classes):
        files = sorted(p for p in (root / name).rglob('*')
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        rng.shuffle(files)
        selected = []
        for path in files:
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            selected.append({'path': str(path.relative_to(root)), 'label': label,
                             'class': name, 'pixel_sha256': digest})
            if len(selected) == train_n + val_n + test_n:
                break
        if len(selected) < train_n + val_n + test_n:
            raise ValueError(f'Not enough unique images in {name}')
        splits['train'].extend(selected[:train_n])
        splits['val'].extend(selected[train_n:train_n + val_n])
        splits['test'].extend(selected[train_n + val_n:])
    return classes, splits


class Images(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            x = self.transform(image.convert('RGB'))
        return x, row['label']


def run_epoch(visual, head, loader, optimizer=None):
    training = optimizer is not None
    visual.train(training)
    head.train(training)
    loss_sum, targets, predictions, confidences = 0., [], [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            features = F.normalize(visual(x), dim=-1)
            logits = head(features)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Assert the adapters actually receive gradients (not only the head).
                grads = [p.grad for n, p in visual.named_parameters()
                         if 'lora_B' in n and p.grad is not None]
                if not grads or not any(g.abs().sum().item() > 0 for g in grads):
                    raise RuntimeError('No nonzero LoRA gradients')
                torch.nn.utils.clip_grad_norm_(
                    [p for p in visual.parameters() if p.requires_grad] + list(head.parameters()), 1.)
                optimizer.step()
            loss_sum += loss.item() * len(y)
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            confidences.extend(logits.softmax(1).max(1).values.detach().cpu().tolist())
    return {'loss': loss_sum / len(targets),
            'accuracy': float(np.mean(np.array(targets) == np.array(predictions)))}, targets, predictions, confidences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/cls/plantvillage/color'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/results/tomato_100'))
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--val-per-class', type=int, default=5)
    parser.add_argument('--test-per-class', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.train_per_class, args.val_per_class, args.test_per_class,
           args.epochs, args.batch_size) <= 0:
        parser.error('Sample counts, epochs and batch size must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists():
        raise FileExistsError('Use a new --output-dir to preserve the previous run')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this experiment')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    print(f'GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}', flush=True)
    classes, splits = make_splits(args.data_root, args.train_per_class,
                                 args.val_per_class, args.test_per_class, args.seed)
    save_json(out / 'split_manifest.json', {'data_root': str(args.data_root.resolve()),
                                         'seed': args.seed, 'classes': classes, 'splits': splits})
    save_json(out / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    print('Samples:', {k: len(v) for k, v in splits.items()}, flush=True)
    print('Classes:', classes, flush=True)
    # OpenCLIP returns THREE values. Use the model's official preprocessing.
    model, train_transform, eval_transform = open_clip.create_model_and_transforms(
        'hf-hub:imageomics/bioclip')
    visual = model.visual
    embed_dim = visual.output_dim
    del model  # The text encoder is unnecessary for supervised leaf classification.
    visual.requires_grad_(False)
    # MLP modules execute normal Linear.forward; do not wrap MHA out_proj,
    # whose weight can be consumed directly by PyTorch's attention implementation.
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                      target_modules=['c_fc', 'c_proj'], bias='none')
    visual = inject_adapter_in_model(lora, visual).cuda()
    head = torch.nn.Linear(embed_dim, len(classes)).cuda()
    parameters = [p for p in visual.parameters() if p.requires_grad] + list(head.parameters())
    trainable = sum(p.numel() for p in parameters)
    print(f'Trainable parameters (visual LoRA + head): {trainable:,}', flush=True)
    loaders = {split: DataLoader(Images(args.data_root, rows,
                           train_transform if split == 'train' else eval_transform),
                           batch_size=args.batch_size, shuffle=split == 'train',
                           num_workers=0, generator=torch.Generator().manual_seed(args.seed))
               for split, rows in splits.items()}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    history, best_loss, best_epoch = [], float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train, *_ = run_epoch(visual, head, loaders['train'], optimizer)
        val, *_ = run_epoch(visual, head, loaders['val'])
        row = {'epoch': epoch, 'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
               'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
               'seconds': time.perf_counter() - epoch_start}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss={train['loss']:.4f} acc={train['accuracy']:.2%}"
              f" | val loss={val['loss']:.4f} acc={val['accuracy']:.2%} | {row['seconds']:.1f}s", flush=True)
        if val['loss'] < best_loss:
            best_loss, best_epoch = val['loss'], epoch
            torch.save({'adapter': {n: p.detach().cpu().clone() for n, p in visual.named_parameters()
                                    if p.requires_grad},
                        'head': {n: p.detach().cpu().clone() for n, p in head.state_dict().items()},
                        'classes': classes, 'epoch': epoch, 'base_model': 'hf-hub:imageomics/bioclip',
                        'lora': {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1,
                                 'target_modules': ['c_fc', 'c_proj']}}, out / 'best.pt')
        with (out / 'history.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    checkpoint = torch.load(out / 'best.pt', weights_only=True, map_location='cpu')
    with torch.no_grad():
        named = dict(visual.named_parameters())
        for name, value in checkpoint['adapter'].items():
            named[name].copy_(value)
    head.load_state_dict(checkpoint['head'])
    # Test is evaluated exactly once, after validation-only checkpoint selection.
    test, y, pred, confidence = run_epoch(visual, head, loaders['test'])
    labels = list(range(len(classes)))
    report = classification_report(y, pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    results = {'model': 'BioCLIP ViT-B/16 + visual MLP LoRA + linear head',
               'counts': {k: len(v) for k, v in splits.items()}, 'seed': args.seed,
               'best_epoch': best_epoch, 'best_val_loss': best_loss,
               'test_loss': test['loss'], 'test_accuracy': test['accuracy'],
               'macro_precision': report['macro avg']['precision'],
               'macro_recall': report['macro avg']['recall'],
               'macro_f1': report['macro avg']['f1-score'],
               'weighted_f1': report['weighted avg']['f1-score'],
               'trainable_parameters': trainable, 'gpu': torch.cuda.get_device_name(0),
               'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
               'elapsed_seconds': time.perf_counter() - started,
               'classification_report': report, 'confusion_matrix': cm.tolist(),
               'limitations': 'Small image-level split; exact pixel duplicates excluded, but leaf identity and near duplicates are not verified. Not a field-performance estimate.'}
    save_json(out / 'metrics.json', results)
    text_report = classification_report(y, pred, labels=labels, target_names=classes,
                                         digits=4, zero_division=0)
    (out / 'classification_report.txt').write_text(text_report)
    with (out / 'predictions.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'true_class', 'predicted_class', 'confidence'])
        for row, target, guess, conf in zip(splits['test'], y, pred, confidence):
            writer.writerow([row['path'], classes[target], classes[guess], conf])
    with (out / 'confusion_matrix.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['true / predicted'] + classes)
        writer.writerows([[name] + row.tolist() for name, row in zip(classes, cm)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split)
        ax.axvline(best_epoch, color='gray', linestyle='--', label='best checkpoint')
        ax.set(xlabel='Epoch', ylabel=metric, title=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=160)
    plt.close(fig)
    short_names = [c.removeprefix('Tomato___').replace('Tomato_', '') for c in classes]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set(xticks=labels, yticks=labels, xticklabels=short_names, yticklabels=short_names,
           xlabel='Predicted class', ylabel='True class', title=f'Tomato test set (n={len(y)})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    print(f'\nBest epoch: {best_epoch}; TEST loss={test["loss"]:.4f}, accuracy={test["accuracy"]:.2%}')
    print(text_report)
    print('Confusion matrix (rows=true, columns=predicted):\n', cm)
    print(f'Outputs: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
