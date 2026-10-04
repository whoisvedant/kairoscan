"""
train_baseline.py
------------------
Stage 1 of the plan: CNN-only spatial baseline.

Fine-tunes a pretrained ResNet-18 for binary REAL(0) vs FAKE(1)
classification on individual face-cropped frames.

This is your Experiment-1 "CNN alone" arm for the ablation
(CNN vs CNN+temporal vs Transformer) — get this working and evaluated
BEFORE moving on to the temporal models.

Usage:
    python src/train_baseline.py --epochs 10 --batch_size 32
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

from dataset import FrameDataset, default_transform, load_split_ids


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # REAL / FAKE
    return model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, desc="train" if train else "eval"):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs += probs.detach().cpu().tolist()
            all_preds += logits.argmax(dim=1).detach().cpu().tolist()
            all_labels += labels.detach().cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in a small eval batch
    return avg_loss, acc, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ids = load_split_ids(args.splits, "train")
    val_ids = load_split_ids(args.splits, "val")

    train_ds = FrameDataset(args.processed_dir, train_ids, transform=default_transform(train=True))
    val_ds = FrameDataset(args.processed_dir, val_ids, transform=default_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), ckpt_dir / "cnn_baseline_best.pt")
            print(f"  -> saved new best checkpoint (val_auc={val_auc:.4f})")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoint at {ckpt_dir / 'cnn_baseline_best.pt'}")


if __name__ == "__main__":
    main()
