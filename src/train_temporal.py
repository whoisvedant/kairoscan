"""
train_temporal.py
------------------
Stage 2: trains either the CNN+LSTM/GRU model or the CNN+Transformer model
on full video clips (sequences of frames), instead of single frames like
train_baseline.py.

This is arm 2 (--model_type lstm or gru) or arm 3 (--model_type transformer)
of the project's core ablation experiment: does temporal modeling improve
detection over the CNN-only baseline, and does attention improve further
over recurrence?

Usage:
    python src/train_temporal.py --model_type lstm --epochs 10
    python src/train_temporal.py --model_type transformer --epochs 10
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

from dataset import VideoClipDataset, default_transform, load_split_ids
from models.temporal_model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for clips, labels, _video_ids in tqdm(loader, desc="train" if train else "eval"):
            clips, labels = clips.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            overall_logits, _window_logits = model(clips)
            loss = criterion(overall_logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * clips.size(0)
            probs = torch.softmax(overall_logits, dim=1)[:, 1]
            all_probs += probs.detach().cpu().tolist()
            all_preds += overall_logits.argmax(dim=1).detach().cpu().tolist()
            all_labels += labels.detach().cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, acc, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["lstm", "gru", "transformer"])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--seq_len", type=int, default=16, help="frames per video clip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="kept smaller than the baseline's, since each item here is a full clip of frames")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_file", default="docs/results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    train_ids = load_split_ids(args.splits, "train")
    val_ids = load_split_ids(args.splits, "val")

    train_ds = VideoClipDataset(args.processed_dir, train_ids, seq_len=args.seq_len,
                                 transform=default_transform(train=True))
    val_ds = VideoClipDataset(args.processed_dir, val_ids, seq_len=args.seq_len,
                               transform=default_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(args.model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1
    best_metrics = None
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model_type}_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_metrics = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                             "val_f1": val_f1, "val_auc": val_auc}
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint (val_auc={val_auc:.4f})")

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoint at {ckpt_path}")

    # Log this run's best result alongside any previous runs, so results
    # across all three model types (baseline CNN, LSTM/GRU, Transformer)
    # can be compared later for the ablation table.
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results[args.model_type] = best_metrics
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Result logged -> {results_path}")


if __name__ == "__main__":
    main()
