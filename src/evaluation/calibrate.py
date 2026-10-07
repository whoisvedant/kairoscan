"""
calibrate.py
------------
Stage 4: fits a single temperature-scaling parameter (Guo et al., 2017) on
the VALIDATION set, then checks whether it actually improves the model's
confidence reliability on the held-out TEST set -- comparing Expected
Calibration Error (ECE) and Brier score before vs after.

Also reports what happens if predictions in a low-confidence band are
flagged "uncertain" instead of forced into REAL/FAKE -- the abstain
mechanism from Objective 4.

This does not retrain the underlying model at all; it only fits one extra
scalar (the temperature) on top of an already-trained checkpoint's logits.

Usage:
    python src/evaluation/calibrate.py --model_type lstm
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss
from tqdm import tqdm

from dataset import VideoClipDataset, default_transform, load_split_ids
from models.temporal_model import build_model


class TemperatureScaler(nn.Module):
    """Wraps a single learned scalar T. Dividing logits by T before softmax
    sharpens (T<1) or softens (T>1) the model's confidence without changing
    which class it predicts -- so accuracy is unaffected, only the
    confidence values are corrected."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for clips, labels, _ids in tqdm(loader, desc="collecting logits", leave=False):
            clips = clips.to(device)
            overall_logits, _window_logits = model(clips)
            all_logits.append(overall_logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels, device):
    logits, labels = logits.to(device), labels.to(device)
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard binned ECE for binary classification: how far, on average,
    the model's stated confidence is from its actual accuracy in that
    confidence range."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = np.maximum(probs, 1 - probs)  # how sure the model is of its own prediction
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(float)

    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = correct[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def summarize(probs: np.ndarray, labels: np.ndarray, label: str):
    ece = expected_calibration_error(probs, labels)
    brier = brier_score_loss(labels, probs)
    acc = ((probs >= 0.5).astype(int) == labels).mean()
    print(f"[{label}] acc={acc:.4f}  ECE={ece:.4f}  Brier={brier:.4f}")
    return {"acc": float(acc), "ece": ece, "brier": float(brier)}


def apply_uncertain_band(probs: np.ndarray, labels: np.ndarray, lower=0.4, upper=0.6):
    confidences = np.maximum(probs, 1 - probs)
    is_uncertain = (probs > lower) & (probs < upper)
    pct_uncertain = is_uncertain.mean()

    confident_mask = ~is_uncertain
    if confident_mask.sum() > 0:
        predictions = (probs[confident_mask] >= 0.5).astype(int)
        acc_confident = (predictions == labels[confident_mask]).mean()
    else:
        acc_confident = float("nan")

    print(f"Uncertain band [{lower}, {upper}]: {pct_uncertain*100:.1f}% of test set flagged UNCERTAIN")
    print(f"Accuracy on the remaining confident predictions: {acc_confident:.4f}")
    return {"pct_uncertain": float(pct_uncertain), "acc_on_confident": float(acc_confident)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["lstm", "gru", "transformer"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--uncertain_lower", type=float, default=0.4)
    parser.add_argument("--uncertain_upper", type=float, default=0.6)
    parser.add_argument("--results_file", default="docs/calibration_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or f"checkpoints/{args.model_type}_best.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model(args.model_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_ids = load_split_ids(args.splits, "val")
    test_ids = load_split_ids(args.splits, "test")
    transform = default_transform(train=False)

    val_ds = VideoClipDataset(args.processed_dir, val_ids, seq_len=args.seq_len, transform=transform)
    test_ds = VideoClipDataset(args.processed_dir, test_ids, seq_len=args.seq_len, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("\nCollecting validation logits (used only to fit the temperature)...")
    val_logits, val_labels = collect_logits(model, val_loader, device)

    print("Fitting temperature on validation set...")
    temperature = fit_temperature(val_logits, val_labels, device)
    print(f"Fitted temperature: {temperature:.4f}")

    print("\nCollecting test logits (held out, never used to fit anything)...")
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_labels_np = test_labels.numpy()

    # BEFORE: raw softmax, no calibration
    probs_before = torch.softmax(test_logits, dim=1)[:, 1].numpy()
    before = summarize(probs_before, test_labels_np, "BEFORE calibration")

    # AFTER: logits divided by the fitted temperature, then softmax
    probs_after = torch.softmax(test_logits / temperature, dim=1)[:, 1].numpy()
    after = summarize(probs_after, test_labels_np, "AFTER calibration ")

    print(f"\nECE change: {before['ece']:.4f} -> {after['ece']:.4f} "
          f"({'improved' if after['ece'] < before['ece'] else 'did not improve'})")
    print(f"Brier change: {before['brier']:.4f} -> {after['brier']:.4f} "
          f"({'improved' if after['brier'] < before['brier'] else 'did not improve'})")

    print(f"\nApplying uncertain band on the CALIBRATED probabilities:")
    uncertain_stats = apply_uncertain_band(probs_after, test_labels_np,
                                            args.uncertain_lower, args.uncertain_upper)

    # Save the fitted temperature on its own, so Stage 5 (evidence/explanation
    # layer) can load just this one number without re-running calibration.
    temp_path = Path("checkpoints") / f"{args.model_type}_temperature.json"
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)
    print(f"\nTemperature value saved -> {temp_path}")

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results[args.model_type] = {
        "temperature": temperature,
        "before": before,
        "after": after,
        "uncertain_band": {"lower": args.uncertain_lower, "upper": args.uncertain_upper, **uncertain_stats},
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results logged -> {results_path}")


if __name__ == "__main__":
    main()
