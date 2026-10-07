"""
robustness_eval.py
-------------------
Stage 3: re-evaluates a trained model on the TEST set under deliberately
degraded versions of the same videos -- JPEG compression, blur, downscaling
(simulating a low-res re-upload), added noise, and a lower effective frame
rate (simulating a dropped-frame upload).

This never retrains anything. It loads an already-trained checkpoint (from
train_baseline.py or train_temporal.py) and re-runs evaluation only, once
per degradation type, so you can see exactly how much performance drops
under each one.

Degradations are implemented directly with OpenCV/numpy rather than a
third-party augmentation library, so this script has no extra version
dependencies beyond what's already installed.

Usage:
    python src/evaluation/robustness_eval.py --model_type lstm
    python src/evaluation/robustness_eval.py --model_type lstm --checkpoint checkpoints/lstm_best.pt
"""

import sys
import json
import argparse
from pathlib import Path

# Make src/ importable (this file lives in src/evaluation/, one level deeper
# than train_baseline.py and train_temporal.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

from dataset import default_transform, load_split_ids
from models.temporal_model import build_model


def apply_degradation(img_bgr: np.ndarray, degradation: str) -> np.ndarray:
    """img_bgr: a single frame as a numpy array (OpenCV's BGR format)."""
    if degradation == "clean":
        return img_bgr

    if degradation == "jpeg_compression":
        ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 25])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if degradation == "gaussian_blur":
        return cv2.GaussianBlur(img_bgr, (9, 9), 0)

    if degradation == "downscale":
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if degradation == "gaussian_noise":
        noise = np.random.normal(0, 20, img_bgr.shape).astype(np.float32)
        noisy = img_bgr.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown degradation: {degradation}")


# "frame_rate_drop" is handled separately (by skipping stored frames), not
# as a per-pixel degradation, so it is not listed in DEGRADATIONS below but
# is added as its own run in main().
DEGRADATIONS = ["clean", "jpeg_compression", "gaussian_blur", "downscale", "gaussian_noise"]


class DegradedVideoClipDataset(Dataset):
    """Same idea as dataset.py's VideoClipDataset, but applies a degradation
    to every frame before it reaches the model, and optionally skips frames
    to simulate a lower effective frame rate."""

    def __init__(self, processed_dir, split_ids, seq_len=16, degradation="clean", frame_stride=1):
        self.processed_dir = Path(processed_dir)
        self.seq_len = seq_len
        self.degradation = degradation
        self.frame_stride = frame_stride
        self.transform = default_transform(train=False)
        self.videos = []

        for video_id in split_ids:
            meta_path = self.processed_dir / video_id / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            self.videos.append((video_id, meta["label"]))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_id, label = self.videos[idx]
        video_dir = self.processed_dir / video_id
        frame_paths = sorted(video_dir.glob("frame_*.jpg")) + sorted(video_dir.glob("frame_*.png"))

        if self.frame_stride > 1:
            frame_paths = frame_paths[::self.frame_stride]

        if len(frame_paths) >= self.seq_len:
            frame_paths = frame_paths[:self.seq_len]
        else:
            frame_paths = frame_paths + [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        frames = []
        for p in frame_paths:
            img_bgr = cv2.imread(str(p))
            img_bgr = apply_degradation(img_bgr, self.degradation)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(img_rgb))
            frames.append(tensor)

        clip = torch.stack(frames, dim=0)
        return clip, torch.tensor(label, dtype=torch.long), video_id


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for clips, labels, _ids in tqdm(loader, desc="eval", leave=False):
            clips, labels = clips.to(device), labels.to(device)
            overall_logits, _window_logits = model(clips)
            probs = torch.softmax(overall_logits, dim=1)[:, 1]
            all_probs += probs.cpu().tolist()
            all_preds += overall_logits.argmax(dim=1).cpu().tolist()
            all_labels += labels.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["lstm", "gru", "transformer"])
    parser.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<model_type>_best.pt")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--results_file", default="docs/robustness_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or f"checkpoints/{args.model_type}_best.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model(args.model_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_ids = load_split_ids(args.splits, "test")
    results = {}

    # Clean + per-pixel degradations
    for degradation in DEGRADATIONS:
        print(f"\n=== {degradation} ===")
        ds = DegradedVideoClipDataset(args.processed_dir, test_ids, seq_len=args.seq_len,
                                       degradation=degradation, frame_stride=1)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        metrics = evaluate(model, loader, device)
        print(f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
        results[degradation] = metrics

    # Frame-rate drop: simulated by taking every 2nd stored frame
    print(f"\n=== frame_rate_drop (half effective frame rate) ===")
    ds = DegradedVideoClipDataset(args.processed_dir, test_ids, seq_len=args.seq_len,
                                   degradation="clean", frame_stride=2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    metrics = evaluate(model, loader, device)
    print(f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
    results["frame_rate_drop"] = metrics

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results[args.model_type] = results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Results logged -> {results_path}")
    print("\nSummary (AUC per condition):")
    for degradation, m in results.items():
        drop = results["clean"]["auc"] - m["auc"]
        print(f"  {degradation:20s} auc={m['auc']:.4f}  (drop from clean: {drop:+.4f})")


if __name__ == "__main__":
    main()
