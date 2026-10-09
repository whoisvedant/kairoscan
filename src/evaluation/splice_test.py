"""
splice_test.py
--------------
A sanity check for Stage 5's evidence timeline.

Problem: FaceForensics++ labels whole videos, so there is no ground truth for
"which seconds are manipulated" and localization accuracy cannot be measured
directly. This script builds a stand-in test with a KNOWN answer:

    real video A  --first half-->  +  fake video B --second half-->   (real -> fake)
    fake video B  --first half-->  +  real video A --second half-->   (fake -> real)

The clips are made from face crops already in data/processed/, then scored by
the trained model exactly as analyze_video.py does. We then ask:

    1. Are frames from the fake half scored higher than frames from the real half?
       (frame-level AUC, mean score on each half)
    2. If we look for the single point where the score changes most, does it land
       near the true splice point?  (hit rate within +/- tolerance frames)

What this does and does NOT show (state this honestly in the report):
    + It checks that per-frame scores follow the content, i.e. the timeline is
      not just noise.
    - It is not accuracy on real partial manipulations. The two halves come from
      different people, so the splice itself is a cue the model could pick up on,
      and the LSTM's memory can make scores lag behind the true boundary.

Usage (from the project root):
    python src/evaluation/splice_test.py --model_type lstm --n_pairs 100
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from analyze_video import load_model, load_temperature, list_frames, read_meta, score_clips
from dataset import default_transform, load_split_ids
from evidence import smooth_scores


def load_fixed_length(video_dir, seq_len, transform):
    """(seq_len, 3, H, W) tensor; pads by repeating the last frame if the
    video has fewer frames (same behaviour as VideoClipDataset)."""
    paths = list_frames(video_dir)[:seq_len]
    if not paths:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    if len(paths) < seq_len:
        paths = paths + [paths[-1]] * (seq_len - len(paths))
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])


def best_change_point(p, rising, min_side=2):
    """Split point c (frames [:c] vs [c:]) with the largest jump in mean score.
    rising=True looks for a score going UP at c (real->fake); False for DOWN."""
    n = len(p)
    best_c, best_d = None, -np.inf
    for c in range(min_side, n - min_side + 1):
        left, right = p[:c].mean(), p[c:].mean()
        d = (right - left) if rising else (left - right)
        if d > best_d:
            best_d, best_c = d, c
    return best_c


def run_splice_test(model, device, temperature, real_ids, fake_ids, processed_dir,
                    seq_len=16, split_at=8, n_pairs=100, seed=42, tolerance=2,
                    smooth_window=3, show_progress=True):
    rng = random.Random(seed)
    transform = default_transform(train=False)
    k, L = split_at, seq_len

    labels_r2f = np.array([0] * k + [1] * (L - k))
    labels_f2r = 1 - labels_r2f

    store = {"pure_real": [], "pure_fake": [], "r2f": [], "f2r": []}

    iterator = range(n_pairs)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="splicing")

    for _ in iterator:
        a_id, b_id = rng.choice(real_ids), rng.choice(fake_ids)
        a = load_fixed_length(Path(processed_dir) / a_id, L, transform)   # real
        b = load_fixed_length(Path(processed_dir) / b_id, L, transform)   # fake
        clips = torch.stack([
            a,                                  # pure real
            b,                                  # pure fake
            torch.cat([a[:k], b[k:]]),          # real -> fake
            torch.cat([b[:k], a[k:]]),          # fake -> real
        ])
        _, frame_p = score_clips(model, clips, temperature, device)
        frame_p = frame_p.numpy()
        for name, row in zip(("pure_real", "pure_fake", "r2f", "f2r"), frame_p):
            store[name].append(row)

    pure_real = np.array(store["pure_real"])
    pure_fake = np.array(store["pure_fake"])
    r2f = np.array(store["r2f"])
    f2r = np.array(store["f2r"])

    def direction_stats(P, labels, rising):
        real_part, fake_part = P[:, labels == 0], P[:, labels == 1]
        hits, peak_hits = 0, 0
        for row in P:
            c = best_change_point(row, rising)
            hits += int(abs(c - k) <= tolerance)
            peak = int(np.argmax(smooth_scores(row, smooth_window)))
            peak_hits += int(labels[peak] == 1)
        return {
            "frame_auc": float(roc_auc_score(np.tile(labels, len(P)), P.ravel())),
            "mean_score_real_half": float(real_part.mean()),
            "mean_score_fake_half": float(fake_part.mean()),
            "boundary_hit_rate": hits / len(P),
            "peak_in_fake_half_rate": peak_hits / len(P),
        }

    results = {
        "n_pairs": n_pairs,
        "seq_len": L,
        "split_at": k,
        "tolerance_frames": tolerance,
        "chance_boundary_hit_rate": float(min(L - 3, 2 * tolerance + 1) / max(1, (L - 3))),
        "unspliced_reference": {
            "mean_score_real_video": float(pure_real.mean()),
            "mean_score_fake_video": float(pure_fake.mean()),
            "frame_auc": float(roc_auc_score(
                np.r_[np.zeros(pure_real.size), np.ones(pure_fake.size)],
                np.r_[pure_real.ravel(), pure_fake.ravel()])),
        },
        "real_to_fake": direction_stats(r2f, labels_r2f, rising=True),
        "fake_to_real": direction_stats(f2r, labels_f2r, rising=False),
    }
    return results


def print_summary(res):
    ref = res["unspliced_reference"]
    print("\n==== Splice sanity test ====")
    print(f"{res['n_pairs']} real/fake pairs, {res['seq_len']} frames per clip, "
          f"splice after frame {res['split_at']}")
    print(f"\nUnspliced reference: real videos average {ref['mean_score_real_video']:.2f}, "
          f"fake videos {ref['mean_score_fake_video']:.2f} (frame AUC {ref['frame_auc']:.3f})")
    for name, label in (("real_to_fake", "REAL -> FAKE"), ("fake_to_real", "FAKE -> REAL")):
        d = res[name]
        print(f"\n{label}")
        print(f"  mean score  real half {d['mean_score_real_half']:.2f}   fake half {d['mean_score_fake_half']:.2f}")
        print(f"  frame AUC (fake half vs real half):            {d['frame_auc']:.3f}")
        print(f"  strongest score change within +/-{res['tolerance_frames']} frames of the splice: "
              f"{d['boundary_hit_rate']:.0%}  (random guessing ~{res['chance_boundary_hit_rate']:.0%})")
        print(f"  highest-scoring window lies in the fake half:  {d['peak_in_fake_half_rate']:.0%}  (random ~50%)")
    print("\nHow to read this: values clearly above the random-guess figures mean the timeline follows "
          "the video's content. It is a sanity check, not accuracy on real partial manipulations "
          "(the splice joins two different people, and the LSTM's memory can blur the boundary).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_type", required=True, choices=["lstm", "gru", "transformer"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--split_at", type=int, default=None, help="default: seq_len // 2")
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_temperature", action="store_true")
    parser.add_argument("--results_file", default="docs/localization_sanity.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = load_model(args.model_type, args.checkpoint, device)
    temperature, how = load_temperature(args.model_type, not args.no_temperature)
    print(f"Confidence from: {how}")

    ids = load_split_ids(args.splits, args.split)
    real_ids, fake_ids = [], []
    for vid in ids:
        meta = read_meta(Path(args.processed_dir) / vid)
        if meta.get("label") == 0:
            real_ids.append(vid)
        elif meta.get("label") == 1:
            fake_ids.append(vid)
    if not real_ids or not fake_ids:
        raise SystemExit(f"Need both real and fake videos in the '{args.split}' split "
                         f"(found {len(real_ids)} real, {len(fake_ids)} fake).")
    print(f"{args.split} split: {len(real_ids)} real and {len(fake_ids)} fake videos available")

    split_at = args.split_at if args.split_at is not None else args.seq_len // 2
    res = run_splice_test(model, device, temperature, real_ids, fake_ids, args.processed_dir,
                          args.seq_len, split_at, args.n_pairs, args.seed, args.tolerance)
    res["calibration"] = how
    print_summary(res)

    path = Path(args.results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    allres = json.load(open(path)) if path.exists() else {}
    allres[args.model_type] = res
    with open(path, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nResults logged -> {path}")


if __name__ == "__main__":
    main()
