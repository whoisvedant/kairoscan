"""
analyze_video.py
----------------
Stage 5: the Kairoscan report. For each video it produces

    * a calibrated fake-risk score and a REAL / FAKE / UNCERTAIN verdict
    * a per-frame score timeline and the suspicious segments inside it
    * the highest-scoring face crops
    * a plain-language explanation built only from those numbers

Everything is written to outputs/reports/<video_id>.json and .png.

This does not train anything. It loads a checkpoint from Stage 2 and, if
present, the temperature fitted in Stage 4 (checkpoints/<model>_temperature.json).

Usage (from the project root):
    # a mix of real and fake videos picked at random from the test split
    python src/evaluation/analyze_video.py --model_type lstm --n 6

    # specific videos (the folder names inside data/processed/)
    python src/evaluation/analyze_video.py --model_type lstm --video_id Deepfakes_000_003 original_000
"""

import sys
import json
import random
import argparse
from pathlib import Path

# src/ (for dataset, models) and src/evaluation/ (for evidence)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image

from dataset import default_transform, load_split_ids
from models.temporal_model import build_model
from evidence import build_report, frame_time_ranges, render_timeline_figure


# --------------------------------------------------------------------------
# Loading helpers (shared with splice_test.py)
# --------------------------------------------------------------------------
def load_model(model_type, checkpoint, device):
    path = checkpoint or f"checkpoints/{model_type}_best.pt"
    print(f"Loading checkpoint: {path}")
    model = build_model(model_type).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_temperature(model_type, use_temperature=True, ckpt_dir="checkpoints"):
    """Temperature fitted by calibrate.py. Returns (value, description)."""
    if not use_temperature:
        return 1.0, "raw model output (no temperature)"
    path = Path(ckpt_dir) / f"{model_type}_temperature.json"
    if path.exists():
        with open(path) as f:
            t = float(json.load(f)["temperature"])
        return t, f"temperature scaling, T={t:.4f}"
    print(f"WARNING: {path} not found - run calibrate.py first. Using raw model output.")
    return 1.0, "raw model output (no temperature file found)"


def list_frames(video_dir):
    video_dir = Path(video_dir)
    return sorted(video_dir.glob("frame_*.jpg")) + sorted(video_dir.glob("frame_*.png"))


def read_meta(video_dir):
    meta_path = Path(video_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


@torch.no_grad()
def score_clips(model, clips, temperature, device):
    """clips: tensor (B, T, 3, H, W). Returns (overall_probs [B], frame_probs [B, T]),
    both as calibrated fake-probabilities in [0, 1]."""
    overall_logits, window_logits = model(clips.to(device))
    overall = torch.softmax(overall_logits / temperature, dim=1)[:, 1].cpu()
    frames = torch.softmax(window_logits / temperature, dim=2)[:, :, 1].cpu()
    return overall, frames


# --------------------------------------------------------------------------
# One video -> one report
# --------------------------------------------------------------------------
def analyze_video(video_id, model, device, temperature, processed_dir="data/processed",
                  seq_len=16, threshold=0.5, smooth=3, n_sampled=24):
    video_dir = Path(processed_dir) / video_id
    paths = list_frames(video_dir)[:seq_len]
    if not paths:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    meta = read_meta(video_dir)

    transform = default_transform(train=False)
    clip = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).unsqueeze(0)
    overall, frame_probs = score_clips(model, clip, temperature, device)

    fps, total = meta.get("fps"), meta.get("total_frames")
    ranges = frame_time_ranges(len(paths), total, fps, n_sampled)
    video_seconds = (total / fps) if (fps and total) else None
    saved = meta.get("frames_saved")
    approx = bool(ranges) and saved is not None and saved != n_sampled
    truth = {0: "REAL", 1: "FAKE"}.get(meta.get("label"))

    report = build_report(
        frame_scores=frame_probs[0].tolist(),
        overall_prob=float(overall[0]),
        time_ranges=ranges,
        video_seconds=video_seconds,
        timing_approximate=approx,
        threshold=threshold,
        smooth_window=smooth,
        video_id=video_id,
        ground_truth=truth,
    )
    return report, [str(p) for p in paths]


def pick_videos(args):
    if args.video_id:
        return list(args.video_id)
    ids = load_split_ids(args.splits, args.split)
    by_label = {0: [], 1: []}
    for vid in ids:
        meta = read_meta(Path(args.processed_dir) / vid)
        if "label" in meta:
            by_label[meta["label"]].append(vid)
    rng = random.Random(args.seed)
    n_real = max(1, args.n // 2)
    n_fake = max(1, args.n - n_real)
    chosen = (rng.sample(by_label[0], min(n_real, len(by_label[0]))) +
              rng.sample(by_label[1], min(n_fake, len(by_label[1]))))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_type", required=True, choices=["lstm", "gru", "transformer"])
    parser.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<model_type>_best.pt")
    parser.add_argument("--video_id", nargs="+", default=None,
                        help="one or more folder names inside data/processed/")
    parser.add_argument("--split", default="test", help="split to sample from when --video_id is not given")
    parser.add_argument("--n", type=int, default=6, help="how many videos to sample (half real, half fake)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="frames analysed per video; 16 matches how the models were trained")
    parser.add_argument("--n_sampled", type=int, default=24,
                        help="frames per video that data_prep.py sampled (used to map frames to seconds)")
    parser.add_argument("--threshold", type=float, default=0.5, help="frame score counted as suspicious")
    parser.add_argument("--smooth", type=int, default=3, help="moving-average window over frame scores")
    parser.add_argument("--no_temperature", action="store_true",
                        help="use the raw model output instead of the fitted temperature")
    parser.add_argument("--out_dir", default="outputs/reports")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = load_model(args.model_type, args.checkpoint, device)
    temperature, how = load_temperature(args.model_type, not args.no_temperature)
    print(f"Confidence from: {how}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_ids = pick_videos(args)
    print(f"Analysing {len(video_ids)} video(s)...\n")

    rows = []
    for vid in video_ids:
        report, paths = analyze_video(vid, model, device, temperature, args.processed_dir,
                                      args.seq_len, args.threshold, args.smooth, args.n_sampled)
        report["calibration"] = how
        with open(out_dir / f"{vid}.json", "w") as f:
            json.dump(report, f, indent=2)
        render_timeline_figure(report, paths, str(out_dir / f"{vid}.png"))
        rows.append(report)
        print(f"{vid:32s} label={report['ground_truth_label'] or '?':5s} "
              f"verdict={report['verdict']:9s} risk={report['fake_risk']:.0%}")
        print(f"    {report['explanation']}\n")

    decided = [r for r in rows if r["verdict"] != "UNCERTAIN" and r["ground_truth_label"]]
    correct = sum(r["verdict"] == r["ground_truth_label"] for r in decided)
    print(f"Done. {len(rows)} reports saved to {out_dir}/ (.json + .png each)")
    if decided:
        print(f"Of the {len(decided)} decided (non-UNCERTAIN) videos, {correct} matched the dataset label.")
        print("(A handful of videos is a demo, not an evaluation - the test-set numbers are in docs/results.md.)")


if __name__ == "__main__":
    main()
