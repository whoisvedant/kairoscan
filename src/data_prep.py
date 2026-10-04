"""
data_prep.py
------------
Stage 0 of the pipeline: VIDEO -> face-cropped frames on disk.

Input layout expected (matches FaceForensics++ structure):
    data/raw/
        original/           *.mp4   (real videos)
        Deepfakes/          *.mp4   (fake videos, method 1)
        Face2Face/          *.mp4   (fake videos, method 2)
        FaceSwap/           *.mp4
        NeuralTextures/     *.mp4

Output layout:
    data/processed/
        <video_id>/
            frame_0000.jpg
            frame_0001.jpg
            ...
            meta.json        # video_id, label, source_method, fps, n_frames_sampled

IMPORTANT (per the video-level split rule): every frame from one video
stays inside one folder, so downstream splitting by video_id (not by
frame) is trivial and leak-proof.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

LABEL_MAP = {
    "original": 0,          # REAL
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}


def sample_frame_indices(total_frames: int, n_samples: int = 24):
    """Evenly sample n_samples frame indices across the video (16-32 range per plan)."""
    if total_frames <= n_samples:
        return list(range(total_frames))
    step = total_frames / n_samples
    return [int(i * step) for i in range(n_samples)]


def process_video(video_path: Path, out_dir: Path, mtcnn: MTCNN, n_samples: int = 24, image_size: int = 224):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = set(sample_frame_indices(total_frames, n_samples))

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MTCNN detects + crops the largest/most confident face
            face = mtcnn(rgb)
            if face is not None:
                # face tensor is CHW, float, roughly [-1,1] normalized by facenet-pytorch
                face_img = ((face.permute(1, 2, 0).numpy() * 128 + 127.5)).clip(0, 255).astype("uint8")
                out_path = out_dir / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                saved += 1
        frame_idx += 1
    cap.release()
    return saved, fps, total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--n_samples", type=int, default=24, help="frames per video (16-32 recommended)")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=args.image_size, margin=20, post_process=True, device=device)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    manifest = []

    for method_dir in sorted(raw_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        label = LABEL_MAP.get(method_dir.name)
        if label is None:
            print(f"Skipping unrecognized folder: {method_dir.name}")
            continue

        videos = list(method_dir.glob("*.mp4"))
        for video_path in tqdm(videos, desc=method_dir.name):
            video_id = f"{method_dir.name}_{video_path.stem}"
            video_out = out_dir / video_id
            n_saved, fps, total_frames = process_video(video_path, video_out, mtcnn, args.n_samples, args.image_size)

            meta = {
                "video_id": video_id,
                "label": label,
                "source_method": method_dir.name,
                "fps": fps,
                "total_frames": total_frames,
                "frames_saved": n_saved,
            }
            with open(video_out / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} videos processed. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
