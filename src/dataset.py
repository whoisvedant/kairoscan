"""
dataset.py
----------
Two Dataset classes:

  FrameDataset   -> for the CNN-only baseline (Stage 1). Each item is one
                     face-cropped frame + its video-level label.

  VideoClipDataset -> for the temporal models (Stage 2: CNN+LSTM/GRU,
                     Transformer). Each item is a fixed-length sequence of
                     frames from one video, in order.

Both read from data/processed/<video_id>/frame_XXXX.jpg using the split
produced by split_dataset.py, so train/val/test never mix frames from the
same video.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(image_size=224, train=True):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FrameDataset(Dataset):
    """Flat frame-level dataset for the CNN baseline."""

    def __init__(self, processed_dir, split_ids, transform=None):
        self.processed_dir = Path(processed_dir)
        self.transform = transform or default_transform(train=False)
        self.samples = []  # (frame_path, label)

        for video_id in split_ids:
            video_dir = self.processed_dir / video_id
            meta_path = video_dir / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            label = meta["label"]
            frame_paths = sorted(video_dir.glob("frame_*.jpg")) + sorted(video_dir.glob("frame_*.png"))
            for frame_path in frame_paths:
                self.samples.append((frame_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        img = Image.open(frame_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class VideoClipDataset(Dataset):
    """Video-level dataset returning an ordered sequence of frames — for
    the temporal models (Stage 2) and for window-level scoring (Stage 5)."""

    def __init__(self, processed_dir, split_ids, seq_len=16, transform=None):
        self.processed_dir = Path(processed_dir)
        self.transform = transform or default_transform(train=False)
        self.seq_len = seq_len
        self.videos = []  # (video_id, label)

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

        # pad/truncate to fixed seq_len so we can batch
        if len(frame_paths) >= self.seq_len:
            frame_paths = frame_paths[:self.seq_len]
        else:
            frame_paths = frame_paths + [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        frames = [self.transform(Image.open(p).convert("RGB")) for p in frame_paths]
        clip = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
        return clip, torch.tensor(label, dtype=torch.long), video_id


def load_split_ids(splits_path, split_name):
    with open(splits_path) as f:
        splits = json.load(f)
    return splits[split_name]