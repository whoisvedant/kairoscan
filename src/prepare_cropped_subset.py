"""
prepare_cropped_subset.py
--------------------------
Stage 0 (fast path) for the small Kaggle "FaceForensics++ Videos Cropped Faces"
sample (~2GB). This dataset already has faces detected and cropped, so there is
no video-reading or MTCNN step needed here -- unlike data_prep.py, which is for
the full raw-video FaceForensics++ dataset.

This script just re-indexes the already-cropped images into the same
data/processed/<video_id>/ + manifest.json format that data_prep.py produces,
so split_dataset.py, dataset.py and train_baseline.py all work completely
unchanged, regardless of which dataset stage you're on.

Expected input layout (as downloaded from Kaggle, unzipped):
    data/raw/archive/
        real/
            995_crops/
                frame_0000.png ... frame_0019.png
            996_crops/
                ...
        fake/
            995_233_crops/
                frame_0000.png ... frame_0019.png
            ...

Output layout (matches data_prep.py exactly):
    data/processed/
        real_995/
            frame_0000.png ... frame_0019.png
            meta.json
        fake_995_233/
            frame_0000.png ... frame_0019.png
            meta.json
        manifest.json
"""

import json
import shutil
import argparse
from pathlib import Path


def index_split(split_dir: Path, label: int, source_method: str, out_dir: Path, manifest: list):
    """Copy every <id>_crops folder under split_dir into out_dir/<source_method>_<id>/."""
    if not split_dir.exists():
        print(f"  (skipping, not found: {split_dir})")
        return

    video_folders = sorted(split_dir.glob("*_crops"))
    for folder in video_folders:
        # "995_crops" -> "995"   |   "995_233_crops" -> "995_233"
        video_id_raw = folder.name.removesuffix("_crops")
        video_id = f"{source_method}_{video_id_raw}"

        out_video_dir = out_dir / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(folder.glob("frame_*.png")) + sorted(folder.glob("frame_*.jpg"))
        for frame_path in frames:
            dest = out_video_dir / frame_path.name
            if not dest.exists():
                shutil.copy2(frame_path, dest)

        meta = {
            "video_id": video_id,
            "label": label,
            "source_method": source_method,
            "fps": None,          # not applicable -- frames were pre-extracted
            "total_frames": None,
            "frames_saved": len(frames),
        }
        with open(out_video_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)

    print(f"  {source_method}: {len(video_folders)} videos indexed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir", default="data/raw/archive",
                         help="Path to the unzipped Kaggle 'archive' folder (contains real/ and fake/)")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    print("Indexing real videos...")
    index_split(archive_dir / "real", label=0, source_method="real", out_dir=out_dir, manifest=manifest)

    print("Indexing fake videos...")
    index_split(archive_dir / "fake", label=1, source_method="fake_sample", out_dir=out_dir, manifest=manifest)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_real = sum(1 for m in manifest if m["label"] == 0)
    n_fake = sum(1 for m in manifest if m["label"] == 1)
    print(f"\nDone. {len(manifest)} videos total ({n_real} real, {n_fake} fake).")
    print(f"Manifest -> {manifest_path}")
    print("\nNext steps:")
    print("  python src/split_dataset.py --manifest data/processed/manifest.json --out data/processed/splits.json")
    print("  python src/train_baseline.py --epochs 5 --batch_size 32")


if __name__ == "__main__":
    main()