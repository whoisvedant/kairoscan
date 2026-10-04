"""
split_dataset.py
-----------------
Splits videos (NOT frames) into train / val / test so that no two frames
from the same source video ever land in different splits. This is the
leak-prevention rule from the project plan (section 12).

Reads data/processed/manifest.json (produced by data_prep.py) and writes:
    data/processed/splits.json  -> {"train": [...video_ids], "val": [...], "test": [...]}
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.json")
    parser.add_argument("--out", default="data/processed/splits.json")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Group by source_method so each split gets a proportional mix of
    # real videos and each manipulation method (stratified video-level split).
    by_method = defaultdict(list)
    for entry in manifest:
        by_method[entry["source_method"]].append(entry["video_id"])

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}

    for method, video_ids in by_method.items():
        rng.shuffle(video_ids)
        n = len(video_ids)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)

        splits["train"] += video_ids[:n_train]
        splits["val"] += video_ids[n_train:n_train + n_val]
        splits["test"] += video_ids[n_train + n_val:]

    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"Splits saved -> {args.out}")


if __name__ == "__main__":
    main()
