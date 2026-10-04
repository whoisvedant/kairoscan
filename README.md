# Kairoscan
**An Evidence-Aware Deepfake Video Detection System**

Milestone 2 project — Vedant Shandilya (S24CSEU1856), Bennett University.

---

## Project Structure

```
kairoscan/
├── README.md                  <- you are here
├── requirements.txt           <- all Python dependencies
├── .gitignore                 <- keeps datasets/checkpoints out of git
│
├── src/                       <- all source code
│   ├── data_prep.py           <- Stage 0: video -> face-cropped frames
│   ├── split_dataset.py       <- Stage 0: leak-proof video-level train/val/test split
│   ├── dataset.py             <- PyTorch Dataset classes (frame-level + clip-level)
│   ├── train_baseline.py      <- Stage 1: CNN-only baseline training
│   ├── models/                <- (Stage 2) temporal model definitions go here
│   └── evaluation/            <- (Stage 3-5) robustness, calibration, localization scripts go here
│
├── data/
│   ├── raw/                   <- original downloaded videos (NOT in git — too large)
│   └── processed/             <- extracted face frames + manifest.json (NOT in git)
│
├── checkpoints/                <- saved trained model weights (NOT in git)
├── logs/                       <- training logs (NOT in git)
├── notebooks/                  <- scratch/exploration notebooks, if any
└── docs/                        <- Milestone reports, diagrams, etc.
```

**Why data/ and checkpoints/ are empty in git:** datasets are tens of GB and model
weights are large binary files — neither belongs in version control. `.gitignore`
is already set up to keep them out automatically. Only code is tracked.

---

## The Two-Laptop Workflow

Code is written and version-controlled on the CPU laptop. Training (which needs a
GPU) happens on the other laptop. Git is the bridge between them — push code from
one, pull it on the other. Datasets and checkpoints stay local to whichever
machine needs them; they're never pushed through git.

```
 CPU laptop                      GPU laptop
 (write code)                    (run training)
     │                                 │
     │   git push ───────────────►     │
     │                                 │
     │   (place/download datasets      │
     │    directly on GPU laptop)      │
     │                                 │
     │              ◄─────── git pull  │   (after making a code fix there, if needed)
```

---

## Setup — run once per laptop

```bash
# 1. Clone the repo (GPU laptop) or just open the folder (CPU laptop)
git clone <your-repo-url> kairoscan
cd kairoscan

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
#    Windows:
venv\Scripts\activate
#    Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Pipeline — run in this order

```bash
# Step 1: extract face-cropped frames from raw videos
python src/data_prep.py --raw_dir data/raw --out_dir data/processed --n_samples 24

# Step 2: create the leak-proof train/val/test split
python src/split_dataset.py --manifest data/processed/manifest.json --out data/processed/splits.json

# Step 3 (GPU laptop only): train the CNN baseline
python src/train_baseline.py --epochs 10 --batch_size 32
```

---

## Current Status

- [x] Stage 0 — data pipeline (extraction + leak-proof split)
- [x] Stage 1 — CNN baseline (ResNet-18)
- [ ] Stage 2 — temporal models (CNN+LSTM/GRU, Transformer)
- [ ] Stage 3 — robustness evaluation
- [ ] Stage 4 — calibration
- [ ] Stage 5 — evidence localization + explanation layer
