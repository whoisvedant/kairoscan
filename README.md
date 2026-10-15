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

### Option A — fast path, using the small pre-cropped Kaggle sample (~2GB)
Use this first, to confirm everything works end-to-end quickly.
Unzip the Kaggle "FaceForensics++ Videos Cropped Faces" download into
`data/raw/archive/` (so you have `data/raw/archive/real/` and
`data/raw/archive/fake/`), then run:

```bash
# Step 1: re-index the already-cropped images into the standard manifest format
python src/prepare_cropped_subset.py --archive_dir data/raw/archive --out_dir data/processed

# Step 2: create the leak-proof train/val/test split
python src/split_dataset.py --manifest data/processed/manifest.json --out data/processed/splits.json

# Step 3 (GPU laptop): train the CNN baseline
python src/train_baseline.py --epochs 5 --batch_size 32
```

### Option B — full path, using the real raw-video FaceForensics++ dataset (~18GB)
Use this once Option A is confirmed working, for your actual reported results.
Unzip the full dataset into `data/raw/` so you have `data/raw/original/`,
`data/raw/Deepfakes/`, etc., then run:

```bash
# Step 1: extract face-cropped frames from raw videos (runs MTCNN face detection)
python src/data_prep.py --raw_dir data/raw --out_dir data/processed --n_samples 24

# Step 2: create the leak-proof train/val/test split
python src/split_dataset.py --manifest data/processed/manifest.json --out data/processed/splits.json

# Step 3 (GPU laptop): train the CNN baseline
python src/train_baseline.py --epochs 10 --batch_size 32
```

Both options produce the exact same `data/processed/manifest.json` format, so
every script downstream (split, train, evaluate) works unchanged either way.

---

## Stage 2 — temporal models (CNN+LSTM/GRU, CNN+Transformer)

Both models share the same CNN backbone as the baseline, so the comparison
isolates the effect of temporal modeling itself. Run each one separately —
they produce separate checkpoints and separately logged results:

```bash
python src/train_temporal.py --model_type lstm --epochs 10
python src/train_temporal.py --model_type transformer --epochs 10
```

Each run saves its checkpoint to `checkpoints/<model_type>_best.pt` and logs
its best metrics into `docs/results.json`. See `docs/results.md` for the
human-readable comparison table across all three models.

**Note on batch size:** these models process full video clips (a sequence of
frames at once), so they use far more GPU memory per batch than the baseline.
The default `--batch_size 8` is intentionally smaller than the baseline's 32
— lower it further (e.g. `--batch_size 4`) if you hit an out-of-memory error.

## Stage 3 — robustness evaluation

Re-evaluates an already-trained model (no retraining) on the TEST set under
realistic video degradations: JPEG compression, Gaussian blur, downscaling
(simulating a low-res re-upload), Gaussian noise, and a halved effective
frame rate. Run it against your best-performing model from Stage 2:

```bash
python src/evaluation/robustness_eval.py --model_type lstm
```

(Swap `lstm` for `gru` or `transformer` to evaluate a different checkpoint —
just make sure that checkpoint actually exists in `checkpoints/` first.)

Results print to the terminal and are saved to `docs/robustness_results.json`.
Copy the numbers into the Robustness table in `docs/results.md` once it's done.

## Stage 4 — calibration

Fits a single temperature-scaling parameter on the VALIDATION set (never the
test set), then checks on the TEST set whether it actually improves
confidence reliability (ECE, Brier score) versus the model's raw output.
Also reports what happens if low-confidence predictions (0.4-0.6) are
flagged "uncertain" instead of forced into a REAL/FAKE decision.

```bash
python src/evaluation/calibrate.py --model_type lstm
```

This does not retrain anything — it only fits one extra number on top of an
already-trained checkpoint. Saves that number to
`checkpoints/lstm_temperature.json` for reuse later, and full results to
`docs/calibration_results.json`.

## Current Status

- [x] Stage 0 — data pipeline (extraction + leak-proof split)
- [x] Stage 1 — CNN baseline (ResNet-18) — best val AUC 0.9785
- [x] Stage 2 — temporal models — LSTM 0.9831, Transformer 0.9830
- [x] Stage 3 — robustness evaluation — see docs/results.md (notable AUC/accuracy gap under compression)
- [x] Stage 4 — calibration — temperature scaling didn't help (model was already well-calibrated), but the uncertain band did (97.1% acc on confident subset)
- [ ] Stage 5 — evidence localization + explanation layer

## Stage 5 - Evidence report and localization check

Turns the model's per-frame scores into something a person can read: a verdict
(REAL / FAKE / UNCERTAIN), a confidence label, the suspicious segments of the
video, the highest-scoring face crops and a plain-language explanation.

```powershell
python tests/test_evidence.py
python tests/test_stage5_smoke.py
python src/evaluation/analyze_video.py --model_type lstm --n 6
python src/evaluation/splice_test.py --model_type lstm --n_pairs 100
```

- Reports are written to `outputs/reports/<video>.json` and `.png`.
- `splice_test.py` joins real and fake face crops at a known point and checks that the
  timeline follows the content. FF++ has no manipulated-segment labels, so this is a
  sanity check, not localization accuracy. Results go to `docs/localization_sanity.json`.
- Limits: only the first 16 of the 24 sampled frames are analysed; per-frame scores are
  trained only through mean pooling, so they are not validated boundaries.
- The fitted temperature is used by default; add `--no_temperature` for raw output.

## Stage 5 - Evidence report and localization check

Turns the model's per-frame scores into something a person can read: a verdict
(REAL / FAKE / UNCERTAIN), a confidence label, the suspicious segments of the
video, the highest-scoring face crops and a plain-language explanation.

```powershell
python tests/test_evidence.py
python tests/test_stage5_smoke.py
python src/evaluation/analyze_video.py --model_type lstm --n 6
python src/evaluation/splice_test.py --model_type lstm --n_pairs 100
```

- Reports are written to `outputs/reports/<video>.json` and `.png`.
- `splice_test.py` joins real and fake face crops at a known point and checks that the
  timeline follows the content. FF++ has no manipulated-segment labels, so this is a
  sanity check, not localization accuracy. Results go to `docs/localization_sanity.json`.
- Limits: only the first 16 of the 24 sampled frames are analysed; per-frame scores are
  trained only through mean pooling, so they are not validated boundaries.
- The fitted temperature is used by default; add `--no_temperature` for raw output.

### Plain-English explanation and Grad-CAM

Each report now also has an explanation split into short sections (what the model found,
how consistent the signal was, where on the face it looked, how much to trust it, next step)
and Grad-CAM heatmaps for the three highest-scoring frames.

- Every sentence is built from the model's own numbers; nothing is written by a language model.
- Heatmaps show where the model's attention was strongest. They are not proof of an edit, and
  the face crops are not aligned, so region names ("lower face") are approximate.
- `analyze_video.py` computes heatmaps by default; add `--no_cam` to skip them.
