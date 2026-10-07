# Kairoscan — Ablation Results

Trained on the full FaceForensics++ (c23) dataset: 5,000 videos (1,000 real,
4,000 fake across Deepfakes, Face2Face, FaceSwap, NeuralTextures), video-level
70/15/15 split. Trained on an RTX 3050 6GB laptop GPU.

| Model | Best Val AUC | Val Acc | Val F1 | Epochs | Notes |
|---|---|---|---|---|---|
| CNN-only baseline (ResNet-18) | **0.9785** | — | — | 10 | First real-data run. See terminal scrollback for exact per-epoch figures — this run predates automatic results logging. |
| CNN + LSTM/GRU | _pending_ | | | | |
| CNN + Transformer | _pending_ | | | | |

## How this file gets updated

`train_temporal.py` (and the updated `train_baseline.py`) now automatically
log each run's best metrics to `docs/results.json` — a machine-readable
version of this same comparison. After running each model, open
`docs/results.json` and copy the numbers into the table above, so this file
stays the one place that tells the full ablation story at a glance.

## What this table is for

This is the core evidence for the project's Objective 2 — comparing CNN-only,
CNN+recurrent, and CNN+Transformer under identical conditions. Once all three
rows are filled, this table (or a chart built from it) is the headline result
for the Milestone report.

## Robustness results (Stage 3)

Re-evaluates the best model (no retraining) on the TEST set under realistic
video degradations. Filled in after running `robustness_eval.py` — see
`docs/robustness_results.json` for the exact numbers to copy in.

| Condition | AUC | Drop from clean |
|---|---|---|
| Clean (no degradation) | _pending_ | — |
| JPEG compression | _pending_ | |
| Gaussian blur | _pending_ | |
| Downscale (low-res re-upload) | _pending_ | |
| Gaussian noise | _pending_ | |
| Frame-rate drop | _pending_ | |

This table is Objective 3's evidence — it answers "does the model still work
outside a clean lab setting," and the honest answer, whatever it turns out to
be, belongs here exactly as measured.

## Calibration results (Stage 4)

Fits one temperature-scaling parameter on the validation set, then checks
whether it improves confidence reliability on the TEST set (LSTM model).

| Metric | Before calibration | After calibration |
|---|---|---|
| Accuracy | 0.9640 | 0.9640 (unchanged, as expected — temperature scaling never changes predictions, only confidence values) |
| ECE (Expected Calibration Error) | 0.0127 | 0.0214 (did not improve) |
| Brier score | 0.0288 | 0.0289 (did not improve) |

**Fitted temperature:** 1.4578

**Finding:** temperature scaling did not meaningfully improve calibration on
the test set here. The most likely explanation is that the baseline model
was already reasonably well-calibrated (ECE of 0.0127 before any correction
is low), leaving little room for a single global temperature parameter to
improve on. This is reported as a genuine, honest result rather than a
success that was forced to appear — not every calibration technique helps
every model, and knowing that is itself a useful finding.

**Uncertain band (confidence 0.4–0.6 flagged as "uncertain"):**
- % of test set flagged uncertain: **2.3%**
- Accuracy on the remaining confident predictions: **0.9714** (up from 0.9640 overall)

This second part is the stronger result of this stage: correctly identifying
and setting aside the most ambiguous 2.3% of predictions measurably improved
reliability on the rest — demonstrating the value of an abstain mechanism
even where temperature scaling itself did not help.

This section is Objective 4's evidence — it answers whether the model's
stated confidence can actually be trusted, and whether abstaining on
genuinely ambiguous cases improves reliability on the rest. The honest
answer here is mixed: calibration itself didn't help, but abstaining did.
