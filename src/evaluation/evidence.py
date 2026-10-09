"""
evidence.py
-----------
Stage 5 logic: turns a trained model's per-frame scores into the things the
Kairoscan report promises -- a verdict (REAL / FAKE / UNCERTAIN), a list of
suspicious time segments, and a plain-language explanation.

This file contains NO model code and does not import torch, so it can be
read, tested and debugged on its own (see tests/test_evidence.py).

Design rule (from the project report): the explanation may only restate
evidence the model actually produced. Every sentence below is built from
numbers in the report dict (scores, segments, thresholds, coverage) or from
limitations that were measured in earlier stages. Nothing is guessed.
"""

import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Same uncertain band that calibrate.py reported results for.
UNCERTAIN_LOWER = 0.4
UNCERTAIN_UPPER = 0.6


# --------------------------------------------------------------------------
# Time mapping
# --------------------------------------------------------------------------
def frame_time_ranges(
    n_frames: int,
    total_frames: Optional[int] = None,
    fps: Optional[float] = None,
    n_sampled: int = 24,
) -> Optional[List[Tuple[float, float]]]:
    """Seconds covered by each analysed frame, or None if the video's timing
    is unknown (e.g. the small pre-cropped Kaggle sample has no fps).

    data_prep.py samples `n_sampled` frames evenly across the video, so
    sampled frame i represents roughly the slice
    [i * step, (i + 1) * step) where step = total_frames / n_sampled.
    """
    if not total_frames or not fps or fps <= 0:
        return None
    n_eff = max(1, min(n_sampled, int(total_frames)))
    step_s = (total_frames / n_eff) / fps
    duration = total_frames / fps
    return [(i * step_s, min((i + 1) * step_s, duration)) for i in range(n_frames)]


# --------------------------------------------------------------------------
# Scores -> segments
# --------------------------------------------------------------------------
def smooth_scores(scores, window: int = 3) -> np.ndarray:
    """Moving average with correct handling at the edges (edge frames are
    averaged over fewer neighbours instead of being pulled towards zero)."""
    s = np.asarray(scores, dtype=float)
    if window <= 1 or len(s) == 0:
        return s.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window)
    num = np.convolve(s, kernel, mode="same")
    den = np.convolve(np.ones_like(s), kernel, mode="same")
    return num / den


def find_segments(smoothed, threshold: float) -> List[Tuple[int, int]]:
    """Contiguous runs of frames whose smoothed score is >= threshold.
    Returns inclusive (start_idx, end_idx) pairs."""
    n = len(smoothed)
    segments, i = [], 0
    while i < n:
        if smoothed[i] >= threshold:
            j = i
            while j + 1 < n and smoothed[j + 1] >= threshold:
                j += 1
            segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def top_frames(scores, k: int = 3, min_gap: int = 2) -> List[int]:
    """Indices of the k highest-scoring frames, skipping near-duplicates
    (frames closer than `min_gap` to one already chosen)."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    chosen: List[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == k:
            break
    return sorted(chosen)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def verdict_from_probability(p: float, lower: float = UNCERTAIN_LOWER,
                             upper: float = UNCERTAIN_UPPER) -> str:
    if p >= upper:
        return "FAKE"
    if p <= lower:
        return "REAL"
    return "UNCERTAIN"


def confidence_label(p: float, verdict: str) -> str:
    if verdict == "UNCERTAIN":
        return "Low"
    conf = max(p, 1.0 - p)
    if conf >= 0.90:
        return "High"
    if conf >= 0.75:
        return "Moderate"
    return "Low"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _describe_span(start_idx: int, end_idx: int, ranges) -> str:
    if ranges:
        return f"{ranges[start_idx][0]:.1f}-{ranges[end_idx][1]:.1f} s"
    if start_idx == end_idx:
        return f"sampled frame {start_idx + 1}"
    return f"sampled frames {start_idx + 1}-{end_idx + 1}"


def build_report(
    frame_scores,
    overall_prob: float,
    time_ranges: Optional[List[Tuple[float, float]]] = None,
    video_seconds: Optional[float] = None,
    timing_approximate: bool = False,
    threshold: float = 0.5,
    smooth_window: int = 3,
    lower: float = UNCERTAIN_LOWER,
    upper: float = UNCERTAIN_UPPER,
    video_id: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full, JSON-serialisable evidence report.

    frame_scores : calibrated fake-probability for each analysed frame
    overall_prob : calibrated fake-probability for the whole clip
    """
    scores = np.asarray(frame_scores, dtype=float)
    n = len(scores)
    smoothed = smooth_scores(scores, smooth_window)
    seg_idx = find_segments(smoothed, threshold)

    verdict = verdict_from_probability(overall_prob, lower, upper)
    conf = confidence_label(overall_prob, verdict)

    segments = []
    for s, e in seg_idx:
        segments.append({
            "start_idx": s,
            "end_idx": e,
            "start_s": float(time_ranges[s][0]) if time_ranges else None,
            "end_s": float(time_ranges[e][1]) if time_ranges else None,
            "mean_score": float(smoothed[s:e + 1].mean()),
            "peak_score": float(smoothed[s:e + 1].max()),
            "n_frames": e - s + 1,
        })
    segments.sort(key=lambda d: d["mean_score"], reverse=True)

    frames = []
    for i in range(n):
        frames.append({
            "index": i,
            "start_s": float(time_ranges[i][0]) if time_ranges else None,
            "end_s": float(time_ranges[i][1]) if time_ranges else None,
            "score": float(scores[i]),
            "smoothed": float(smoothed[i]),
        })

    analysed_seconds = float(time_ranges[-1][1]) if time_ranges and n else None
    frames_over = int((smoothed >= threshold).sum())

    # ---------------- explanation (template, grounded in the numbers) -------
    risk_txt = f"{overall_prob:.0%}"
    if verdict == "FAKE":
        head = (f"The model rates this video {risk_txt} likely to be manipulated "
                f"(confidence: {conf}).")
        if segments:
            spans = "; ".join(
                _describe_span(sg["start_idx"], sg["end_idx"], time_ranges)
                for sg in segments[:3]
            )
            if frames_over >= 0.8 * n:
                body = (f"The suspicious signal is present across almost the whole "
                        f"analysed clip ({frames_over} of {n} sampled frames scored "
                        f"above {threshold:.0%}) rather than in one short burst.")
            else:
                body = (f"The evidence is concentrated in {spans} "
                        f"({frames_over} of {n} sampled frames scored above "
                        f"{threshold:.0%}).")
        else:
            body = ("No single window stood out; the suspicion is spread thinly "
                    "across the clip.")
    elif verdict == "REAL":
        head = (f"The model rates this video only {risk_txt} likely to be "
                f"manipulated, so it leans REAL (confidence: {conf}).")
        if segments:
            spans = "; ".join(
                _describe_span(sg["start_idx"], sg["end_idx"], time_ranges)
                for sg in segments[:3]
            )
            body = (f"Short windows ({spans}) scored above {threshold:.0%}, but "
                    f"they were not strong enough to change the overall result.")
        else:
            body = f"No window scored above {threshold:.0%}."
    else:
        head = (f"The model cannot decide: its fake-risk of {risk_txt} falls inside "
                f"the uncertain band ({lower:.0%}-{upper:.0%}).")
        body = "Treat this result as inconclusive rather than as REAL or FAKE."
    explanation = f"{head} {body}"

    # ---------------- limitations (all measured or structural) -------------
    limitations = [
        "This is a statistical estimate, not proof that a video is authentic or manipulated.",
        "Detection reliability drops sharply on heavily compressed, blurred or noisy "
        "video (measured in the robustness tests, docs/results.md).",
        "Per-window scores show where the model's evidence is concentrated; the model "
        "was not trained with frame-level labels, so these are not validated "
        "manipulation boundaries.",
    ]
    if video_seconds and analysed_seconds and analysed_seconds < 0.95 * video_seconds:
        limitations.append(
            f"Only the first {analysed_seconds:.1f} s of the {video_seconds:.1f} s "
            f"video were analysed."
        )
    if timing_approximate:
        limitations.append(
            "Timestamps are approximate because face detection failed on some sampled frames."
        )
    if not time_ranges:
        limitations.append(
            "No frame-rate information is available for this input, so evidence is "
            "shown by sampled-frame number instead of seconds."
        )

    return {
        "video_id": video_id,
        "verdict": verdict,
        "fake_risk": float(overall_prob),
        "confidence": conf,
        "threshold": threshold,
        "uncertain_band": [lower, upper],
        "has_timing": bool(time_ranges),
        "analysed_seconds": analysed_seconds,
        "video_seconds": video_seconds,
        "timing_approximate": timing_approximate,
        "frames": frames,
        "segments": segments,
        "representative_frames": top_frames(scores, k=3, min_gap=2),
        "explanation": explanation,
        "limitations": limitations,
        "ground_truth_label": ground_truth,
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
BELOW_COLOR = "#7A8FA6"   # frames under the threshold
ABOVE_COLOR = "#C2452D"   # frames over the threshold


def render_timeline_figure(report: Dict[str, Any], frame_paths, out_path: str) -> None:
    """Evidence-timeline picture: score per frame, suspicious segments shaded,
    the highest-scoring face crops, then the explanation and limitations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    frames = report["frames"]
    thr = report["threshold"]
    has_t = report["has_timing"]

    if has_t:
        centers = [(f["start_s"] + f["end_s"]) / 2 for f in frames]
        widths = [(f["end_s"] - f["start_s"]) * 0.9 for f in frames]
        xlabel = "Time in video (seconds)"
    else:
        centers = [f["index"] + 1 for f in frames]
        widths = [0.9] * len(frames)
        xlabel = "Sampled frame number"
    scores = [f["score"] for f in frames]
    colors = [ABOVE_COLOR if f["smoothed"] >= thr else BELOW_COLOR for f in frames]

    fig = plt.figure(figsize=(10, 9.2), facecolor="white")
    gs = GridSpec(3, 3, height_ratios=[2.3, 1.5, 2.1], hspace=0.85, wspace=0.12,
                  left=0.07, right=0.97, top=0.88, bottom=0.03)

    ax = fig.add_subplot(gs[0, :])
    ax.bar(centers, scores, width=widths, color=colors, edgecolor="none", zorder=2)
    for seg in report["segments"]:
        if has_t:
            x0, x1 = seg["start_s"], seg["end_s"]
        else:
            x0, x1 = seg["start_idx"] + 0.5, seg["end_idx"] + 1.5
        ax.axvspan(x0, x1, color=ABOVE_COLOR, alpha=0.10, zorder=1)
    ax.axhline(thr, color="#444444", linestyle="--", linewidth=1, zorder=3)
    ax.text(ax.get_xlim()[1], thr + 0.02, f" threshold {thr:.0%}", ha="right",
            va="bottom", fontsize=8, color="#444444")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fake probability\n(calibrated)", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        f"Kairoscan evidence report  -  {report['video_id'] or 'video'}",
        fontsize=13, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.925,
        f"{report['verdict']}   |   Fake risk {report['fake_risk']:.0%}   |   "
        f"Confidence: {report['confidence']}",
        ha="center", fontsize=11,
        color=ABOVE_COLOR if report["verdict"] == "FAKE" else "#222222",
    )

    reps = report["representative_frames"]
    img_axes = []
    for col in range(3):
        axi = fig.add_subplot(gs[1, col])
        axi.axis("off")
        img_axes.append(axi)
        if col < len(reps) and reps[col] < len(frame_paths):
            idx = reps[col]
            img = Image.open(frame_paths[idx]).convert("RGB")
            axi.imshow(img)
            f = frames[idx]
            when = (f"{f['start_s']:.1f}-{f['end_s']:.1f} s" if has_t
                    else f"frame {idx + 1}")
            axi.set_title(f"{when}  |  score {f['score']:.0%}", fontsize=9)
    fig.canvas.draw()
    row_top = img_axes[0].get_position().y1
    fig.text(0.07, row_top + 0.055, "Highest-scoring frames", fontsize=9, color="#555555")

    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    lines = ["Explanation"]
    lines += textwrap.wrap(report["explanation"], 105)
    lines += ["", "Limitations"]
    for lim in report["limitations"]:
        lines += textwrap.wrap(lim, 103, initial_indent="- ", subsequent_indent="  ")
    if report.get("ground_truth_label"):
        lines += ["", f"Dataset label (for reference only): {report['ground_truth_label']}"]
    y = 1.0
    for line in lines:
        bold = line in ("Explanation", "Limitations")
        axt.text(0.0, y, line, fontsize=9.5 if bold else 9, fontweight="bold" if bold else "normal",
                 va="top", transform=axt.transAxes, color="#111111" if bold else "#2a2a2a")
        y -= 0.082 if not bold else 0.09

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
