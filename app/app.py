"""
app.py
------
Kairoscan web interface. Shows the reports that analyze_video.py writes to
outputs/reports/, with the face crops from data/processed/.

It does not run the model. To add videos to the library, run:
    python src/evaluation/analyze_video.py --model_type lstm --n 12

Start it (from the project root):
    python app/app.py
then open http://127.0.0.1:5000
"""

import re
import json
import argparse
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")

app = Flask(__name__)
app.config["REPORTS_DIR"] = ROOT / "outputs" / "reports"
app.config["FRAMES_DIR"] = ROOT / "data" / "processed"

VERDICT_TEXT = {
    "FAKE": ("Likely manipulated", "fake"),
    "REAL": ("Leans real", "real"),
    "UNCERTAIN": ("Cannot decide", "uncertain"),
}


def _check_id(video_id):
    if not SAFE_ID.match(video_id) or video_id.startswith("."):
        abort(404)


def load_report(video_id):
    _check_id(video_id)
    path = app.config["REPORTS_DIR"] / f"{video_id}.json"
    if not path.exists():
        abort(404)
    with open(path) as f:
        return json.load(f)


def frame_names(video_id, n):
    """File names of the first n face crops of a video, in order."""
    folder = app.config["FRAMES_DIR"] / video_id
    names = sorted(p.name for p in folder.glob("frame_*.jpg")) + \
            sorted(p.name for p in folder.glob("frame_*.png"))
    return names[:n]


def summary(report):
    text, css = VERDICT_TEXT.get(report["verdict"], ("Unknown", "uncertain"))
    names = frame_names(report["video_id"], len(report["frames"]))
    reps = report.get("representative_frames") or [0]
    thumb = names[reps[0]] if names and reps[0] < len(names) else None
    return {
        "id": report["video_id"],
        "verdict": report["verdict"],
        "verdict_text": text,
        "css": css,
        "risk": round(report["fake_risk"] * 100),
        "confidence": report["confidence"],
        "label": report.get("ground_truth_label"),
        "scores": [round(f["score"], 3) for f in report["frames"]],
        "thumb": thumb,
        "names": names,
        "has_segments": bool(report.get("segments")),
    }


@app.route("/")
def index():
    reports = []
    folder = app.config["REPORTS_DIR"]
    if folder.exists():
        for p in sorted(folder.glob("*.json")):
            try:
                with open(p) as f:
                    reports.append(summary(json.load(f)))
            except (OSError, ValueError, KeyError):
                continue  # skip files that are not valid reports
    order = {"FAKE": 0, "UNCERTAIN": 1, "REAL": 2}
    reports.sort(key=lambda r: (order.get(r["verdict"], 3), -r["risk"], r["id"]))
    hero = next((r for r in reports if r["verdict"] == "FAKE" and r["has_segments"]
                 and r["names"] and min(r["scores"]) < 0.5), None) \
        or next((r for r in reports if r["names"]), None)
    counts = {k: sum(1 for r in reports if r["css"] == k) for k in ("fake", "uncertain", "real")}
    return render_template("index.html", reports=reports, hero=hero, counts=counts)


def build_chart(report):
    """Coordinates for the SVG score timeline (viewBox 1000 x 230)."""
    frames = report["frames"]
    n = len(frames)
    left, right, top, height = 46.0, 14.0, 16.0, 160.0
    step = (1000 - left - right) / max(1, n - 1) if n > 1 else 0

    def x(i):
        return left + i * step if n > 1 else 500.0

    def y(v):
        return top + (1 - min(1.0, max(0.0, v))) * height

    raw = " ".join(f"{x(f['index']):.1f},{y(f['score']):.1f}" for f in frames)
    smooth_pts = [(x(f["index"]), y(f["smoothed"])) for f in frames]
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in smooth_pts)
    base = top + height
    area = (f"{smooth_pts[0][0]:.1f},{base:.1f} {line} {smooth_pts[-1][0]:.1f},{base:.1f}"
            if smooth_pts else "")
    half = step / 2 if n > 1 else 20
    segs = []
    for k, sg in enumerate(report["segments"], start=1):
        x0 = max(left - half, x(sg["start_idx"]) - half)
        x1 = min(1000 - right + half, x(sg["end_idx"]) + half)
        segs.append({"x": round(x0, 1), "w": round(x1 - x0, 1), "k": k})
    ticks = []
    every = max(1, n // 8)
    for f in frames:
        if f["index"] % every == 0:
            label = f"{f['start_s']:.1f}" if f["start_s"] is not None else str(f["index"] + 1)
            ticks.append({"x": round(x(f["index"]), 1), "label": label})
    return {
        "raw": raw, "line": line, "area": area, "segs": segs, "ticks": ticks,
        "thr_y": round(y(report["threshold"]), 1), "top": top, "base": base,
        "grid": [{"y": round(y(v), 1), "label": f"{int(v * 100)}%"} for v in (0, 0.5, 1)],
        "dots": [{"x": round(x(f["index"]), 1), "y": round(y(f["score"]), 1),
                  "hot": f["score"] >= report["threshold"]} for f in frames],
    }


@app.route("/report/<video_id>")
def report_page(video_id):
    report = load_report(video_id)
    s = summary(report)
    names = frame_names(video_id, len(report["frames"]))
    frames = []
    for f in report["frames"]:
        i = f["index"]
        frames.append({**f, "img": names[i] if i < len(names) else None,
                       "pct": round(f["score"] * 100)})
    seg_of = {}
    for k, sg in enumerate(report["segments"], start=1):
        for i in range(sg["start_idx"], sg["end_idx"] + 1):
            seg_of.setdefault(i, k)
    match = None
    if report.get("ground_truth_label") and report["verdict"] != "UNCERTAIN":
        match = report["ground_truth_label"] == report["verdict"]
    return render_template(
        "report.html", r=report, s=s, frames=frames, seg_of=seg_of, match=match,
        chart=build_chart(report),
        thr=round(report["threshold"] * 100),
        band=[round(x * 100) for x in report["uncertain_band"]],
    )


@app.route("/frames/<video_id>/<name>")
def frame_file(video_id, name):
    _check_id(video_id)
    if not SAFE_ID.match(name) or not name.startswith("frame_"):
        abort(404)
    return send_from_directory(app.config["FRAMES_DIR"] / video_id, name)


@app.route("/api/report/<video_id>")
def api_report(video_id):
    return jsonify(load_report(video_id))


@app.template_filter("seconds")
def seconds_filter(v):
    return "" if v is None else f"{v:.1f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--reports_dir", default=None)
    parser.add_argument("--frames_dir", default=None)
    args = parser.parse_args()
    if args.reports_dir:
        app.config["REPORTS_DIR"] = Path(args.reports_dir)
    if args.frames_dir:
        app.config["FRAMES_DIR"] = Path(args.frames_dir)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
