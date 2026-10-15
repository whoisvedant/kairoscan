"""
Tests for src/evaluation/evidence.py -- the Stage 5 logic.
No GPU, dataset or trained model needed.

Run from the project root:
    python tests/test_evidence.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

import numpy as np
from evidence import (
    build_report, find_segments, frame_time_ranges, smooth_scores,
    top_frames, verdict_from_probability, confidence_label,
)


def test_time_ranges_exact():
    # 10 s video at 25 fps = 250 frames, 24 sampled -> each sample covers 250/24/25 s
    r = frame_time_ranges(24, total_frames=250, fps=25, n_sampled=24)
    assert abs(r[0][0] - 0.0) < 1e-9
    assert abs(r[-1][1] - 10.0) < 1e-6          # last slice ends at the video end
    assert all(r[i][1] <= r[i + 1][0] + 1e-9 for i in range(23))   # no overlap


def test_time_ranges_unknown():
    assert frame_time_ranges(16, total_frames=None, fps=None) is None
    assert frame_time_ranges(16, total_frames=250, fps=0) is None


def test_smoothing_keeps_length_and_edges():
    s = smooth_scores([0, 0, 1, 0, 0], window=3)
    assert len(s) == 5
    assert abs(s[2] - 1 / 3) < 1e-9
    assert abs(s[0] - 0.0) < 1e-9               # edge: mean of [0, 0]
    flat = smooth_scores([0.8] * 6, window=3)
    assert np.allclose(flat, 0.8)               # edges must not be dragged down


def test_find_segments():
    sm = np.array([0.1, 0.7, 0.8, 0.2, 0.9, 0.9, 0.9])
    assert find_segments(sm, 0.5) == [(1, 2), (4, 6)]     # run touching the last frame
    assert find_segments(np.array([0.1, 0.2]), 0.5) == []
    assert find_segments(np.array([0.9, 0.9]), 0.5) == [(0, 1)]


def test_top_frames_skips_neighbours():
    sc = [0.1, 0.9, 0.95, 0.2, 0.1, 0.7, 0.1, 0.1]
    picked = top_frames(sc, k=3, min_gap=2)
    assert 2 in picked and 1 not in picked      # 1 is next to the stronger frame 2
    assert len(picked) == 3


def test_verdict_bands():
    assert verdict_from_probability(0.95) == "FAKE"
    assert verdict_from_probability(0.05) == "REAL"
    assert verdict_from_probability(0.5) == "UNCERTAIN"
    assert verdict_from_probability(0.6) == "FAKE"        # boundary belongs to FAKE
    assert verdict_from_probability(0.4) == "REAL"        # boundary belongs to REAL
    assert confidence_label(0.5, "UNCERTAIN") == "Low"
    assert confidence_label(0.97, "FAKE") == "High"
    assert confidence_label(0.80, "FAKE") == "Moderate"


def test_report_fake_with_burst():
    scores = [0.1, 0.1, 0.2, 0.9, 0.95, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ranges = frame_time_ranges(16, total_frames=250, fps=25, n_sampled=24)
    rep = build_report(scores, overall_prob=0.82, time_ranges=ranges, video_seconds=10.0)
    assert rep["verdict"] == "FAKE"
    assert len(rep["segments"]) == 1
    seg = rep["segments"][0]
    assert seg["start_idx"] <= 3 and seg["end_idx"] >= 5
    assert "concentrated" in rep["explanation"]
    assert " s" in rep["explanation"]                      # a time range was quoted
    assert any("first" in lim for lim in rep["limitations"])   # 16 of 24 frames -> coverage note


def test_report_real_and_uncertain():
    flat = [0.05] * 16
    rep = build_report(flat, overall_prob=0.04)
    assert rep["verdict"] == "REAL" and rep["segments"] == []
    assert "No window scored above" in rep["explanation"]
    assert not rep["has_timing"]
    assert any("sampled-frame number" in l for l in rep["limitations"])

    unc = build_report([0.5] * 16, overall_prob=0.5)
    assert unc["verdict"] == "UNCERTAIN"
    assert "inconclusive" in unc["explanation"]


def test_report_fake_everywhere():
    rep = build_report([0.9] * 16, overall_prob=0.9)
    assert rep["verdict"] == "FAKE"
    assert "whole" in rep["explanation"]                   # "almost the whole analysed clip"


def test_explanation_never_invents_numbers():
    rep = build_report([0.1, 0.9, 0.9, 0.1] * 4, overall_prob=0.55)
    # 55% is inside the 40-60% band -> must NOT be called FAKE or REAL
    assert rep["verdict"] == "UNCERTAIN"
    assert "55%" in rep["explanation"]


def test_json_serialisable():
    import json
    rep = build_report([0.2, 0.8, 0.9, 0.3], overall_prob=0.7, video_id="x", ground_truth="FAKE")
    json.dumps(rep)                                        # raises if numpy types leak through


def test_figure_renders():
    from evidence import render_timeline_figure
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(16):
            p = Path(d) / f"frame_{i:04d}.png"
            Image.fromarray(np.full((64, 64, 3), 10 * i, dtype=np.uint8)).save(p)
            paths.append(str(p))
        ranges = frame_time_ranges(16, total_frames=250, fps=25, n_sampled=24)
        rep = build_report([0.1] * 5 + [0.9] * 4 + [0.1] * 7, 0.8, time_ranges=ranges,
                           video_seconds=10.0, video_id="synthetic")
        out = Path(d) / "fig.png"
        render_timeline_figure(rep, paths, str(out))
        assert out.exists() and out.stat().st_size > 5000
        # also the no-timing path (small Kaggle sample has no fps)
        rep2 = build_report([0.1] * 16, 0.05, video_id="no_timing")
        render_timeline_figure(rep2, paths, str(Path(d) / "fig2.png"))
        assert (Path(d) / "fig2.png").exists()



def test_sections_are_built_from_numbers():
    rep = build_report([0.9] * 8 + [0.1] * 8, 0.8, frame_time_ranges(16, 300, 30.0, 24), 10.0)
    titles = [x["title"] for x in rep["sections"]]
    assert titles == ["What the model found", "How consistent the signal was",
                      "How much to trust it", "Suggested next step"], titles
    assert "8 of 16" in rep["sections"][1]["text"]
    assert "90%" in rep["sections"][1]["text"] and "10%" in rep["sections"][1]["text"]


def test_sections_include_face_region_when_given():
    info = [{"index": 2, "label": "the lower face (mouth and jaw)", "focused": True, "shares": {"upper": 0.1, "middle": 0.2, "lower": 0.7}},
            {"index": 5, "label": "the lower face (mouth and jaw)", "focused": True, "shares": {"upper": 0.1, "middle": 0.2, "lower": 0.7}}]
    rep = build_report([0.9] * 16, 0.95, None, None, region_info=info)
    where = [x for x in rep["sections"] if x["title"].startswith("Where")]
    assert len(where) == 1 and "lower face" in where[0]["text"]
    assert "not proof" in where[0]["text"]
    spread = [{"index": 2, "label": "no single region", "focused": False, "shares": {"upper": 0.34, "middle": 0.33, "lower": 0.33}}]
    rep2 = build_report([0.9] * 16, 0.95, None, None, region_info=spread)
    assert "spread across the face" in [x for x in rep2["sections"] if x["title"].startswith("Where")][0]["text"]


def test_sections_uncertain_does_not_pick_a_side():
    rep = build_report([0.5] * 16, 0.5, None, None)
    text = " ".join(x["text"] for x in rep["sections"])
    assert "does not decide" in text and "higher-quality copy" in text


def test_no_region_claims_without_a_heatmap():
    info = [{"index": 1, "label": "no clear region", "shares": None, "focused": False}]
    rep = build_report([0.05] * 16, 0.04, None, None, region_info=info)
    assert not [x for x in rep["sections"] if x["title"].startswith("Where")]

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
