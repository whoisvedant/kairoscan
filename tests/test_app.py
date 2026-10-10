"""
test_app.py
-----------
Checks the web interface with tiny fake reports (no model, no dataset needed).

Run from the project root:
    python tests/test_app.py
"""

import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from PIL import Image
from evidence import build_report, frame_time_ranges
import app as webapp


def make_report(tmp, vid, scores, prob, truth):
    frames = Path(tmp) / "processed" / vid
    frames.mkdir(parents=True)
    for i in range(24):
        Image.new("RGB", (32, 32), (i * 8, 80, 80)).save(frames / f"frame_{i:03d}.png")
    rep = build_report(scores, prob, frame_time_ranges(16, 300, 30.0, 24), 10.0,
                       video_id=vid, ground_truth=truth)
    (Path(tmp) / "reports").mkdir(exist_ok=True)
    json.dump(rep, open(Path(tmp) / "reports" / f"{vid}.json", "w"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        make_report(tmp, "fake_a", [0.95] * 16, 0.97, "FAKE")
        make_report(tmp, "real_a", [0.05] * 16, 0.04, "REAL")
        make_report(tmp, "mid_a", [0.5] * 16, 0.5, "FAKE")
        webapp.app.config["REPORTS_DIR"] = Path(tmp) / "reports"
        webapp.app.config["FRAMES_DIR"] = Path(tmp) / "processed"
        c = webapp.app.test_client()

        page = c.get("/")
        assert page.status_code == 200
        html = page.get_data(as_text=True)
        assert "fake_a" in html and "real_a" in html and "Cannot decide" in html
        print("PASS library page lists all reports")

        for vid, text in (("fake_a", "Likely manipulated"), ("real_a", "Leans real"), ("mid_a", "Cannot decide")):
            r = c.get(f"/report/{vid}")
            assert r.status_code == 200 and text in r.get_data(as_text=True), vid
        print("PASS report page shows the right verdict wording")

        assert c.get("/frames/fake_a/frame_000.png").status_code == 200
        assert c.get("/api/report/fake_a").get_json()["verdict"] == "FAKE"
        print("PASS frame images and JSON are served")

        for bad in ("/report/nope", "/frames/fake_a/secret.txt", "/frames/fake_a/frame_999.png",
                    "/frames/..%2F/frame_000.png", "/report/.hidden"):
            # 404 = refused; 308 = Flask redirecting to a cleaned-up path, which is then refused too
            assert c.get(bad).status_code in (404, 308), bad
        print("PASS missing or unsafe paths are refused")

        webapp.app.config["REPORTS_DIR"] = Path(tmp) / "empty"
        assert "No reports yet" in c.get("/").get_data(as_text=True)
        print("PASS empty library shows how to create reports")
    print("All app tests passed.")


if __name__ == "__main__":
    main()
