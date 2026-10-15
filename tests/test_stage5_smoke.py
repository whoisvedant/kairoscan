"""
test_stage5_smoke.py
--------------------
Checks that analyze_video.py and splice_test.py RUN end to end.

It uses tiny fake data and a randomly initialised model, so it says nothing about
accuracy. It only proves the plumbing works (files are found, shapes line up,
reports and figures are written). It needs no dataset, no internet, no GPU.

Run from the project root:
    python tests/test_stage5_smoke.py
"""

import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image

# Avoid downloading ImageNet weights: build the same ResNet-18 with random weights.
_orig_resnet18 = tvm.resnet18
tvm.resnet18 = lambda *a, **k: _orig_resnet18(weights=None)

import models.temporal_model as tm  # noqa: E402
tm.resnet18 = tvm.resnet18

from analyze_video import analyze_video, load_model, load_temperature  # noqa: E402
from splice_test import run_splice_test  # noqa: E402
from evidence import render_timeline_figure  # noqa: E402
from gradcam import compute_cams, region_summary  # noqa: E402


def make_fake_dataset(root):
    rng = np.random.default_rng(0)
    ids = {"real": [], "fake": []}
    for i in range(8):
        label = i % 2
        vid = f"{'fake' if label else 'real'}_{i:03d}"
        d = Path(root) / vid
        d.mkdir(parents=True)
        for f in range(24):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"frame_{f:03d}.png")
        meta = {"label": label, "frames_saved": 24,
                "fps": 30.0 if i != 3 else None,
                "total_frames": 300 if i != 3 else None}
        json.dump(meta, open(d / "meta.json", "w"))
        ids["fake" if label else "real"].append(vid)
    return ids


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proc = tmp / "processed"
        ids = make_fake_dataset(proc)

        model = tm.build_model("lstm")
        ckpt = tmp / "lstm_best.pt"
        torch.save(model.state_dict(), ckpt)
        json.dump({"temperature": 1.3}, open(tmp / "lstm_temperature.json", "w"))

        loaded = load_model("lstm", str(ckpt), "cpu")
        T, how = load_temperature("lstm", True, ckpt_dir=str(tmp))
        assert abs(T - 1.3) < 1e-9, T
        T_raw, _ = load_temperature("lstm", False, ckpt_dir=str(tmp))
        assert T_raw == 1.0

        # one video with fps/total_frames, one without
        for vid in (ids["real"][0], ids["fake"][1]):
            cam_dir = tmp / "cams"
            cam_dir.mkdir(exist_ok=True)
            report, paths = analyze_video(vid, loaded, "cpu", T, str(proc), out_dir=str(cam_dir))
            assert len(report["sections"]) >= 4
            for h in report["heatmaps"]:
                assert (cam_dir / h["file"]).exists()
            assert report["verdict"] in ("REAL", "FAKE", "UNCERTAIN")
            assert len(report["frames"]) == 16
            json.dumps(report)
            out = tmp / f"{vid}.png"
            render_timeline_figure(report, paths, str(out))
            assert out.exists() and out.stat().st_size > 1000
        print("analyze_video: OK")

        # Grad-CAM must work for every model type, including the Transformer
        clip = torch.rand(1, 16, 3, 224, 224)
        for kind in ("lstm", "gru", "transformer"):
            m = tm.build_model(kind).eval()
            cams = compute_cams(m, clip, [3, 9], "cpu")
            assert set(cams) == {3, 9}
            for c in cams.values():
                assert c.shape == (224, 224) and 0.0 <= c.min() and c.max() <= 1.0
                assert region_summary(c)["label"]
        print("gradcam: OK (lstm, gru, transformer)")

        res = run_splice_test(loaded, "cpu", T, ids["real"], ids["fake"], str(proc),
                              n_pairs=4, show_progress=False)
        for key in ("real_to_fake", "fake_to_real", "unspliced_reference"):
            assert key in res
        json.dumps(res)
        assert 0 <= res["real_to_fake"]["boundary_hit_rate"] <= 1
        print("splice_test: OK")
    print("Smoke test passed (random weights: this checks plumbing, not accuracy).")


if __name__ == "__main__":
    main()
