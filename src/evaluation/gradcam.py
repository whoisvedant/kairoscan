"""
gradcam.py
----------
"Where on the face did the model look?"

Grad-CAM takes the last convolution layer of the ResNet-18 (a 7 x 7 grid of
feature maps per frame) and asks: which grid cells pushed this frame's
fake-vs-real score upwards? The answer is a heatmap over the face crop.

What it does and does not show (the report says this too):
    + It shows where the model's attention was strongest for a frame.
    - It does not prove that region is manipulated, and the face crops are not
      aligned, so region names ("lower face") are approximate.

The score used is the frame's own window logit: fake logit minus real logit.
With the LSTM and Transformer, that score also depends on earlier frames; the
heatmap shows only the part that comes from this frame's own picture.
"""

from typing import Dict, List

import numpy as np
import torch

# Position of the last ResNet stage inside model.encoder.backbone
# (backbone = Sequential(conv1, bn1, relu, maxpool, layer1..layer4, avgpool)).
LAST_CONV_STAGE = 7


def compute_cams(model, clip, frame_indices: List[int], device) -> Dict[int, np.ndarray]:
    """clip: tensor (1, T, 3, H, W). Returns {frame_index: heatmap (H, W) in 0..1}.
    A heatmap is all zeros when no region pushed the score towards fake."""
    model.eval()
    layer = model.encoder.backbone[LAST_CONV_STAGE]
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("acts", o))
    height, width = clip.shape[-2], clip.shape[-1]
    cams: Dict[int, np.ndarray] = {}
    try:
        # cuDNN's LSTM refuses to run backward in eval mode; turning cuDNN off
        # for this call avoids that and gives the same numbers.
        with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
            _, window_logits = model(clip.to(device))
            acts = store["acts"]                      # (T, 512, 7, 7)
            for t in frame_indices:
                score = window_logits[0, t, 1] - window_logits[0, t, 0]
                grad = torch.autograd.grad(score, acts, retain_graph=True)[0][t]
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((weights * acts[t]).sum(dim=0)).detach()
                if float(cam.max()) <= 0:
                    cams[t] = np.zeros((height, width), dtype=np.float32)
                    continue
                cam = cam / cam.max()
                cam = torch.nn.functional.interpolate(
                    cam[None, None], size=(height, width), mode="bilinear",
                    align_corners=False)[0, 0]
                cams[t] = cam.cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
    return cams


def region_summary(cam: np.ndarray) -> Dict:
    """Which horizontal band of the face crop holds most of the attention.
    Bands: upper (forehead and eyes), middle (nose and cheeks), lower (mouth and jaw)."""
    total = float(cam.sum())
    if total <= 0:
        return {"label": "no clear region", "shares": None, "focused": False}
    h = cam.shape[0]
    bands = [cam[: h // 3].sum(), cam[h // 3: 2 * h // 3].sum(), cam[2 * h // 3:].sum()]
    shares = [float(b) / total for b in bands]
    names = ["the upper face (forehead and eyes)",
             "the middle of the face (nose and cheeks)",
             "the lower face (mouth and jaw)"]
    best = int(np.argmax(shares))
    focused = shares[best] >= 0.45
    return {
        "label": names[best] if focused else "no single region (attention was spread across the face)",
        "shares": {"upper": round(shares[0], 3), "middle": round(shares[1], 3),
                   "lower": round(shares[2], 3)},
        "focused": focused,
    }


def save_overlay(image_path: str, cam: np.ndarray, out_path: str, alpha: float = 0.45):
    """Face crop with the heatmap blended on top (red = strongest attention)."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base = np.asarray(img, dtype=np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    mask = (cam[..., None] * alpha * 1.6).clip(0, alpha)   # transparent where attention is low
    out = base * (1 - mask) + heat * mask
    Image.fromarray((out * 255).astype(np.uint8)).save(out_path)
