"""
temporal_model.py
------------------
Stage 2: the two temporal model variants compared against the CNN-only
baseline (train_baseline.py).

Both models share the same CNN feature extractor (ResNet-18, same backbone
as the baseline) so the comparison isolates the effect of the temporal
modeling choice -- not a difference in the spatial feature extractor.

Both models return TWO things per video clip:
    overall_logits  (B, 2)    -- the single real/fake decision for the video
    window_logits   (B, T, 2) -- a per-frame/per-window score

window_logits is not used for training here (only overall_logits feeds the
loss), but it is exactly the per-window evidence the project's Stage 5
(evidence localization) will read from later, so the architecture is already
shaped the way the report's diagram describes -- no redesign needed when we
get to that stage.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

FEATURE_DIM = 512  # ResNet-18's output feature size before its original fc layer


class CNNEncoder(nn.Module):
    """Same backbone as the CNN-only baseline, used here as a per-frame
    feature extractor instead of a direct classifier."""

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Keep every layer except the final classification layer (fc),
        # so this outputs a 512-dim feature vector per image instead of
        # a 2-class prediction.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # x: (N, 3, H, W) -> (N, 512, 1, 1) -> (N, 512)
        feat = self.backbone(x)
        return feat.flatten(1)


class TemporalLSTM(nn.Module):
    """CNN + LSTM/GRU temporal model (Stage 2, arm 2 of the ablation)."""

    def __init__(self, hidden_size=256, num_layers=1, rnn_type="lstm", bidirectional=False):
        super().__init__()
        self.encoder = CNNEncoder()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.window_head = nn.Linear(out_dim, 2)

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, FEATURE_DIM)

        rnn_out, _ = self.rnn(feats)              # (B, T, hidden)
        window_logits = self.window_head(rnn_out)  # (B, T, 2) -- per-frame evidence
        overall_logits = window_logits.mean(dim=1)  # (B, 2) -- pooled video-level decision
        return overall_logits, window_logits


class TemporalTransformer(nn.Module):
    """CNN + Transformer encoder temporal model (Stage 2, arm 3 of the ablation)."""

    def __init__(self, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1, max_len=32):
        super().__init__()
        self.encoder = CNNEncoder()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, FEATURE_DIM))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_DIM,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.window_head = nn.Linear(FEATURE_DIM, 2)

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, FEATURE_DIM)
        feats = feats + self.pos_embedding[:, :T, :]

        trans_out = self.transformer(feats)         # (B, T, 512)
        window_logits = self.window_head(trans_out)  # (B, T, 2)
        overall_logits = window_logits.mean(dim=1)   # (B, 2)
        return overall_logits, window_logits


def build_model(model_type: str):
    if model_type == "lstm":
        return TemporalLSTM(rnn_type="lstm")
    elif model_type == "gru":
        return TemporalLSTM(rnn_type="gru")
    elif model_type == "transformer":
        return TemporalTransformer()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm', 'gru', or 'transformer'.")
