"""
Multi-view SOME/IP intrusion detection: model definition.

Implements the proposed lightweight multi-view fusion detector:
  - View A head: stream-wise behavioral features (linear/MLP)
  - View B head: compact payload/header sequence encoder (GRU or self-attention)
  - Fusion: attention-weighted late fusion
  - Auxiliary anomaly score: lightweight autoencoder on benign traffic

Target: PyTorch >= 1.10, CPU or CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# View B encoders (payload/header token sequences)
# ---------------------------------------------------------------------------
class GRUEncoder(nn.Module):
    """Compact single-layer GRU encoder for token sequences."""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.pool = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L)
        e = self.emb(x)                      # (B, L, emb)
        out, _ = self.gru(e)                 # (B, L, hidden)
        # mean pooling over the sequence -> (B, hidden)
        pooled = out.mean(dim=1)
        return torch.tanh(self.pool(self.dropout(pooled)))


class AttentionEncoder(nn.Module):
    """Single-layer self-attention encoder (lightweight)."""

    def __init__(self, vocab_size, emb_dim=128, heads=2, hidden=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.reproj = nn.Linear(emb_dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        e = self.emb(x)
        attn_out, _ = self.attn(e, e, e)
        out = self.norm(e + attn_out)
        pooled = out.mean(dim=1)
        return torch.tanh(self.reproj(self.dropout(pooled)))


class CNNEncoder(nn.Module):
    """1D-CNN encoder; good at localized byte changes (e.g., tamper/replay)."""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        e = self.emb(x)                    # (B, L, emb)
        e = e.transpose(1, 2)              # (B, emb, L)
        c = self.conv(e)                   # (B, hidden, L)
        pooled = c.max(dim=2).values       # (B, hidden)
        return torch.tanh(self.proj(self.dropout(pooled)))


# ---------------------------------------------------------------------------
# View A head (behavioral features -> scalar/logit)
# ---------------------------------------------------------------------------
class BehavioralHead(nn.Module):
    def __init__(self, d_a, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_a, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, f_a):
        # f_a: (B, d_a) -> (B, 1)
        return self.net(f_a)


# ---------------------------------------------------------------------------
# Multi-view fusion detector
# ---------------------------------------------------------------------------
class MultiViewFusion(nn.Module):
    """
    Combines behavioral + payload-semantic views.

    Args:
        d_a: behavioral feature dimension (View A).
        vocab_size: token vocab size for View B encoder.
        encoder: 'gru' or 'attn'.
        num_classes: 1 for binary (sigmoid), >1 for multiclass.
    """

    def __init__(self, d_a, vocab_size, encoder='gru', num_classes=1,
                 emb_dim=128, hidden=128, heads=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.view_a = BehavioralHead(d_a, hidden=64)

        if encoder == 'gru':
            self.view_b = GRUEncoder(vocab_size, emb_dim, hidden, dropout=dropout)
        elif encoder == 'attn':
            self.view_b = AttentionEncoder(vocab_size, emb_dim, heads, hidden, dropout=dropout)
        elif encoder == 'cnn':
            self.view_b = CNNEncoder(vocab_size, emb_dim, hidden, dropout=dropout)
        else:
            raise ValueError(f"unknown encoder {encoder}")

        # Attention weights over the two view scores (lambda_A, lambda_B)
        self.lambda_weights = nn.Parameter(torch.zeros(2))

        # Fusion classifier head
        # Use the RAW View A features (d_a) concatenated with View B pooled vector.
        in_dim = d_a + hidden  # raw View A features + View B pooled
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, f_a, seq_b):
        s_a = self.view_a(f_a)                     # (B, 1)
        v_b = self.view_b(seq_b)                   # (B, hidden)
        lam = torch.softmax(self.lambda_weights, dim=0)
        # Concatenate the raw View A features with the View B vector.
        feats = torch.cat([f_a, v_b], dim=1)       # (B, d_a + hidden)
        logits = self.classifier(feats)            # (B, num_classes)
        return logits, lam


class AnomalyScorer(nn.Module):
    """Lightweight autoencoder to score anomaly distance from benign traffic."""

    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(),
                                 nn.Linear(d // 2, d // 4))
        self.dec = nn.Sequential(nn.Linear(d // 4, d // 2), nn.ReLU(),
                                 nn.Linear(d // 2, d))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)  # reconstruction


def reconstruction_anomaly_score(model, x):
    """L2 reconstruction error used as auxiliary anomaly score."""
    with torch.no_grad():
        pred = model(x)
    return torch.norm(x - pred, dim=-1, keepdim=True)  # (B, 1)
