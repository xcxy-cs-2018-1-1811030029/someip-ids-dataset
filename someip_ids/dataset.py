"""
Dataset builders for the multi-view SOME/IP detector.

Provides:
  - tokenization of SOME/IP records (payload bytes + header semantic tokens) for View B
  - a torch Dataset combining View A feature vectors and View B token sequences
  - helpers to build feature/token arrays from parsed records (pcap or pickle source)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FEATURE_DIM_A, compute_view_a_features


# ---------------------------------------------------------------------------
# View B tokenization
# ---------------------------------------------------------------------------
PAD = 0
# Vocabulary: bytes 0..255 directly, plus a few special tokens for header fields.
TOKEN_OFFSET = 0
MAX_LEN = 128


def payload_to_tokens(payload, L=MAX_LEN):
    """Map payload bytes to token ids (byte value + 1 so PAD=0 is reserved)."""
    if payload is None:
        payload = []
    toks = [int(b) + 1 for b in payload[:L]]
    if len(toks) < L:
        toks += [PAD] * (L - len(toks))
    return np.array(toks, dtype=np.int64)


# ---------------------------------------------------------------------------
# Build views from parsed records
# ---------------------------------------------------------------------------
def build_view_a_from_records(records, P, L=128):
    """
    records: list of flow dicts -> each flow has list of packet dicts.
    Returns: (N, FEATURE_DIM_A) array of View A vectors, one per flow.
    """
    vecs = []
    for flow in records:
        v = compute_view_a_features(flow, P, L)
        if v is not None:
            vecs.append(v)
    if len(vecs) == 0:
        return np.zeros((0, FEATURE_DIM_A), dtype=np.float32)
    return np.stack(vecs, axis=0)


def build_view_b_from_records(records, L=MAX_LEN):
    """
    records: list of flow dicts. For View B we take the payload of a representative
    packet per flow (e.g., the first non-empty payload) as the token sequence.
    """
    seqs = []
    for flow in records:
        payload = None
        for pkt in flow:
            if pkt.get('payload'):
                payload = pkt['payload']
                break
        seqs.append(payload_to_tokens(payload, L))
    if len(seqs) == 0:
        return np.zeros((0, L), dtype=np.int64)
    return np.stack(seqs, axis=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SOMEIPMultiViewDataset(Dataset):
    def __init__(self, f_a, seq_b, labels, fa_mean=None, fa_std=None):
        self.f_a = torch.tensor(np.asarray(f_a, dtype=np.float32))
        self.seq_b = torch.tensor(np.asarray(seq_b, dtype=np.int64))
        self.labels = torch.tensor(np.asarray(labels, dtype=np.float32))
        self.fa_mean = torch.tensor(np.asarray(fa_mean, dtype=np.float32)) if fa_mean is not None else None
        self.fa_std = torch.tensor(np.asarray(fa_std, dtype=np.float32)) if fa_std is not None else None
        self.fa_std = torch.clamp(self.fa_std, min=1e-6) if self.fa_std is not None else None

    def __len__(self):
        return self.f_a.shape[0]

    def __getitem__(self, idx):
        x = self.f_a[idx]
        if self.fa_mean is not None:
            x = (x - self.fa_mean) / self.fa_std
        return x, self.seq_b[idx], self.labels[idx]


def make_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
