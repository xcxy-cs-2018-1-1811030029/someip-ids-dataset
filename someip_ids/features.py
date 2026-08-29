"""
View A: stream-wise behavioral feature extraction for SOME/IP.

Features (computed per flow, i.e., within same src/dst IP+port pair):
  1. time-interval variation        Δt_i = t_i - t_{i-1}
  2. payload log-likelihood          from position-wise byte model (Laplace smoothing)
  3. payload cross-entropy
  4. payload Hamming-distance change between consecutive payloads
  5. length change between consecutive IP / UDP / SOME/IP packets

These follow the behavioral design in the paper and are cheap and interpretable.
"""

import math
import numpy as np


def _position_wise_byte_distribution(normal_payloads, L, alpha=1.0, vocab=256):
    """
    Estimate P_i(b): probability of byte b at position i, from benign payloads.
    All payloads are assumed padded/truncated to length L.
    Returns: P (L, vocab) with probabilities (Laplace-smoothed).
    """
    counts = np.full((L, vocab), 0.0)
    for p in normal_payloads:
        for i, b in enumerate(p):
            if i < L:
                counts[i, int(b)] += 1.0
    counts += alpha
    denom = counts.sum(axis=1, keepdims=True)
    return counts / denom


def payload_log_likelihood(payload, P):
    """Sum of log P_i(x_i) for payload (float/list of bytes)."""
    ll = 0.0
    for i, b in enumerate(payload):
        if i < P.shape[0]:
            p = P[i, int(b)]
            ll += math.log(p if p > 1e-12 else 1e-12)
    return ll


def payload_cross_entropy(payload, P):
    """-1/L * sum log P_i(x_i)."""
    if len(payload) == 0:
        return 0.0
    return -payload_log_likelihood(payload[:P.shape[0]], P) / min(len(payload), P.shape[0])


def hamming_change(p1, p2):
    """Hamming distance between two equal-length byte sequences (XOR + popcount)."""
    n = min(len(p1), len(p2))
    if n == 0:
        return 0.0
    d = 0
    for i in range(n):
        d += bin(int(p1[i]) ^ int(p2[i])).count('1')
    return d


def compute_view_a_features(flow, P, L):
    """
    flow: list of dicts, each with keys:
        'ts' (float), 'payload' (bytes/list), 'length' (int)
    Returns np.ndarray of per-flow aggregated behavioral features (fixed dim).
    We output a few aggregate statistics over the flow to yield a fixed-length vector.
    """
    if len(flow) < 2:
        return None

    intervals = [flow[i]['ts'] - flow[i - 1]['ts'] for i in range(1, len(flow))]
    length_deltas = [abs(flow[i]['length'] - flow[i - 1]['length']) for i in range(1, len(flow))]
    ll = [payload_log_likelihood(f['payload'], P) for f in flow]
    ce = [payload_cross_entropy(f['payload'], P) for f in flow]
    hamming = [hamming_change(flow[i - 1]['payload'], flow[i]['payload']) for i in range(1, len(flow))]

    def stats(arr):
        if len(arr) == 0:
            return [0.0, 0.0, 0.0]
        return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))]

    feats = []
    feats += stats(intervals)       # time-interval variation
    feats += stats(ll)              # payload log-likelihood
    feats += stats(ce)              # payload cross-entropy
    feats += stats(hamming)         # payload Hamming change
    feats += stats(length_deltas)   # length change
    return np.array(feats, dtype=np.float32)  # 15-dim (5 groups x 3 stats)


FEATURE_DIM_A = 15  # 5 feature groups * (mean, std, max)


def train_position_model(all_normal_payloads, L=128, alpha=1.0):
    """Train the position-wise byte model from a list of benign payloads."""
    return _position_wise_byte_distribution(all_normal_payloads, L, alpha)
