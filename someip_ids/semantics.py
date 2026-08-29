"""
semantics.py

Payload semantic-validity modeling for SOME/IP messages.

Assumes a known signal schema: the payload is a fixed layout of typed little-endian
signals (e.g., 3 float32: speed, acceleration, yaw-rate). From BENIGN traffic we learn
the normal value statistics per signal (mean, std, min, max). The semantic score of a
message is the max standardized deviation (z-score) and range violation over its
decoded values. This catches semantics-preserving tampering (an implausible value)
that behavior/aggregate-byte statistics and raw-byte models cannot.
"""

import struct

import numpy as np

# Signal schema: (name, byte_offset, format_char). float32 -> 'f'.
SCHEMA = [("speed", 0, "f"),
          ("accel", 4, "f"),
          ("yaw", 8, "f")]
N_SIGNALS = len(SCHEMA)
PAYLOAD_LEN = 12  # 3 * 4 bytes


def decode_payload(payload_bytes):
    """Decode a SOME/IP payload (bytes) into a tuple of signal values."""
    if len(payload_bytes) < PAYLOAD_LEN:
        return None
    try:
        fmt = "<" + "".join(s[2] for s in SCHEMA)
        return struct.unpack(fmt, payload_bytes[:PAYLOAD_LEN])
    except (struct.error, TypeError):
        return None


def learn_semantic_stats(normal_values):
    """From a list of decoded value tuples (benign), return (mean, std, min, max)."""
    arr = np.array(normal_values, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return mean, std, lo, hi


def semantic_score(values, mean, std, lo, hi):
    """Max z-deviation plus range-violation margin. Higher = more implausible."""
    if values is None:
        return 0.0
    vals = np.array(values, dtype=np.float64)
    z = np.abs((vals - mean) / std)          # standardized deviation per signal
    # prefer the max z; but also penalize hard range violations
    violation = 0.0
    for i in range(len(vals)):
        if vals[i] < lo[i] or vals[i] > hi[i]:
            violation = max(violation, z[i])
    return float(max(np.max(z), violation))
