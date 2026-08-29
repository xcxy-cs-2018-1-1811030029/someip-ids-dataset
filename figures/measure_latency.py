"""Measure per-packet latency of behavioral feature extraction and semantic decoding (pure numpy/python)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from someip_ids.semantics import decode_payload, learn_semantic_stats, semantic_score

rng = np.random.default_rng(0)
N = 20000
# representative 12-byte payloads (3 x float32)
payloads = [bytes(rng.integers(0, 256, 12, dtype=np.uint8)) for _ in range(N)]

def behavioral(p):
    a = np.frombuffer(p, dtype=np.uint8)
    cnt = np.bincount(a, minlength=256).astype(float) / len(a); cnt = cnt[cnt > 0]
    return float(-(cnt * np.log(cnt)).sum()), float(a.mean()), float(a.std())

# semantic stats from a small normal set (so semantic_score is realistic)
normal_vals = [decode_payload(p) for p in payloads[:500]]
stats = learn_semantic_stats([v for v in normal_vals if v])

# --- timing ---
def timeit(fn, reps=5):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for p in payloads:
            fn(p)
        dt = (time.perf_counter_ns() - t0) / N
        best = min(best, dt)
    return best / 1e3  # microseconds

t_beh = timeit(behavioral)
t_sem = timeit(lambda p: semantic_score(decode_payload(p), *stats))
def fused(p):
    behavioral(p)
    semantic_score(decode_payload(p), *stats)
t_fus = timeit(fused)

print(f"behavioral feature: {t_beh:.1f} us/packet  ({1e6/t_beh:,.0f} pkt/s)")
print(f"semantic decode+score: {t_sem:.1f} us/packet  ({1e6/t_sem:,.0f} pkt/s)")
print(f"fused (beh+sem): {t_fus:.1f} us/packet  ({1e6/t_fus:,.0f} pkt/s)")
print(f"deep (from benchmark, GPU-batched): 3.3 us/packet  (~303k pkt/s)")
