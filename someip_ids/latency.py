"""
latency.py -- fair per-packet latency comparison, ONE environment, CPU batch-1.

Replaces the paper's CPU-python vs GPU-batched comparison. All detectors are
measured on the same machine, same CPU, single-threaded, batch size 1:
  - behavioral feature extraction (pure python/numpy, as in benchmark.build)
  - GBM inference (300 trees) including feature extraction
  - semantic decode + score
  - GRU batch-1 forward
  - TCN batch-1 forward
Reports per-packet median / p95 in us and throughput in pkt/s.

Usage:  python3 someip_ids/latency.py --gen vs_ip_gen/generate/
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats, semantic_score

SEED = 1234


def per_packet_behavioral(r, prev, state):
    """Single-message behavioral feature vector (identical to benchmark.build)."""
    ev = r["event"]
    b = r["payload"]
    a = np.frombuffer(b, dtype=np.uint8) if len(b) else np.zeros(0, np.uint8)
    p = prev.get(ev)
    dt = (r["ts"] - p["ts"]) if p else 0.0
    dl = (r["plen"] - p["plen"]) if p else 0.0
    st = state.get(ev, 0.0)
    st = st + 1.0 if dt < 0.5 else 0.0
    state[ev] = st
    prev[ev] = r
    cnt = np.bincount(a, minlength=256).astype(float) / max(len(a), 1)
    cnt = cnt[cnt > 0]
    ent = float(-(cnt * np.log(cnt)).sum()) if len(cnt) else 0.0
    return np.array([ent,
                     float(a.mean()) if len(a) else 0.0,
                     float(a.std()) if len(a) else 0.0,
                     dt, dl, st,
                     float(ev % 1000) / 1000.0], dtype=np.float32)


def time_packets(fn, n, warmup=200, reps=5):
    """Best-of-reps mean per-packet time in microseconds."""
    best = np.inf
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter_ns() - t0) / n)
    return best / 1e3


def time_packets_dist(fn, n, warmup=50):
    """Median/p95 per-packet time in microseconds (single pass, warmed up)."""
    for _ in range(warmup):
        fn()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = (time.perf_counter_ns() - t0) / 1e3
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    rows, labels = [], []
    for p in csvs:
        sc, _, _ = B.parse_name(p)
        r = B.read_rows(p)
        rows += r
        labels += [0 if sc == "normal" else 1] * len(r)
    ts_all = np.array([x["ts"] for x in rows])
    normal_vals = [decode_payload(x["payload"]) for x, l in zip(rows, labels) if l == 0]
    stats = learn_semantic_stats([v for v in normal_vals if v])

    feats, sem, toks, ok = B.build(rows, stats)
    labels = np.array(labels)
    mask = ok > 0
    feats, sem, toks, labels, ts_all = feats[mask], sem[mask], toks[mask], labels[mask], ts_all[mask]

    # train GBM once (as in benchmark) so GBM timing is a real 300-tree model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    tr_i, _ = train_test_split(np.arange(len(labels)), test_size=0.3,
                               random_state=SEED, stratify=labels)
    print("training GBM (300 trees) for the timing harness ...")
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
    gbm.fit(feats[tr_i], labels[tr_i])

    import torch
    torch.set_num_threads(1)
    from someip_ids.benchmark import GRUCls, TCN
    gru = GRUCls().eval()
    tcn = TCN().eval()
    tok = toks[0:1].astype(np.int64)

    n = 2000
    print(f"\n===== per-packet latency (CPU, single thread, batch-1, n={n}) =====")

    # behavioral: per-packet incremental feature extraction on a rotating window
    prev_s = {}
    state_s = {}
    win = rows[10000:11000]

    def beh_fn():
        nonlocal prev_s, state_s
        per_packet_behavioral(win[len(prev_s) % len(win)], prev_s, state_s)

    med, p95 = time_packets_dist(beh_fn, n)
    print(f"behavioral features:  med={med:7.1f} us  p95={p95:7.1f} us  ({1e6/med:,.0f} pkt/s)")

    def gbm_fn():
        prev_s.clear(); state_s.clear()
        r = win[0]
        f = per_packet_behavioral(r, prev_s, state_s)
        gbm.predict_proba(f.reshape(1, -1))

    med, p95 = time_packets_dist(gbm_fn, n)
    print(f"GBM (feat+300 trees): med={med:7.1f} us  p95={p95:7.1f} us  ({1e6/med:,.0f} pkt/s)")

    p0 = win[0]["payload"]

    def sem_fn():
        semantic_score(decode_payload(p0), *stats)

    med, p95 = time_packets_dist(sem_fn, n)
    print(f"semantic decode+sc : med={med:7.1f} us  p95={p95:7.1f} us  ({1e6/med:,.0f} pkt/s)")

    def gru_fn():
        with torch.inference_mode():
            gru(torch.from_numpy(tok))

    med, p95 = time_packets_dist(gru_fn, n)
    print(f"GRU batch-1 CPU     : med={med:7.1f} us  p95={p95:7.1f} us  ({1e6/med:,.0f} pkt/s)")

    def tcn_fn():
        with torch.inference_mode():
            tcn(torch.from_numpy(tok))

    med, p95 = time_packets_dist(tcn_fn, n)
    print(f"TCN batch-1 CPU     : med={med:7.1f} us  p95={p95:7.1f} us  ({1e6/med:,.0f} pkt/s)")


if __name__ == "__main__":
    main()
