"""
pivot_experiment.py

Provable Pivot A experiment: behavior-only vs semantic-validity vs fused detector.

Reads the SOME/IP CSV logs (ts,event,session,payload_len,payload_hex) produced by the
generator. For each message:
  - aggregate behavioral features (byte entropy, mean/std, time-interval, length change)
  - a semantic-validity score (decode signals, compare to normal value stats learned
    from benign traffic)

Detectors:
  B1  = behavioral-only (gradient boosting on behavioral features)   [prior-work style]
  B2  = semantic-validity      (threshold on semantic score)        [ours]
  FUS = B1 OR B2 (fused decision)                                     [ours]

We report overall metrics and per-scenario recall, focusing on the hard
semantics-preserving tampering scenario.

Usage:
  python3 someip_ids/pivot_experiment.py --gen vs_ip_gen/generate/ --len-b 128
"""

import argparse
import glob
import os
import sys
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids.semantics import decode_payload, learn_semantic_stats, semantic_score

SEED = 1234
np.random.seed(SEED)


def parse_hex(hs):
    hs = hs.strip()
    return bytes(int(x, 16) for x in hs.split()) if hs else b""


def byte_entropy(b):
    if not b:
        return 0.0
    a = np.frombuffer(b, dtype=np.uint8)
    cnt = np.bincount(a, minlength=256).astype(float) / len(a)
    cnt = cnt[cnt > 0]
    return float(-(cnt * np.log(cnt)).sum())


def read_rows(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.rstrip("\n").split(",", 4)
            ts = float(parts[0])
            ev = parts[1]
            session = parts[2] if len(parts) > 3 else "0"
            plen = int(parts[-2])
            payload = parse_hex(parts[-1])
            rows.append(dict(ts=ts, event=ev, session=session, plen=plen, payload=payload))
    return rows


def build_features(rows, stats=None):
    """Return (feat[n,5], sem_score[n], ok[n]) per packet, grouped by event (stream)."""
    groups = {}
    for r in rows:
        groups.setdefault(r["event"], []).append(r)
    feats, sem, ok = [], [], []
    for ev, grp in groups.items():
        grp.sort(key=lambda r: r["ts"])
        prev = None
        for r in grp:
            b = r["payload"]
            a = np.frombuffer(b, dtype=np.uint8) if len(b) else np.zeros(0, dtype=np.uint8)
            f = np.array([
                byte_entropy(b),
                float(a.mean()) if len(a) else 0.0,
                float(a.std()) if len(a) else 0.0,
                (r["ts"] - prev["ts"]) if prev else 0.0,
                (r["plen"] - prev["plen"]) if prev else 0.0,
            ], dtype=np.float32)
            feats.append(f)
            vals = decode_payload(b)
            if vals is not None and stats is not None:
                sem.append(semantic_score(vals, *stats))
                ok.append(1.0)
            else:
                sem.append(0.0)
                ok.append(0.0)
            prev = r
    return np.stack(feats), np.array(sem), np.array(ok)


def best_threshold(y, s):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return th[np.argmax(f1)] if len(th) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--n-est", type=int, default=300)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    if not csvs:
        raise SystemExit("no *_cli.csv found in --gen")

    # build rows + labels + scenario per csv
    all_rows, all_labels, all_scen = [], [], []
    for sc, path in enumerate(csvs):
        rows = read_rows(path)
        label = 0 if os.path.basename(path).startswith("normal") else 1
        all_rows.extend(rows)
        all_labels += [label] * len(rows)
        all_scen += [sc] * len(rows)
    print(f"total messages: {len(all_rows)}")

    # learn semantic stats from normal rows
    normal_vals = []
    for r, l in zip(all_rows, all_labels):
        if l == 0:
            v = decode_payload(r["payload"])
            if v is not None:
                normal_vals.append(v)
    if not normal_vals:
        raise SystemExit("no normal payloads to learn semantic stats")
    stats = learn_semantic_stats(normal_vals)
    print(f"semantic stats learned from {len(normal_vals)} normal payloads: "
          f"mean={stats[0].round(2)} std={stats[1].round(2)} range=[{stats[2].round(2)},{stats[3].round(2)}]")

    # features
    feat_rows = [r for r in all_rows]
    feats, sem, ok = build_features(feat_rows, stats)
    labels = np.array(all_labels, dtype=int)
    scen = np.array(all_scen, dtype=int)
    # keep only samples where semantic decoding succeeded
    mask = ok > 0
    feats, sem, labels, scen = feats[mask], sem[mask], labels[mask], scen[mask]
    print(f"decoded samples: {len(labels)}")

    Xtr, Xte, str_, s_te, ytr, yte, scen_tr, scen_te = train_test_split(
        feats, sem, labels, scen, test_size=0.3, random_state=SEED, stratify=labels)

    # B1 behavioral-only GBM
    clf = GradientBoostingClassifier(n_estimators=args.n_est, max_depth=3, random_state=SEED)
    clf.fit(Xtr, ytr)
    b1 = clf.predict_proba(Xte)[:, 1]
    # B2 semantic-only (threshold calibrated on train)
    thr_sem = best_threshold(ytr, str_)
    b2 = (s_te >= thr_sem).astype(int)
    # Fused: behavioral proba OR semantic flag
    thr_b1 = best_threshold(yte, b1)
    fused = ((b1 >= thr_b1) | (s_te >= thr_sem)).astype(int)

    def report(name, pred):
        if np.unique(yte).size > 1:
            print(f"{name:6s} precision={precision_score(yte, pred):.4f} recall={recall_score(yte, pred):.4f} "
                  f"f1={f1_score(yte, pred):.4f}")

    print("\n===== OVERALL (imbalanced test) =====")
    report("B1", (b1 >= thr_b1).astype(int))
    report("B2", b2)
    report("FUS", fused)
    # per-scenario recall
    print("\n===== PER-SCENARIO recall =====")
    for sid in np.unique(scen_te):
        m = (scen_te == sid)
        name = os.path.basename(csvs[int(sid)])
        sy = yte[m]; sp = fused[m]; sb = (b1[m] >= thr_b1).astype(int); ss = b2[m]
        def rec(pred, y):
            npos = int((y == 1).sum())
            return int(((y == 1) & (pred == 1)).sum()) / npos if npos > 0 else float('nan')
        print(f"  {name:24s} n={int(m.sum()):5d}  B1_rec={rec(sb, sy):.3f}  B2sem_rec={rec(ss, sy):.3f}  FUS_rec={rec(sp, sy):.3f}")


if __name__ == "__main__":
    main()
