#!/usr/bin/env python3
"""
external_baselines.py -- reimplementations of prior SOME/IP IDS methods on OUR
dataset, under the same unified protocol (run-independent split, fixed 1% FPR
global train threshold, overall ROC/PR/MCC), so the paper's comparison table
contains NUMERICAL comparisons, not just a positioning table.

  Heo2022 (IEICE Trans. Inf. & Syst., E105-D(11)): "SOME/IP intrusion detection
    system using machine learning". RF on 10 RFECV-selected features from the
    SOME/IP header + reception timestamp: timestamp, session ID, return code,
    src/dst port, src/dst MAC, service ID, method ID, client ID. Our CSVs expose
    timestamp, event (method/event) ID, session ID, payload length, so the
    faithful mapping is RF on [dt, event_id, session_id, plen, dl]. The
    remaining fields (ports, MACs, return code, client ID) are CONSTANT in the
    single-service topology -- exactly the protocol-state finding of Sec. 4.4.

  Kim2026 (Systems 14(2):196): "XGBoost-based anomaly detection framework".
    Behavior-centric features (time-interval variation, payload entropy /
    likelihood, payload and length change rates) -> XGBoost. Mapping:
    [dt, entropy, byte-mean, byte-std, dl]. CTGAN augmentation is orthogonal
    imbalance handling; our balanced-ablation analysis (Sec. 8.1) covers the
    imbalance question, so the base XGBoost is reported without augmentation.

Usage:
  python3 someip_ids/external_baselines.py --gen vs_ip_gen/generate/ \
      --out vs_ip_gen/results/external_baselines.txt
"""

import argparse
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score,
                             f1_score)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats

SEED = 1234
np.random.seed(SEED)


def tpr_at_fpr(train_score, ytr, test_score, yte, scen_te, fpr=0.01):
    ns = train_score[ytr == 0]
    thr = float(np.percentile(ns, (1 - fpr) * 100)) if len(ns) else 0.5
    pred = (test_score >= thr).astype(int)
    out = {}
    for s in sorted(set(scen_te)):
        m = (scen_te == s)
        sy = yte[m]
        np_ = int((sy == 1).sum())
        out[s] = float(((sy == 1) & (pred[m] == 1)).sum() / np_) if np_ else float("nan")
    return out


def report(ytr, yte, tr_s, te_s, scen_te, name, out):
    thr = B.best_threshold(yte, te_s)
    pred = (te_s >= thr).astype(int)
    row = (f"{name:22s} ROC={roc_auc_score(yte, te_s):.3f} "
           f"PR={average_precision_score(yte, te_s):.3f} "
           f"MCC={matthews_corrcoef(yte, pred):.3f} "
           f"P={precision_score(yte, pred):.3f} R={recall_score(yte, pred):.3f} "
           f"F1={f1_score(yte, pred):.3f}")
    print(row)
    out.append(row)
    return tpr_at_fpr(tr_s, ytr, te_s, yte, scen_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--out", default="vs_ip_gen/results/external_baselines.txt")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    if not csvs:
        raise SystemExit("no *_cli.csv in --gen")
    rows, labels, scen, run = [], [], [], []
    for p in csvs:
        sc, ru, it = B.parse_name(p)
        r = B.read_rows(p)
        rows += r
        labels += [0 if sc == "normal" else 1] * len(r)
        scen += [sc] * len(r)
        run += [ru] * len(r)
    normal_vals = [decode_payload(x["payload"]) for x, l in zip(rows, labels) if l == 0]
    stats = learn_semantic_stats([v for v in normal_vals if v])

    feats, sem, toks, ok = B.build(rows, stats)
    labels = np.array(labels)
    scen = np.array(scen)
    run = np.array(run)
    mask = ok > 0
    feats, labels, scen, run = feats[mask], labels[mask], scen[mask], run[mask]
    rows = [r for r, m in zip(rows, mask) if m]

    # run-independent split (identical to benchmark --split run)
    max_run = {s: max(int(run[i]) for i in range(len(run)) if scen[i] == s)
               for s in set(scen)}
    tr_i = np.array([i for i in range(len(labels)) if int(run[i]) != max_run[scen[i]]])
    te_i = np.array([i for i in range(len(labels)) if int(run[i]) == max_run[scen[i]]])
    ytr, yte = labels[tr_i], labels[te_i]
    scen_te = scen[te_i]

    def sess_num(r):
        try:
            return float(int(str(r["session"]), 0))
        except (ValueError, TypeError):
            return 0.0

    # Heo2022-style: header fields + timestamp-derived dt
    heo = np.array([[feats[i, 3],                       # dt (timestamp)
                     float(rows[i]["event"]),           # method/event ID
                     sess_num(rows[i]),                 # session ID
                     float(rows[i]["plen"]),            # payload length
                     feats[i, 4]]                       # length delta
                    for i in range(len(labels))], dtype=np.float32)

    # Kim2026-style: behavior-centric features
    kim = np.array([[feats[i, 3],                       # time-interval variation
                     feats[i, 0],                       # payload entropy
                     feats[i, 1],                       # payload byte mean
                     feats[i, 2],                       # payload byte std
                     feats[i, 4]]                       # length change rate
                    for i in range(len(labels))], dtype=np.float32)

    out = []
    out.append(f"messages={len(labels)}  split=run  protocol=Table7(1%FPR global thr)")
    out.append(f"train={len(tr_i)}  test={len(te_i)}  test_pos={yte.mean():.3f}")
    tpr = {}

    print("--- Heo2022-style RF (header+timing) ---")
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(heo[tr_i], ytr)
    tpr["Heo2022-RF"] = report(ytr, yte, rf.predict_proba(heo[tr_i])[:, 1],
                               rf.predict_proba(heo[te_i])[:, 1], scen_te,
                               "Heo2022 RF (hdr+tim)", out)

    print("--- Kim2026-style XGBoost (behavior-centric) ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, eval_metric="logloss")
    xgb.fit(kim[tr_i], ytr)
    tpr["Kim2026-XGB"] = report(ytr, yte, xgb.predict_proba(kim[tr_i])[:, 1],
                                xgb.predict_proba(kim[te_i])[:, 1], scen_te,
                                "Kim2026 XGB (behav.)", out)

    scen_names = sorted(set(scen_te))
    out.append("\nFPR_TPR detector      " + " ".join(f"{s:>9s}" for s in scen_names))
    for name in ["Heo2022-RF", "Kim2026-XGB"]:
        out.append(f"FPR_TPR {name:12s} | " + " | ".join(f"{tpr[name][s]:9.3f}"
                                                         for s in scen_names))
        print(f"FPR_TPR {name:12s} | " + " | ".join(f"{tpr[name][s]:9.3f}"
                                                    for s in scen_names))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
