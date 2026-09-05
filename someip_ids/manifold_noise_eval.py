"""
manifold_noise_eval.py

A/B evaluation for the noiseless-manifold question: run the benchmark detectors
(GBM, semantic, TCN, mean-pooled GRU) on ONE generator directory (either the
noiseless clean control or the sensor-noise variant), fixed random split, so the
two runs can be diffed. Answers whether evaluating contextual tampering on a
deterministic, noiseless generator manifold inflates deep-model anomaly
boundaries.

Usage:
  python3 someip_ids/manifold_noise_eval.py --gen vs_ip_gen/generate/clean_ctrl/ \
      --out vs_ip_gen/results/manifold_clean.txt
  python3 someip_ids/manifold_noise_eval.py --gen vs_ip_gen/generate/noisy/ \
      --out vs_ip_gen/results/manifold_noisy.txt
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats

SEED = 1234


class GRUMean(nn.Module):
    def __init__(self, vocab=257, emb=48, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.linear = nn.Linear(hid, 1)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.gru(e)
        return self.linear(out.mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    if not csvs:
        raise SystemExit(f"no *_cli.csv in {args.gen}")
    rows, labels, scen = [], [], []
    for p in csvs:
        sc, _, _ = B.parse_name(p)
        r = B.read_rows(p)
        rows += r
        labels += [0 if sc == "normal" else 1] * len(r)
        scen += [sc] * len(r)
    normal_vals = [decode_payload(x["payload"]) for x, l in zip(rows, labels) if l == 0]
    stats = learn_semantic_stats([v for v in normal_vals if v])
    feats, sem, toks, ok = B.build(rows, stats)
    labels = np.array(labels); scen = np.array(scen)
    mask = ok > 0
    feats, sem, toks, labels, scen = feats[mask], sem[mask], toks[mask], \
        labels[mask], scen[mask]

    tr_i, te_i = train_test_split(np.arange(len(labels)), test_size=0.3,
                                  random_state=SEED, stratify=labels)
    yte = labels[te_i]; scen_te = scen[te_i]

    out = []
    def log(s=""):
        print(s)
        out.append(s)

    log(f"gen={args.gen}  messages={len(labels)}  positive={labels.mean():.3f}")
    log(f"train_pos={labels[tr_i].mean():.3f}  test_pos={yte.mean():.3f}")

    def report(name, score, scen_recs=False):
        thr = B.best_threshold(yte, score)
        pred = (score >= thr).astype(int)
        log(f"{name:14s} ROC={roc_auc_score(yte, score):.3f} "
            f"PR={average_precision_score(yte, score):.3f} "
            f"MCC={matthews_corrcoef(yte, pred):.3f} "
            f"P={precision_score(yte, pred):.3f} R={recall_score(yte, pred):.3f}")
        if scen_recs:
            for s in sorted(set(scen_te)):
                m = (scen_te == s) & (yte == 1)
                rec = float(pred[m].mean()) if m.sum() else float("nan")
                log(f"    {s:14s} rec={rec:.3f}")

    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
    gbm.fit(feats[tr_i], labels[tr_i])
    report("GBM", gbm.predict_proba(feats[te_i])[:, 1], scen_recs=True)

    thr_s = float(np.percentile(sem[tr_i][labels[tr_i] == 0], 99))
    report("semantic", sem[te_i], scen_recs=True)

    for name, model in [("TCN", B.TCN()), ("GRU-mean", GRUMean())]:
        score = B.train_torch(model, toks[tr_i], labels[tr_i], toks[te_i], yte,
                              args.epochs)
        report(name, score, scen_recs=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
