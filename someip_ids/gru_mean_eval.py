"""
gru_mean_eval.py

Corrected-GRU evaluation in the full benchmark setting. The released benchmark's
GRU reads out the LAST hidden state; diag_gru.py showed this readout collapses to
constant scores on the real corpus (ROC 0.5 at every training prevalence), while
the same GRU with a mean-pooled readout reaches ROC 0.81 on a balanced subsample.
This script evaluates the mean-pooled GRU exactly like benchmark.py (full corpus,
same splits, 30 epochs, class-weighted BCE) so the paper can report a corrected
deep-sequence number.

Usage:  python3 someip_ids/gru_mean_eval.py --gen vs_ip_gen/generate/ --splits random
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats

SEED = 1234


class GRUMean(nn.Module):
    """Same GRU as benchmark.GRUCls but mean-pooled readout."""

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
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--splits", default="random")
    ap.add_argument("--out", default="vs_ip_gen/results/gru_mean_eval.txt")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    rows, labels, scen, run = [], [], [], []
    for p in csvs:
        sc, ru, _ = B.parse_name(p)
        r = B.read_rows(p)
        rows += r
        labels += [0 if sc == "normal" else 1] * len(r)
        scen += [sc] * len(r)
        run += [ru] * len(r)
    ts_all = np.array([x["ts"] for x in rows])
    normal_vals = [decode_payload(x["payload"]) for x, l in zip(rows, labels) if l == 0]
    stats = learn_semantic_stats([v for v in normal_vals if v])
    _, _, toks, ok = B.build(rows, stats)
    labels = np.array(labels); scen = np.array(scen); run = np.array(run)
    mask = ok > 0
    toks, labels, scen, run, ts_all = (toks[mask], labels[mask], scen[mask],
                                       run[mask], ts_all[mask])

    out = []
    def log(s=""):
        print(s)
        out.append(s)

    log(f"messages={len(labels)}  overall_positive={labels.mean():.3f}")
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split == "run":
            max_run = {s: max(int(run[i]) for i in range(len(run)) if scen[i] == s)
                       for s in set(scen)}
            tr_i = np.array([i for i in range(len(labels)) if int(run[i]) != max_run[scen[i]]])
            te_i = np.array([i for i in range(len(labels)) if int(run[i]) == max_run[scen[i]]])
        elif split == "temporal":
            stream_key = np.array([f"{scen[i]}_{run[i]}" for i in range(len(labels))])
            te_mask = np.zeros(len(labels), bool)
            for key in np.unique(stream_key):
                idxs = np.where(stream_key == key)[0]
                o = np.argsort(ts_all[idxs])
                hold = max(1, int(len(o) * 0.2))
                te_mask[idxs[o[-hold:]]] = True
            tr_i = np.where(~te_mask)[0]; te_i = np.where(te_mask)[0]
        else:
            tr_i, te_i = train_test_split(np.arange(len(labels)), test_size=0.3,
                                          random_state=SEED, stratify=labels)
        yte = labels[te_i]
        log(f"\n===== split={split}  train_pos={labels[tr_i].mean():.3f} "
            f"test_pos={yte.mean():.3f} =====")
        t0 = time.time()
        score = B.train_torch(GRUMean(), toks[tr_i], labels[tr_i],
                              toks[te_i], yte, args.epochs)
        thr = B.best_threshold(yte, score)
        pred = (score >= thr).astype(int)
        log(f"GRU-mean-pooled  ROC={roc_auc_score(yte, score):.3f} "
            f"PR={average_precision_score(yte, score):.3f} "
            f"MCC={matthews_corrcoef(yte, pred):.3f}  "
            f"posrate={pred.mean():.3f}  ({int(time.time()-t0)//60} min)")
        scen_te = scen[te_i]
        log("  per-scenario recall:")
        for s in sorted(set(scen_te)):
            m = (scen_te == s) & (yte == 1)
            rec = float(pred[m].mean()) if m.sum() else float("nan")
            log(f"    {s:16s} rec={rec:.3f}")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
