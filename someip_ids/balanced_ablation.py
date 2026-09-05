"""
balanced_ablation.py

Ablation study answering the reviewer question: is the GRU collapse (ROC-AUC 0.50,
MCC 0.000) an artifact of the 91.5%-attack training distribution rather than an
architecture limitation?

The SAME GRU/TCN (identical hyper-parameters) is trained at several training
prevalences (original ~91.5%, and downsampled 50% / 30% / 10%) and always evaluated
on the SAME fixed test split (kept at its natural distribution). If performance
recovers as the training set becomes balanced, the collapse is a data-distribution
artifact. A "posrate" column reports the fraction of test samples predicted positive
at the best-F1 threshold -- values near 1.0 indicate the all-positive collapse.

Usage:
  python3 someip_ids/balanced_ablation.py --gen vs_ip_gen/generate/ --splits random --ratios orig,0.5,0.3,0.1
  python3 someip_ids/balanced_ablation.py --gen vs_ip_gen/generate/ --splits run --ratios orig,0.1
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score)

SEED = 1234


def make_splits(split, labels, scen, run, ts_all):
    """Reproduce benchmark.py split logic exactly. Returns (tr_i, te_i)."""
    if split == "temporal":
        stream_key = np.array([f"{scen[i]}_{run[i]}" for i in range(len(labels))])
        te_mask = np.zeros(len(labels), bool)
        for key in np.unique(stream_key):
            idxs = np.where(stream_key == key)[0]
            o = np.argsort(ts_all[idxs])
            hold = max(1, int(len(o) * 0.2))
            te_mask[idxs[o[-hold:]]] = True
        return np.where(~te_mask)[0], np.where(te_mask)[0]
    if split == "run":
        max_run = {s: max(int(run[i]) for i in range(len(run)) if scen[i] == s)
                   for s in set(scen)}
        tr_i = np.array([i for i in range(len(labels)) if int(run[i]) != max_run[scen[i]]])
        te_i = np.array([i for i in range(len(labels)) if int(run[i]) == max_run[scen[i]]])
        return tr_i, te_i
    from sklearn.model_selection import train_test_split
    tr_i, te_i = train_test_split(np.arange(len(labels)), test_size=0.3,
                                  random_state=SEED, stratify=labels)
    return tr_i, te_i


def subsample_prevalence(tr_i, y, r, rng):
    """Downsample training positives so positives make up fraction r of training."""
    pos = tr_i[y[tr_i] == 1]
    neg = tr_i[y[tr_i] == 0]
    if r is None or len(neg) == 0:
        return tr_i
    n_pos = min(len(pos), int(round(len(neg) * r / (1.0 - r))))
    pos_sub = rng.choice(pos, size=max(n_pos, 1), replace=False)
    return np.concatenate([pos_sub, neg])


def evaluate(score, y):
    if np.unique(y).size < 2:
        return dict(roc=float("nan"), pr=float("nan"), mcc=float("nan"), posrate=float("nan"))
    thr = B.best_threshold(y, score)
    p = (score >= thr).astype(int)
    return dict(
        roc=roc_auc_score(y, score),
        pr=average_precision_score(y, score),
        mcc=matthews_corrcoef(y, p),
        posrate=float(p.mean()),
        prec=precision_score(y, p),
        rec=recall_score(y, p),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--splits", default="random", help="comma-separated: random,run,temporal")
    ap.add_argument("--ratios", default="orig,0.5,0.3,0.1",
                    help="training prevalences; 'orig' = no subsampling")
    ap.add_argument("--out", default="vs_ip_gen/results/balanced_ablation.txt")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    if not csvs:
        raise SystemExit("no *_cli.csv in --gen")
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
    print(f"messages={len(labels)}  overall_positive={labels.mean():.3f}")

    lines = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        tr_i, te_i = make_splits(split, labels, scen, run, ts_all)
        yte = labels[te_i]
        print(f"\n===== split={split}  train_pos={labels[tr_i].mean():.3f} "
              f"test_pos={yte.mean():.3f} =====")
        print(f"{'model':6s} {'train_prev':>10s} {'ROC':>7s} {'PR':>7s} {'MCC':>7s} "
              f"{'posrate':>8s} {'P':>6s} {'R':>6s} {'min':>7s}")
        for ratio_s in [s.strip() for s in args.ratios.split(",") if s.strip()]:
            r = None if ratio_s == "orig" else float(ratio_s)
            rng = np.random.default_rng(SEED)
            tr_sub = subsample_prevalence(tr_i, labels, r, rng)
            prev = labels[tr_sub].mean()
            for name, model in [("GRU", B.GRUCls()), ("TCN", B.TCN())]:
                t0 = time.time()
                score = B.train_torch(model, toks[tr_sub], labels[tr_sub],
                                      toks[te_i], yte, args.epochs)
                m = evaluate(score, yte)
                print(f"{name:6s} {prev:10.3f} {m['roc']:7.3f} {m['pr']:7.3f} "
                      f"{m['mcc']:7.3f} {m['posrate']:8.3f} {m['prec']:6.3f} "
                      f"{m['rec']:6.3f} {int(time.time()-t0)//60:5d}m")
                lines.append((split, name, ratio_s, prev, m))

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w") as f:
        f.write("split model train_prev ROC PR MCC posrate prec rec\n")
        for split, name, ratio_s, prev, m in lines:
            f.write(f"{split} {name} {ratio_s} {prev:.3f} {m['roc']:.3f} {m['pr']:.3f} "
                    f"{m['mcc']:.3f} {m['posrate']:.3f} {m['prec']:.3f} {m['rec']:.3f}\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
