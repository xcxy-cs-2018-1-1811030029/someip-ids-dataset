"""
baselines.py

Behavioral-only baseline: gradient-boosted trees on View A features only
(no payload/View B). This is the "prior work" style detector (cf. Kim et al. XGBoost).

Usage:
  python3 someip_ids/baselines.py --data data/
"""

import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 1234


def best_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return th[np.argmax(f1)] if len(th) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-est", type=int, default=300)
    args = ap.parse_args()

    f_a = np.load(os.path.join(args.data, "f_a.npy"))
    labels = np.load(os.path.join(args.data, "labels.npy"))
    scen = np.load(os.path.join(args.data, "scenario.npy")) if os.path.exists(
        os.path.join(args.data, "scenario.npy")) else None

    stratify = labels.astype(int) if np.unique(labels).size > 1 else None
    if scen is not None:
        Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
            f_a, labels, scen, test_size=0.3, random_state=SEED, stratify=stratify)
    else:
        Xtr, Xte, ytr, yte = train_test_split(f_a, labels, test_size=0.3,
                                              random_state=SEED, stratify=stratify)
        str_te = None

    clf = GradientBoostingClassifier(n_estimators=args.n_est, learning_rate=0.05,
                                     max_depth=3, random_state=SEED)
    clf.fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]
    y = yte

    print("===== BASELINE: gradient boosting on View A (behavioral only) =====")
    if np.unique(y).size > 1:
        thr = best_threshold(y, score)
        p = (score >= thr).astype(int)
        print(f"n={len(y)}, positive={y.mean():.4f}, threshold={thr:.4f}")
        print(f"precision={precision_score(y, p):.4f}  recall={recall_score(y, p):.4f}  f1={f1_score(y, p):.4f}")
        print(f"ROC-AUC={roc_auc_score(y, score):.4f}  PR-AUC={average_precision_score(y, score):.4f}")

        print("\n----- per-scenario (global threshold) ---")
        thr = best_threshold(y, score)
        if str_te is not None:
            for sid in np.unique(str_te):
                m = (str_te == sid)
                s_true, s_score = y[m], score[m]
                s_pred = (s_score >= thr).astype(int)
                npos = int((s_true == 1).sum())
                if npos > 0:
                    rec = int(((s_true == 1) & (s_pred == 1)).sum()) / npos
                    print(f"  scenario {int(sid)}: n={int(m.sum())}  pos_recall={rec:.4f}  (attack detection)")
                else:
                    nneg = int((s_true == 0).sum())
                    fp = int(((s_true == 0) & (s_pred == 1)).sum())
                    print(f"  scenario {int(sid)}: n={int(m.sum())}  false_positive={fp/nneg:.4f}  (normal)")


if __name__ == "__main__":
    main()
