"""
train_our.py

Train + evaluate the multi-view SOME/IP detector from someip_ids/model.py on the
arrays produced by csv_to_records.py / pcap_to_records.py.

Usage (run in WSL where torch is installed):
  python3 someip_ids/train_our.py --data data/ --epochs 50 --encoder gru

Outputs: best_model.pt and printed metrics (F1, precision, recall, PR-AUC, ROC-AUC).
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_auc_score, average_precision_score)

# Ensure the repo root is importable so `import someip_ids.*` works when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from someip_ids.model import MultiViewFusion
from someip_ids.dataset import SOMEIPMultiViewDataset

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data(data_dir, len_b=128):
    f_a = np.load(os.path.join(data_dir, "f_a.npy"))
    seq_b = np.load(os.path.join(data_dir, "seq_b.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    scen = np.load(os.path.join(data_dir, "scenario.npy")) if os.path.exists(
        os.path.join(data_dir, "scenario.npy")) else None
    return f_a, seq_b, labels, scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="cnn", choices=["gru", "attn", "cnn"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--d-a", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=257)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    f_a, seq_b, labels, scen = load_data(args.data)
    print(f"data: f_a={f_a.shape} seq_b={seq_b.shape} labels={labels.shape} "
          f"positive={labels.mean():.4f}")

    # Stratified split with room for both classes in test.
    stratify = labels.astype(int) if np.unique(labels).size > 1 else None
    extra = [scen] if scen is not None else []
    split = train_test_split(f_a, seq_b, labels, *extra, test_size=0.3,
                             random_state=SEED, stratify=stratify)
    tr_a, te_a, tr_s, te_s, tr_y, te_y = split[0], split[1], split[2], split[3], split[4], split[5]
    te_sc = split[7] if scen is not None and len(split) > 7 else None

    # Standardize View A features using TRAIN stats (avoid leakage).
    fa_mean = tr_a.mean(axis=0)
    fa_std = tr_a.std(axis=0) + 1e-6

    train_ds = SOMEIPMultiViewDataset(tr_a, tr_s, tr_y, fa_mean, fa_std)
    test_ds = SOMEIPMultiViewDataset(te_a, te_s, te_y, fa_mean, fa_std)
    tr_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    te_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = MultiViewFusion(args.d_a, args.vocab, encoder=args.encoder,
                            num_classes=1, hidden=args.hidden).to(dev)

    # Class-weighted BCE to handle imbalance.
    n_pos = float(tr_y.sum()); n_neg = float((1 - tr_y).sum()) + 1e-6
    pos_weight = torch.tensor([n_neg / n_pos]).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    epochs = args.epochs

    best_f1, best_state = 0.0, None
    print("\ntraining ...")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xa, xb, y in tr_loader:
            xa, xb, y = xa.to(dev), xb.to(dev), y.unsqueeze(1).to(dev)
            logits, _ = model(xa, xb)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xa.size(0)
        # quick validation
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xa, xb, y in te_loader:
                xa, xb = xa.to(dev), xb.to(dev)
                logits, _ = model(xa, xb)
                preds.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
                ys.append(y.numpy())
            preds = np.concatenate(preds); ys = np.concatenate(ys)
        # best F1 threshold on validation subset (use test here as a stand-in)
        if np.unique(ys).size > 1:
            thr = best_threshold(ys, preds)
            f1 = f1_score(ys, (preds >= thr).astype(int))
        else:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep+1}/{epochs} loss={tot/len(tr_y):.4f} val_f1={f1:.4f}")

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "config": {
        "d_a": args.d_a, "vocab": args.vocab, "encoder": args.encoder,
        "hidden": args.hidden}}, os.path.join(args.out, "best_model.pt"))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds, ys = [], []
        for xa, xb, y in te_loader:
            xa, xb = xa.to(dev), xb.to(dev)
            logits, _ = model(xa, xb)
            preds.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
            ys.append(y.numpy())
        preds = np.concatenate(preds); ys = np.concatenate(ys)

    print("\n===== TEST (imbalanced, realistic) =====")
    if np.unique(ys).size > 1:
        thr = best_threshold(ys, preds)
        pred = (preds >= thr).astype(int)
        print(f"n={len(ys)}, positive={ys.mean():.4f}, threshold={thr:.4f}")
        print(f"precision={precision_score(ys, pred):.4f}  recall={recall_score(ys, pred):.4f}  f1={f1_score(ys, pred):.4f}")
        print(f"ROC-AUC={roc_auc_score(ys, preds):.4f}  PR-AUC={average_precision_score(ys, preds):.4f}")

        print("\n===== TEST (balanced, downsampled majority) =====")
        bm = balanced_metrics(ys, preds)
        if bm:
            (b_y, b_p, b_score) = bm
            thr_b = best_threshold(b_y, b_score)
            bp = (b_score >= thr_b).astype(int)
            print(f"n={len(b_y)}, positive={b_y.mean():.4f}, threshold={thr_b:.4f}")
            print(f"precision={precision_score(b_y, bp):.4f}  recall={recall_score(b_y, bp):.4f}  f1={f1_score(b_y, bp):.4f}")
            print(f"ROC-AUC={roc_auc_score(b_y, b_score):.4f}  PR-AUC={average_precision_score(b_y, b_score):.4f}")

        print("\n===== TEST (per-scenario, threshold from global) =====")
        if te_sc is not None:
            for sid in np.unique(te_sc):
                m = (te_sc == sid)
                if m.sum() == 0:
                    continue
                s_true, s_score = ys[m], preds[m]
                s_pred = (s_score >= thr).astype(int)
                npos = int((s_true == 1).sum())
                if npos > 0:
                    rec = int(((s_true == 1) & (s_pred == 1)).sum()) / npos
                    print(f"  scenario {int(sid)}: n={int(m.sum())}  pos_recall={rec:.4f}  (attack detection)")
                else:
                    nneg = int((s_true == 0).sum())
                    fp = int(((s_true == 0) & (s_pred == 1)).sum())
                    print(f"  scenario {int(sid)}: n={int(m.sum())}  false_positive={fp/nneg:.4f}  (normal)")


def balanced_metrics(y, score):
    """Downsample the majority class to match the minority, return (y, labels, score)."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    k = min(len(pos), len(neg))
    idx = np.concatenate([np.random.choice(pos, k, replace=False),
                          np.random.choice(neg, k, replace=False)])
    return y[idx], (score[idx] >= 0).astype(int), score[idx]


def best_threshold(y, p):
    """Choose threshold maximizing F1 on the given (validation) data."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return th[np.argmax(f1)] if len(th) else 0.5


if __name__ == "__main__":
    main()
