"""
diag_gru.py -- diagnose why the benchmark GRU outputs constant scores (ROC 0.5).

Test 1 (synthetic): first-byte-separable 12-byte payloads. If GRUCls + the same
training loop learns this, the code path is sound and the failure is data/model
mismatch, not a bug.
Test 2 (real data): balanced subsample of the real corpus. GRUCls (last hidden
state) vs GRUMean (mean-pooled) vs lower learning rate, to see whether the
last-state readout is the failure mode.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B

SEED = 1234


def quick_train(model, toks_tr, y_tr, toks_ev, epochs=10, batch=32, lr=1e-3, verbose=True):
    from torch.utils.data import DataLoader, TensorDataset
    xt = torch.tensor(toks_tr)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = model.to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose and (ep % 2 == 0 or ep == epochs - 1):
            print(f"  ep{ep:2d} loss={tot / max(nb, 1):.4f}")
    m.eval()
    with torch.no_grad():
        sc = torch.sigmoid(m(torch.tensor(toks_ev).to(dev))).squeeze(1).cpu().numpy()
    return sc


def score_stats(name, sc, y):
    print(f"{name}: score min={sc.min():.3f} max={sc.max():.3f} std={sc.std():.4f} "
          f"ROC={roc_auc_score(y, sc):.4f}")


class GRUMean(nn.Module):
    """Same GRU but mean-pooled readout instead of last hidden state."""

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
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    if not args.skip_synth:
        print("=== Test 1: synthetic first-byte-separable task ===")
        rng = np.random.default_rng(0)
        toks = rng.integers(1, 256, (4000, 12)).astype(np.int64)
        y = (toks[:, 0] >= 128).astype(np.int64)
        sc = quick_train(B.GRUCls(), toks[:3000], y[:3000], toks[3000:], epochs=10)
        score_stats("GRUCls (last-state)", sc, y[3000:])
        sc = quick_train(GRUMean(), toks[:3000], y[:3000], toks[3000:], epochs=10)
        score_stats("GRUMean (mean-pool)", sc, y[3000:])

    if not args.skip_real:
        print("\n=== Test 2: real corpus, balanced 20k subsample ===")
        csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
        rows, labels = [], []
        for p in csvs:
            sc_, _, _ = B.parse_name(p)
            r = B.read_rows(p)
            rows += r
            labels += [0 if sc_ == "normal" else 1] * len(r)
        from someip_ids.semantics import decode_payload, learn_semantic_stats
        normal_vals = [decode_payload(x["payload"]) for x, l in zip(rows, labels) if l == 0]
        stats = learn_semantic_stats([v for v in normal_vals if v])
        _, _, toks, ok = B.build(rows, stats)
        labels = np.array(labels)
        mask = ok > 0
        toks, labels = toks[mask], labels[mask]
        rng = np.random.default_rng(SEED)
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        sub = np.concatenate([rng.choice(pos, 10000, replace=False),
                              rng.choice(neg, 10000, replace=False)])
        rng.shuffle(sub)
        tk, y = toks[sub], labels[sub]

        for name, model, lr in [("GRUCls lr1e-3", B.GRUCls(), 1e-3),
                                ("GRUCls lr1e-4", B.GRUCls(), 1e-4),
                                ("GRUMean lr1e-3", GRUMean(), 1e-3)]:
            print(f"-- {name} --")
            sc = quick_train(model, tk[:15000], y[:15000], tk[15000:],
                             epochs=30, lr=lr, verbose=False)
            score_stats(name, sc, y[15000:])


if __name__ == "__main__":
    main()
