"""
benchmark.py

Comprehensive benchmark for the SOME/IP intrusion-detection dataset (Q2-tier).

Detector families:
  B-behavior  : gradient boosting + random forest on behavioral + protocol-state features
  B-semantic  : payload semantic-validity (per-signal z-score / range violation)
  B-deep-gru  : GRU on payload byte tokens (torch)
  B-deep-tcn  : temporal convolutional network on payload byte tokens (torch)
  B-fused     : hard-decision OR of behavioral and semantic

Split strategies (to avoid temporal leakage):
  random    : stratified random split
  temporal  : chronological (sorted by timestamp) -> first 60/20/20
  run       : run-independent (train on some runs, test on held-out runs per scenario)

Outputs overall + per-scenario metrics per split. Run all three splits for the paper.

Usage (in WSL, torch installed):
  python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split random --epochs 30
  python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split temporal --epochs 30
  python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split run --epochs 30
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             matthews_corrcoef)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids.semantics import decode_payload, learn_semantic_stats, semantic_score

SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


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


def parse_name(path):
    base = os.path.basename(path).replace("_cli.csv", "")
    base = re.sub(r"_run\d+$", "", base)          # strip run suffix -> dos_i0.3
    m_int = re.search(r"_i([\d.]+)$", base)
    intensity = float(m_int.group(1)) if m_int else 1.0
    scenario = re.sub(r"_i[\d.]+$", "", base)     # strip intensity -> dos
    m = re.search(r"_run(\d+)$", os.path.basename(path).replace("_cli.csv", ""))
    run = int(m.group(1)) if m else 1
    return scenario, run, intensity


def read_rows(path):
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",", 4)
            rows.append(dict(ts=float(parts[0]), event=int(parts[1]),
                             session=parts[2] if len(parts) > 3 else "0",
                             plen=int(parts[-2]), payload=parse_hex(parts[-1])))
    return rows


def build(rows, stats, len_b=128):
    """Return behavioral+protocol-state features, semantic scores, payload tokens, ok flags.

    IMPORTANT: features are emitted in the SAME order as the input `rows` (each processed
    row -> one feature), so callers can align labels/scenario/run arrays by index. Stream
    properties (dt, burst state) are computed per-event via an incremental prev/state map,
    which handles multiple runs/files correctly without reordering the output.
    """
    feats, sem, toks, ok = [], [], [], []
    prev = {}            # event -> last row of that event stream
    state = {}           # event -> current burst counter
    for r in rows:
        ev = r["event"]
        b = r["payload"]
        a = np.frombuffer(b, dtype=np.uint8) if len(b) else np.zeros(0, np.uint8)
        p = prev.get(ev)
        dt = (r["ts"] - p["ts"]) if p else 0.0
        dl = (r["plen"] - p["plen"]) if p else 0.0
        st = state.get(ev, 0.0)
        st = st + 1.0 if dt < 0.5 else 0.0
        state[ev] = st
        feats.append(np.array([byte_entropy(b),
                               float(a.mean()) if len(a) else 0.0,
                               float(a.std()) if len(a) else 0.0,
                               dt, dl,
                               st,                                   # protocol-state (burstiness)
                               float(ev % 1000) / 1000.0,            # protocol-state (event id)
                               ], dtype=np.float32))
        t = [int(x) + 1 for x in b[:len_b]]
        t = t + [0] * (len_b - len(t)) if len(t) < len_b else t[:len_b]
        toks.append(np.array(t, dtype=np.int64))
        vals = decode_payload(b)
        if vals is not None:
            sem.append(semantic_score(vals, *stats)); ok.append(1.0)
        else:
            # not a valid 12-byte signal frame (e.g. a short fuzz payload): keep the row
            # so byte/sequence detectors still see it; the semantic score is 0 (no value
            # violation claim), which is honest -- semantic simply does not cover it.
            sem.append(0.0); ok.append(1.0 if len(b) > 0 else 0.0)
        prev[ev] = r
    return np.stack(feats), np.array(sem), np.stack(toks), np.array(ok)


def best_threshold(y, s):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return th[np.argmax(f1)] if len(th) else 0.5


# ---------------- torch baselines ----------------
class GRUCls(nn.Module):
    def __init__(self, vocab=257, emb=48, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.linear = nn.Linear(hid, 1)
    def forward(self, x):
        e = self.emb(x); _, h = self.gru(e); return self.linear(h[-1])


class TCN(nn.Module):
    def __init__(self, vocab=257, emb=48, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv1 = nn.Conv1d(emb, hid, 3, padding=1)
        self.conv2 = nn.Conv1d(hid, hid, 3, padding=1)
        self.linear = nn.Linear(hid, 1)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = torch.relu(self.conv1(e)); c = torch.relu(self.conv2(c))
        return self.linear(c.max(dim=2).values)


def predict_batches(model, toks, dev, batch=64):
    """Forward over a large token array in mini-batches (avoids GPU OOM on the
    whole test set) with a one-time CPU fallback if the GPU still overflows."""
    active = dev
    m = model.to(active)
    outs = []
    for i in range(0, len(toks), batch):
        chunk = toks[i:i + batch]
        with torch.no_grad():
            try:
                z = m(torch.tensor(chunk).to(active)).squeeze(1)
            except RuntimeError:                      # CUDA OOM -> fall back to CPU once
                if active != "cpu":
                    active = "cpu"; m = model.to("cpu")
                    z = m(torch.tensor(chunk)).squeeze(1)
                else:
                    raise
        outs.append(z.cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0)


def train_torch(model, toks_tr, y_tr, toks_te, y_te, epochs, batch=32, lr=1e-3):
    from torch.utils.data import DataLoader, TensorDataset
    xt = torch.tensor(toks_tr); yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)
    d = dev
    m = model.to(d)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)]).to(d))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(d), yb.to(d)
            try:
                loss = crit(m(xb), yb)
            except RuntimeError:                       # CUDA OOM -> fall back to CPU
                if d != "cpu":
                    d = "cpu"; m = model.to(d)
                    crit = nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor([(1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)]).to(d))
                    opt = torch.optim.AdamW(m.parameters(), lr=lr)
                    xb, yb = xb.to(d), yb.to(d)
                    loss = crit(m(xb), yb)
                else:
                    raise
            opt.zero_grad(); loss.backward(); opt.step()
    te_score = predict_batches(m, toks_te, d)
    return 1 / (1 + np.exp(-te_score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split", default="random", choices=["random", "temporal", "run"])
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="timing-only vs byte-only feature comparison")
    ap.add_argument("--fpr-eval", action="store_true", help="fixed 1% FPR evaluation (per-attack TPR)")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.gen, "*_cli.csv")))
    if not csvs:
        raise SystemExit("no *_cli.csv in --gen")

    rows, labels, scen, run, inten = [], [], [], [], []
    for p in csvs:
        sc, ru, it = parse_name(p)
        r = read_rows(p)
        rows += r
        labels += [0 if sc == "normal" else 1] * len(r)
        scen += [sc] * len(r)
        run += [ru] * len(r)
        inten += [it] * len(r)

    ts_all = np.array([x["ts"] for x in rows])
    normal_vals = []
    for x, l in zip(rows, labels):
        if l == 0:
            v = decode_payload(x["payload"])
            if v: normal_vals.append(v)
    stats = learn_semantic_stats(normal_vals)
    print(f"messages={len(rows)}  normal_payloads={len(normal_vals)}  runs={max(run)}")

    feats, sem, toks, ok = build(rows, stats)
    labels = np.array(labels); scen = np.array(scen); run = np.array(run)
    mask = ok > 0
    feats, sem, toks, labels, scen, run, ts_all = (feats[mask], sem[mask], toks[mask],
                                                   labels[mask], scen[mask], run[mask], ts_all[mask])
    print(f"decoded={len(labels)}  positive={labels.mean():.3f}")

    if args.split == "temporal":
        # per-run temporal holdout: within each (scenario, run) stream, test on the last 20% by timestamp
        stream_key = np.array([f"{scen[i]}_{run[i]}" for i in range(len(labels))])
        te_mask = np.zeros(len(labels), bool)
        for key in np.unique(stream_key):
            idxs = np.where(stream_key == key)[0]
            o = np.argsort(ts_all[idxs])
            hold = max(1, int(len(o) * 0.2))
            te_mask[idxs[o[-hold:]]] = True
        tr_i = np.where(~te_mask)[0]; te_i = np.where(te_mask)[0]
    elif args.split == "run":
        max_run = {s: max(int(run[i]) for i in range(len(run)) if scen[i] == s) for s in set(scen)}
        tr_i = np.array([i for i in range(len(labels)) if int(run[i]) != max_run[scen[i]]])
        te_i = np.array([i for i in range(len(labels)) if int(run[i]) == max_run[scen[i]]])
    else:
        tr_i, te_i = train_test_split(np.arange(len(labels)), test_size=0.3, random_state=SEED, stratify=labels)

    Xtr, st_tr, tk_tr, ytr = feats[tr_i], sem[tr_i], toks[tr_i], labels[tr_i]
    Xte, st_te, tk_te, yte = feats[te_i], sem[te_i], toks[te_i], labels[te_i]
    scen_te, run_te = scen[te_i], run[te_i]

    print(f"\n===== SPLIT = {args.split} =====")
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED).fit(Xtr, ytr)

    def rep_baseline(name, score):
        if np.unique(yte).size > 1:
            thr = best_threshold(yte, score); p = (score >= thr).astype(int)
            print(f"  {name:14s} P={precision_score(yte, p):.3f} R={recall_score(yte, p):.3f} "
                  f"F1={f1_score(yte, p):.3f}  ROC={roc_auc_score(yte, score):.3f} "
                  f"PR={average_precision_score(yte, score):.3f}  MCC={matthews_corrcoef(yte, p):.3f}")

    print("--- behavioral / semantic ---")
    rep_baseline("GBM", gbm.predict_proba(Xte)[:, 1])
    rep_baseline("RandomForest", rf.predict_proba(Xte)[:, 1])
    # semantic calibration -> low-false-alarm specialist (1% of normal flagged)
    normal_sem = st_tr[ytr == 0]
    thr_s = float(np.percentile(normal_sem, 99)) if len(normal_sem) > 0 else 0.0
    ss = st_te
    if np.unique(yte).size > 1:
        ps = (ss >= thr_s).astype(int)
        print(f"  Semantic      P={precision_score(yte, ps):.3f} R={recall_score(yte, ps):.3f} "
              f"F1={f1_score(yte, ps):.3f}")
    thr_b = best_threshold(ytr, gbm.predict_proba(Xtr)[:, 1])
    fused = ((gbm.predict_proba(Xte)[:, 1] >= thr_b) | (ss >= thr_s)).astype(int)
    if np.unique(yte).size > 1:
        print(f"  Fused         P={precision_score(yte, fused):.3f} R={recall_score(yte, fused):.3f} "
              f"F1={f1_score(yte, fused):.3f}")

    deep_scores = {}
    if not args.no_deep:
        print("--- deep sequence ---")
        for name, model in [("GRU", GRUCls()), ("TCN", TCN())]:
            sc = train_torch(model, tk_tr, ytr, tk_te, yte, args.epochs)
            deep_scores[name] = sc
            rep_baseline(name, sc)

    print("\n--- per-scenario recall (FUS / GBM / Sem / GRU / TCN) ---")
    gbm_te = gbm.predict_proba(Xte)[:, 1]
    for sname in sorted(set(scen_te)):
        m = (scen_te == sname); sy = yte[m]
        def rec(pred, y):
            np_ = int((y == 1).sum()); return int(((y == 1) & (pred == 1)).sum()) / np_ if np_ else float('nan')
        txt = f"  {sname:18s} n={int(m.sum()):5d}  FUS={rec(fused[m], sy):.3f}  "
        txt += f"GBM={rec((gbm_te[m]>=thr_b).astype(int), sy):.3f}  Sem={rec((ss[m]>=thr_s).astype(int), sy):.3f}"
        for nm, sc in deep_scores.items():
            s_thr = best_threshold(sy, sc[m]) if np.unique(sy).size > 1 else 0.5
            txt += f"  {nm}={rec((sc[m]>=s_thr).astype(int), sy):.3f}"
        print(txt)

    if args.diagnose:
        print("\n===== DIAGNOSE: timing-only vs byte-only vs full (GBM, GLOBAL train threshold) =====")
        splits = [("timing-only", [3, 5]),   # dt, burst state (pure timing)
                  ("byte-only", [0, 1, 2]),  # entropy, mean, std (pure payload)
                  ("protocol-only", [4, 6]), # length-delta, event id (SOME/IP state)
                  ("full", list(range(feats.shape[1])))]
        for name, idx in splits:
            g = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
            g.fit(Xtr[:, idx], ytr)
            tr_sc = g.predict_proba(Xtr[:, idx])[:, 1]
            thr = best_threshold(ytr, tr_sc)            # GLOBAL threshold calibrated on TRAIN
            sc = g.predict_proba(Xte[:, idx])[:, 1]
            if np.unique(yte).size > 1:
                print(f"  {name:12s} F1={f1_score(yte, (sc>=thr).astype(int)):.3f}  "
                      f"ROC={roc_auc_score(yte, sc):.3f}")
            print(f"  per-scenario recall (global thr):")
            for sname in sorted(set(scen_te)):
                m = (scen_te == sname); sy = yte[m]
                def rec2(pred, y):
                    np_ = int((y == 1).sum()); return int(((y == 1) & (pred == 1)).sum()) / np_ if np_ else float('nan')
                print(f"    {sname:16s} rec={rec2((sc[m]>=thr).astype(int), sy):.3f}")

    if args.fpr_eval:
        print("\n===== SIGNAL ISOLATION: fixed 1% FPR (TPR per attack, fair calibr.) =====")
        def tpr_at_fpr(train_score, y_tr, test_score, y_te, fpr=0.01):
            ns = train_score[y_tr == 0]
            thr = float(np.percentile(ns, (1 - fpr) * 100)) if len(ns) else 0.5
            pred = (test_score >= thr).astype(int)
            out = {}
            for sname in sorted(set(scen_te)):
                m = (scen_te == sname); sy = yte[m]; np_ = int((sy == 1).sum())
                out[sname] = int(((sy == 1) & (pred[m] == 1)).sum()) / np_ if np_ else float('nan')
            return out
        from torch.utils.data import DataLoader, TensorDataset
        def tcn_both():
            m = TCN().to(dev)
            xt = torch.tensor(tk_tr); yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
            dl = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True)
            crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)]).to(dev))
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for _ in range(15):
                m.train()
                for xb, yb in dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    loss = crit(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
            tr_s = predict_batches(m, tk_tr, dev)
            te_s = predict_batches(m, tk_te, dev)
            return 1 / (1 + np.exp(-tr_s)), 1 / (1 + np.exp(-te_s))
        header = "  " + "detector".ljust(12) + " | " + " | ".join(f"{s:9s}" for s in sorted(set(scen_te)))
        print(header)
        for name, idx in [("timing-only", [3, 5]), ("byte-only", [0, 1, 2]),
                          ("full", list(range(feats.shape[1])))]:
            g = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED).fit(Xtr[:, idx], ytr)
            t = tpr_at_fpr(g.predict_proba(Xtr[:, idx])[:, 1], ytr,
                           g.predict_proba(Xte[:, idx])[:, 1], yte)
            print(f"  {name:12s} | " + " | ".join(f"{t[s]:9.3f}" for s in sorted(set(scen_te))))
        tr_s, te_s = tcn_both()
        t = tpr_at_fpr(tr_s, ytr, te_s, yte)
        print(f"  {'TCN':12s} | " + " | ".join(f"{t[s]:9.3f}" for s in sorted(set(scen_te))))


if __name__ == "__main__":
    main()
