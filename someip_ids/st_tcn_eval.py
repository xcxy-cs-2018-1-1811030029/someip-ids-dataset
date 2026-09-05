#!/usr/bin/env python3
"""
st_tcn_eval.py -- spatio-temporal TCN: early fusion of byte tokens + timing channels.

Answers the reviewer architecture question: can a deep model ingest inter-arrival
timing at the INPUT (multi-channel tensor) and thereby close the TCN blind spots on
burst-timing attacks (DoS 0.085, drop 0.100, slowslow 0.518 at fixed 1% FPR)?

Design (early fusion, no hard logic, continuous score):
  per-token channels = byte token (learned embedding) || timing (log1p(dt_ms),
  log1p(|dl|), burst state) broadcast along the token axis, standardized on train;
  two-layer causal Conv1d over the concatenated channel dim, max-pool, linear head.

Protocol replicates the paper's Table 7 exactly so rows are directly comparable:
run-independent split, 15 epochs, pos-weighted BCE, batch 32, fixed 1% FPR global
threshold calibrated on the training set. The same protocol is applied to EVERY
detector (GBM variants, semantic, TCN, GRU-mean, ST-TCN), which also yields the
unified-threshold per-scenario recall table that fixes the Table 5 fairness issue.

Usage:
  python3 someip_ids/st_tcn_eval.py --gen vs_ip_gen/generate/ \
      --out vs_ip_gen/results/st_tcn_eval.txt
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class STTCN(nn.Module):
    """TCN with timing channels fused at the input (early fusion)."""

    def __init__(self, vocab=257, emb=48, hid=64, n_timing=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.tproj = nn.Linear(n_timing, 16)
        self.conv1 = nn.Conv1d(emb + 16, hid, 3, padding=1)
        self.conv2 = nn.Conv1d(hid, hid, 3, padding=1)
        self.linear = nn.Linear(hid, 1)

    def forward(self, x, t):
        e = self.emb(x)
        tt = self.tproj(t).unsqueeze(1).expand(-1, e.size(1), -1)
        z = torch.cat([e, tt], dim=2).transpose(1, 2)
        c = torch.relu(self.conv1(z))
        c = torch.relu(self.conv2(c))
        return self.linear(c.max(dim=2).values)


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


def timing_channels(feats, mu=None, sd=None):
    """feats cols: [ent, mean, std, dt, dl, burst, event] -> [N,3] timing tensor."""
    dt_ms = np.log1p(np.maximum(feats[:, 3], 0.0) * 1000.0)
    dl_l = np.log1p(np.abs(feats[:, 4]))
    burst = feats[:, 5]
    t = np.stack([dt_ms, dl_l, burst], axis=1).astype(np.float32)
    if mu is None:
        mu = t.mean(0)
        sd = t.std(0) + 1e-6
    return (t - mu) / sd, mu, sd


def train_torch(model_fn, tr_inputs, ytr, epochs=15, batch=32, lr=1e-3):
    """Pos-weighted BCE training, identical to benchmark fpr-eval tcn_both()."""
    from torch.utils.data import DataLoader, TensorDataset
    xts = [torch.tensor(a) for a in tr_inputs]
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(*xts, yt), batch_size=batch, shuffle=True)
    m = model_fn().to(dev)
    pos_w = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train()
        for *xb, yb in dl:
            xb = [x.to(dev) for x in xb]
            yb = yb.to(dev)
            loss = crit(m(*xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def score_model(m, te_inputs, batch=64):
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(te_inputs[0]), batch):
            chunk = [torch.tensor(a[i:i + batch]).to(dev) for a in te_inputs]
            outs.append(m(*chunk).squeeze(1).cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(outs)))


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


def time_packets_dist(fn, n, warmup=50):
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="vs_ip_gen/results/st_tcn_eval.txt")
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
    feats, sem, toks, labels, scen, run = (feats[mask], sem[mask], toks[mask],
                                           labels[mask], scen[mask], run[mask])

    # run-independent split (identical to benchmark --split run)
    max_run = {s: max(int(run[i]) for i in range(len(run)) if scen[i] == s)
               for s in set(scen)}
    tr_i = np.array([i for i in range(len(labels)) if int(run[i]) != max_run[scen[i]]])
    te_i = np.array([i for i in range(len(labels)) if int(run[i]) == max_run[scen[i]]])

    Xtr, Xte = feats[tr_i], feats[te_i]
    ytr, yte = labels[tr_i], labels[te_i]
    scen_te = scen[te_i]
    tk_tr, tk_te = toks[tr_i], toks[te_i]
    sem_tr, sem_te = sem[tr_i], sem[te_i]

    out = []

    def log(s=""):
        print(s)
        out.append(s)

    log(f"messages={len(labels)}  split=run  epochs={args.epochs}  protocol=Table7(1%FPR global thr)")
    log(f"train={len(tr_i)}  test={len(te_i)}  test_pos={yte.mean():.3f}")

    # timing channels (standardized on train)
    tm_tr, mu, sd = timing_channels(Xtr)
    tm_te, _, _ = timing_channels(Xte, mu, sd)
    log("timing channels: log1p(dt_ms), log1p(|dl|), burst (standardized on train)")

    # ---- classical baselines (same as benchmark fpr-eval) ----
    def gbm_row(name, cols):
        g = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
        g.fit(Xtr[:, cols], ytr)
        tr_s = g.predict_proba(Xtr[:, cols])[:, 1]
        te_s = g.predict_proba(Xte[:, cols])[:, 1]
        return tr_s, te_s

    scen_names = sorted(set(scen_te))
    tpr = {}
    for name, cols in [("timing-only", [3, 5]), ("byte-only", [0, 1, 2]),
                       ("full-GBM", list(range(feats.shape[1])))]:
        tr_s, te_s = gbm_row(name, cols)
        tpr[name] = tpr_at_fpr(tr_s, ytr, te_s, yte, scen_te)

    # semantic: its own fixed threshold (99th pct of train NORMAL scores, = 1% FPR)
    thr_s = float(np.percentile(sem_tr[ytr == 0], 99))
    pred_s = (sem_te >= thr_s).astype(int)
    tpr["semantic"] = {}
    for s in scen_names:
        m = (scen_te == s)
        sy = yte[m]
        np_ = int((sy == 1).sum())
        tpr["semantic"][s] = float(((sy == 1) & (pred_s[m] == 1)).sum() / np_) \
            if np_ else float("nan")

    # ---- deep models: TCN first so torch RNG consumption matches the paper run ----
    models = {"TCN": (B.TCN, [tk_tr], [tk_te])}
    models["GRU-mean"] = (GRUMean, [tk_tr], [tk_te])
    models["ST-TCN"] = (STTCN, [tk_tr, tm_tr], [tk_te, tm_te])
    deep = {}
    for name, (mk, tr_in, te_in) in models.items():
        m = train_torch(mk, tr_in, ytr, args.epochs)
        tr_s = score_model(m, tr_in)
        te_s = score_model(m, te_in)
        deep[name] = (m, tr_s, te_s)
        tpr[name] = tpr_at_fpr(tr_s, ytr, te_s, yte, scen_te)
        log(f"trained {name}  train_score_ok={len(tr_s)}")

    log("\nFPR_TPR detector      " + " ".join(f"{s:>9s}" for s in scen_names))
    for name in ["timing-only", "byte-only", "full-GBM", "semantic",
                 "TCN", "GRU-mean", "ST-TCN"]:
        log(f"FPR_TPR {name:12s} | " + " | ".join(f"{tpr[name][s]:9.3f}"
                                                 for s in scen_names))

    # overall ROC/PR for the deep models
    from sklearn.metrics import roc_auc_score, average_precision_score
    log("\noverall (run split, same protocol)")
    for name in ["TCN", "GRU-mean", "ST-TCN"]:
        te_s = deep[name][2]
        log(f"OVERALL {name:12s} ROC={roc_auc_score(yte, te_s):.3f}  "
            f"PR={average_precision_score(yte, te_s):.3f}")

    # ---- fair latency: CPU single-thread batch-1 (torch), then ONNX Runtime ----
    torch.set_num_threads(1)
    tok1 = tk_te[0:1].astype(np.int64)
    tm1 = tm_te[0:1].astype(np.float32)
    n = 2000
    log("\n===== CPU batch-1 latency (single thread, n=2000) =====")
    for name in ["TCN", "GRU-mean", "ST-TCN"]:
        m = deep[name][0].to("cpu").eval()
        ins = (tok1, tm1) if name == "ST-TCN" else (tok1,)

        def fn():
            with torch.inference_mode():
                m(*[torch.from_numpy(x) for x in ins])

        med, p95 = time_packets_dist(fn, n)
        log(f"LATENCY_TORCH {name:12s} med={med:7.1f} us  p95={p95:7.1f} us  "
            f"({1e6/med:,.0f} pkt/s)")

    try:
        import onnx
        import onnxruntime as ort
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        dump = os.path.join(os.path.dirname(os.path.abspath(args.out)), "onnx_tmp")
        os.makedirs(dump, exist_ok=True)
        sess = {}
        for name in ["TCN", "GRU-mean", "ST-TCN"]:
            m = deep[name][0].to("cpu").eval()
            path = os.path.join(dump, f"{name}.onnx")
            ins = (tok1, tm1) if name == "ST-TCN" else (tok1,)
            if name == "ST-TCN":
                torch.onnx.export(m, (torch.from_numpy(tok1), torch.from_numpy(tm1)),
                                  path, opset_version=17,
                                  input_names=["tokens", "timing"],
                                  output_names=["logit"])
            else:
                torch.onnx.export(m, (torch.from_numpy(tok1),), path,
                                  opset_version=17,
                                  input_names=["tokens"], output_names=["logit"])
            sess[name] = ort.InferenceSession(path,
                                              providers=["CPUExecutionProvider"])
        log("===== ONNX Runtime CPU batch-1 latency (single thread) =====")
        for name in ["TCN", "GRU-mean", "ST-TCN"]:
            s = sess[name]
            if name == "ST-TCN":
                inps = {"tokens": tok1, "timing": tm1}
            else:
                inps = {"tokens": tok1}

            def fn():
                s.run(None, inps)

            med, p95 = time_packets_dist(fn, n)
            log(f"LATENCY_ONNX {name:12s} med={med:7.1f} us  p95={p95:7.1f} us  "
                f"({1e6/med:,.0f} pkt/s)")
    except ImportError:
        log("(onnx/onnxruntime not installed -- skipped ONNX section)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
