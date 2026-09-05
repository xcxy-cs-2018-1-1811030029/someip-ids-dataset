"""
jitter_robustness.py

Single-machine substitute for a physical-switch re-run: perturb every client-side
message timestamp by a per-message network delay eps sampled from calibrated jitter
models, then re-evaluate the timing-sensitive detectors on the SAME fixed random
split. Answers: are timing features robust to real network jitter, and how far?

Levels:
  L0 baseline   : eps = 0 (loopback timestamps, as released)
  L1 switch100M : eps ~ U(5,12) us         (store-and-forward 64B @ 100BASE-T1 + light queue)
  L2 loaded1G   : eps = 50us + Exp(50us)   (moderate switch queueing)
  L3 contended  : eps = 500us + Exp(500us) (heavy BE contention, ms-scale bursts)
  L4 measured   : eps ~ measured campus-path one-way delay (Wi-Fi path upper bound,
                 see measure_jitter.py -- needs vs_ip_gen/results/jitter_rtt_gateway.txt)

Detectors: timing-only GBM ([dt, burst]), byte-only GBM (jitter-immune control),
full GBM (all features), semantic score (jitter-immune control, evaluated once).
Outputs ROC / PR / MCC per level, plus per-scenario recall (global train threshold)
for the timing-only and full detectors.

Usage:  python3 someip_ids/jitter_robustness.py --gen vs_ip_gen/generate/
"""

import argparse
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from someip_ids import benchmark as B
from someip_ids.semantics import decode_payload, learn_semantic_stats

SEED = 1234

TIMING_IDX = [3, 5]        # dt, burst state
BYTE_IDX = [0, 1, 2]       # entropy, byte mean, byte std


def build_with_ts(rows, ts_ov):
    """Replicate benchmark.build, but with overridden (perturbed) timestamps.
    Only dt and burst-state depend on ts; other features are identical."""
    feats = []
    prev = {}
    state = {}
    for r, ts in zip(rows, ts_ov):
        ev = r["event"]
        b = r["payload"]
        a = np.frombuffer(b, dtype=np.uint8) if len(b) else np.zeros(0, np.uint8)
        p = prev.get(ev)
        dt = (ts - p["ts"]) if p else 0.0
        dl = (r["plen"] - p["plen"]) if p else 0.0
        st = state.get(ev, 0.0)
        st = st + 1.0 if dt < 0.5 else 0.0
        state[ev] = st
        cnt = np.bincount(a, minlength=256).astype(float) / max(len(a), 1)
        cnt = cnt[cnt > 0]
        ent = float(-(cnt * np.log(cnt)).sum()) if len(cnt) else 0.0
        feats.append(np.array([ent,
                               float(a.mean()) if len(a) else 0.0,
                               float(a.std()) if len(a) else 0.0,
                               dt, dl, st,
                               float(ev % 1000) / 1000.0], dtype=np.float32))
        pr = dict(r); pr["ts"] = ts
        prev[ev] = pr
    return np.stack(feats)


def make_eps(level, n, rng, resdir):
    if level == "baseline":
        return np.zeros(n)
    if level == "switch100M":
        return rng.uniform(5e-6, 12e-6, n)
    if level == "loaded1G":
        return 50e-6 + rng.exponential(50e-6, n)
    if level == "contended":
        return 500e-6 + rng.exponential(500e-6, n)
    if level == "measured":
        p = os.path.join(resdir, "jitter_rtt_gateway.txt")
        if not os.path.exists(p):
            p = os.path.join(resdir, "jitter_rtt_external.txt")
        if not os.path.exists(p):
            raise SystemExit(f"no measured jitter file; run measure_jitter.py first ({p})")
        rtts = np.loadtxt(p) / 1000.0        # ms -> s, one-way ~ RTT/2
        return rng.choice(rtts / 2.0, size=n, replace=True)
    raise ValueError(level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="vs_ip_gen/generate/")
    ap.add_argument("--levels", default="baseline,switch100M,loaded1G,contended,measured")
    ap.add_argument("--out", default="vs_ip_gen/results/jitter_robustness.txt")
    args = ap.parse_args()
    resdir = os.path.dirname(args.out)

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
    feats0, sem, _, ok = B.build(rows, stats)
    labels = np.array(labels); scen = np.array(scen)
    mask = ok > 0
    feats0, sem, labels, scen = feats0[mask], sem[mask], labels[mask], scen[mask]
    ts_all = ts_all[mask]

    tr_i, te_i = train_test_split(np.arange(len(labels)), test_size=0.3,
                                  random_state=SEED, stratify=labels)
    yte = labels[te_i]; ytr = labels[tr_i]
    scen_te = scen[te_i]

    out = []
    def log(s=""):
        print(s)
        out.append(s)

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    header = (f"{'level':12s} {'eps_med':>9s} {'eps_std':>9s} | "
              f"{'timing ROC':>11s} {'timing PR':>10s} {'timing MCC':>11s} | "
              f"{'byte ROC':>9s} {'byte MCC':>9s} | "
              f"{'full ROC':>9s} {'full PR':>9s} {'full MCC':>9s}")
    log(f"messages={len(labels)}  positive={labels.mean():.3f}")
    log("Detectors: timing-only GBM[dt,burst] | byte-only GBM[ent,mean,std] | full GBM[all]")
    log(header)

    scen_names = sorted(set(scen_te))
    scen_rows = {lvl: [] for lvl in levels}

    for lvl in levels:
        rng = np.random.default_rng(SEED)
        eps = make_eps(lvl, len(rows), rng, resdir)          # per raw row
        eps = eps[mask]
        if lvl == "baseline":
            F = feats0
        else:
            ts_ov = ts_all + eps
            F = build_with_ts(rows, ts_ov)[mask]
            assert np.allclose(F[:, BYTE_IDX], feats0[:, BYTE_IDX]), "byte features must be identical"

        def gbm_run(idx):
            g = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=SEED)
            g.fit(F[tr_i][:, idx], ytr)
            tr_sc = g.predict_proba(F[tr_i][:, idx])[:, 1]
            thr = B.best_threshold(ytr, tr_sc)
            sc = g.predict_proba(F[te_i][:, idx])[:, 1]
            pred = (sc >= thr).astype(int)
            rec = {}
            for s in scen_names:
                m = (scen_te == s) & (yte == 1)
                rec[s] = float(pred[m].mean()) if m.sum() else float("nan")
            return sc, thr, rec

        t_sc, t_thr, t_rec = gbm_run(TIMING_IDX)
        b_sc, _, _ = gbm_run(BYTE_IDX)
        f_sc, f_thr, f_rec = gbm_run(list(range(F.shape[1])))

        def mets(sc):
            thr = B.best_threshold(yte, sc)
            p = (sc >= thr).astype(int)
            return (roc_auc_score(yte, sc), average_precision_score(yte, sc),
                    matthews_corrcoef(yte, p))

        troc, tpr, tmcc = mets(t_sc)
        broc, _, bmcc = mets(b_sc)
        froc, fpr, fmcc = mets(f_sc)
        log(f"{lvl:12s} {np.median(eps)*1e6:8.1f}u {eps.std()*1e6:8.1f}u | "
            f"{troc:11.3f} {tpr:10.3f} {tmcc:11.3f} | "
            f"{broc:9.3f} {bmcc:9.3f} | "
            f"{froc:9.3f} {fpr:9.3f} {fmcc:9.3f}")
        scen_rows[lvl] = (t_rec, f_rec)

    log("\n--- per-scenario recall at global train threshold (timing-only) ---")
    log(f"{'level':12s} " + " ".join(f"{s:>8s}" for s in scen_names))
    for lvl in levels:
        log(f"{lvl:12s} " + " ".join(f"{scen_rows[lvl][0][s]:8.3f}" for s in scen_names))
    log("\n--- per-scenario recall at global train threshold (full GBM) ---")
    log(f"{'level':12s} " + " ".join(f"{s:>8s}" for s in scen_names))
    for lvl in levels:
        log(f"{lvl:12s} " + " ".join(f"{scen_rows[lvl][1][s]:8.3f}" for s in scen_names))

    # semantic: jitter-immune reference (payload only)
    thr_s = float(np.percentile(sem[tr_i][ytr == 0], 99))
    ps = (sem[te_i] >= thr_s).astype(int)
    from sklearn.metrics import precision_score, recall_score
    log(f"\nsemantic (payload-only, jitter-immune): "
        f"P={precision_score(yte, ps):.3f} R={recall_score(yte, ps):.3f} "
        f"(unchanged across levels by construction)")

    os.makedirs(resdir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
