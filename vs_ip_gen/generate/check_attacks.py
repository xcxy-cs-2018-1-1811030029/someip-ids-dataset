#!/usr/bin/env python3
"""
check_attacks.py -- VERIFY attack signatures reach the client CSVs.

Each attack has a DIFFERENT signature, so we check the right one per family:
  tamper      -> signal-1 (speed) changes its VALUE (normal max ~150)
  fuzz        -> payload LENGTH changes / byte-entropy drops (short random bytes)
  dos         -> packet RATE is far above normal (flood)
  drop        -> packet RATE far below normal / large inter-arrival gaps
  slowslow    -> very low rate (few messages over the run)
  ctx_tamper  -> CROSS-SIGNAL correlation is destroyed (each signal stays in range)

Run:  python3 check_attacks.py [dir]
"""
import glob, os, re, sys
import numpy as np

GEN = sys.argv[1] if len(sys.argv) > 1 else "."

def parse_name(path):
    base = os.path.basename(path).replace("_cli.csv", "")
    base = re.sub(r"_run\d+$", "", base)
    m = re.search(r"_i([\d.]+)$", base)
    intensity = float(m.group(1)) if m else 1.0
    scenario = re.sub(r"_i[\d.]+$", "", base)
    return scenario, intensity

def ent(b):
    if not b: return 0.0
    a = np.frombuffer(b, dtype=np.uint8)
    c = np.bincount(a, minlength=256).astype(float) / len(a)
    c = c[c > 0]
    return float(-(c * np.log(c)).sum())

per = {}
for path in sorted(glob.glob(os.path.join(GEN, "*_cli.csv"))):
    sc, it = parse_name(path)
    ts_l, s1_l, s2_l, lens, ents = [], [], [], [], []
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",", 4)
            b = bytes(int(x, 16) for x in p[-1].split()) if p[-1].split() else b""
            ts_l.append(float(p[0])); lens.append(len(b)); ents.append(ent(b))
            if len(b) >= 12 and len(b) % 4 == 0:
                v = np.frombuffer(b, dtype=np.float32)[:3]
                s1_l.append(float(v[0])); s2_l.append(float(v[1]))
    per[(sc, it)] = dict(ts=np.array(ts_l), lens=np.array(lens), ent=np.array(ents),
                         s1=np.array(s1_l), s2=np.array(s2_l))

# normal reference
ref = per[("normal", 1.0)]
ts = ref["ts"]; dur = float(ts.max() - ts.min()) if ts.size > 1 else 1.0
ref_rate = ref["ts"].size / max(dur, 1e-6)
ref_corr = float(np.corrcoef(ref["s1"], ref["s2"])[0, 1]) if ref["s1"].size > 3 else 0.0
ref_ent = float(ref["ent"].mean())
print(f"NORMAL REF: rate={ref_rate:.1f} msg/s  corr(s1,s2)={ref_corr:.3f}  ent={ref_ent:.3f}  max_s1={ref['s1'].max():.1f}\n")

def dt_stats(path):
    t = per[key]["ts"]
    t = np.sort(t)
    d = np.diff(t)
    return (float(np.median(d)) if d.size else float('nan'),
            float(d.max()) if d.size else float('nan'))

print(f"{'scenario':12s} {'int':5s} {'rate':8s} {'med_dt':8s} {'max_s1':8s} {'<12':5s} {'ent_d':7s} {'corr':7s}  verdict")
for key in sorted(per):
    d = per[key]
    if key[0] == "normal": continue
    t = d["ts"]; dur = float(t.max() - t.min()) if t.size > 1 else 1.0
    rate = d["ts"].size / max(dur, 1e-6)
    med_dt, max_gap = dt_stats(None) if False else (float('nan'), float('nan'))
    tt = np.sort(t)
    dd = np.diff(tt); med_dt = float(np.median(dd)) if dd.size else float('nan')
    max_gap = float(dd.max()) if dd.size else float('nan')
    s1max = float(d["s1"].max()) if d["s1"].size else 0.0
    frac_short = float((d["lens"] < 12).mean()) if d["lens"].size else 0.0
    ent_d = float(d["ent"].mean() - ref_ent) if d["ent"].size else 0.0
    corr = float(np.corrcoef(d["s1"], d["s2"])[0, 1]) if d["s1"].size > 3 else float('nan')

    sc = key[0]
    if sc == "tamper":   verdict = "yes" if s1max > 165 else "NO"
    elif sc == "fuzz":   verdict = "yes" if frac_short > 0.05 or abs(ent_d) > 0.5 else "NO"
    elif sc == "dos":    verdict = "yes" if rate > 2 * ref_rate else "NO"
    elif sc == "drop":   verdict = "yes" if rate < 0.5 * ref_rate else "NO"
    elif sc == "slowslow": verdict = "yes" if rate < 0.5 * ref_rate else "NO"
    elif sc == "ctx_tamper": verdict = "yes" if (not np.isnan(corr)) and abs(corr - ref_corr) > 0.15 else "NO"
    else: verdict = "?"

    print(f"{sc:12s} {key[1]:5.1f} {rate:8.2f} {med_dt:8.4f} {s1max:8.2f} {frac_short:5.2f} {ent_d:+7.3f} "
          f"{corr if not np.isnan(corr) else float('nan'):7.3f}  {verdict}")
