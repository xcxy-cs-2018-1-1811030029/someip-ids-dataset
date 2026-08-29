"""
csv_to_records.py

Convert the SOME/IP CSV logs produced by signal_client (or signal_service) into
the training arrays consumed by someip_ids/dataset.py
(View A features + View B payload tokens + labels).

CSV columns (from signal_client): ts,event,session,payload_len,payload_hex
CSV columns (from signal_service): ts,event,payload_len,payload_hex

Usage:
  python3 csv_to_records.py --manifest manifest.json --out out/ --len-b 128
  # manifest.json: {"scenario_cli.csv": 0, "dos_cli.csv": 1, ...}
  # label 0 = normal, 1 = attack
"""

import argparse
import json
import os
import numpy as np

D_A = 5
LEN_B = 128


def parse_hex(hs):
    hs = hs.strip()
    if not hs:
        return b""
    return bytes(int(x, 16) for x in hs.split())


def _payload_model(normal_payloads, L, alpha=1.0):
    counts = np.full((L, 256), 0.0)
    for p in normal_payloads:
        for i, b in enumerate(p[:L]):
            counts[i, b] += 1.0
    counts += alpha
    return counts / counts.sum(axis=1, keepdims=True)


def _log_ll(payload, P):
    s = 0.0
    for i, b in enumerate(payload[:P.shape[0]]):
        p = float(P[i, b])
        s += np.log(max(p, 1e-12))
    return s


def _hamming(a, b):
    n = min(len(a), len(b))
    return sum(bin(a[i] ^ b[i]).count("1") for i in range(n))


def read_csv(path):
    """Return list of dicts {ts,event,session,payload(bytes),length} from CSV."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        # columns may be: ts,event,session,payload_len,payload_hex  OR ts,event,payload_len,payload_hex
        for line in f:
            parts = line.rstrip("\n").split(",", 4)

            def get(col):
                try:
                    return header.index(col)
                except ValueError:
                    return -1
            i_ts = get("ts"); i_ev = get("event"); i_ses = get("session")
            i_plen = get("payload_len"); i_hex = get("payload_hex")
            # Fallback: if header didn't parse, assume fixed layout.
            if i_ts < 0:
                ts = float(parts[0])
                ev = parts[1]
                plen = int(parts[-2])
                hexs = parts[-1]
                session = parts[2] if len(parts) > 3 else "0"
                rows.append(dict(ts=ts, event=ev, session=session, plen=plen, payload=parse_hex(hexs)))
                continue
            ts = float(parts[i_ts])
            ev = parts[i_ev]
            session = parts[i_ses] if i_ses >= 0 and len(parts) > i_ses else "0"
            plen = int(parts[i_plen]) if i_plen >= 0 else 0
            hexs = parts[i_hex] if i_hex >= 0 else parts[-1]
            rows.append(dict(ts=ts, event=ev, session=session, plen=plen, payload=parse_hex(hexs)))
    return rows


def build_rows(path, P, label, len_b):
    rows = read_csv(path)
    # group by event (each event is a stream)
    groups = {}
    for r in rows:
        groups.setdefault(r["event"], []).append(r)
    f_a, seq_b, labels = [], [], []
    for ev, grp in groups.items():
        grp.sort(key=lambda r: r["ts"])
        prev = None
        for r in grp:
            feat = np.zeros(D_A, dtype=np.float32)
            if prev is not None:
                feat[0] = r["ts"] - prev["ts"]
                feat[3] = _hamming(prev["payload"], r["payload"])
                feat[4] = r["plen"] - prev["plen"]
            feat[1] = _log_ll(r["payload"], P)
            feat[2] = -_log_ll(r["payload"], P) / max(min(len(r["payload"]), P.shape[0]), 1)
            toks = [b + 1 for b in r["payload"][:len_b]]
            if len(toks) < len_b:
                toks += [0] * (len_b - len(toks))
            f_a.append(feat)
            seq_b.append(np.array(toks, dtype=np.int64))
            labels.append(label)
            prev = r
    if not f_a:
        return None
    return np.stack(f_a), np.stack(seq_b), np.array(labels, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="json map csv->label")
    ap.add_argument("--out", default="out")
    ap.add_argument("--len-b", type=int, default=LEN_B)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = [(os.path.abspath(c), int(l)) for c, l in json.load(open(args.manifest)).items()]

    normal_payloads = []
    for p, lab in pairs:
        if lab == 0:
            for r in read_csv(p):
                normal_payloads.append(r["payload"])
    if not normal_payloads:
        raise SystemExit("no normal CSV to train position model")
    P = _payload_model(normal_payloads, LEN_B)
    print(f"position model trained on {len(normal_payloads)} benign payloads")

    all_f, all_s, all_l, all_sc = [], [], [], []
    for sc_id, (p, lab) in enumerate(pairs):
        r = build_rows(p, P, lab, len_b=args.len_b)
        if r is None:
            print(f"skip (no rows): {p}"); continue
        fa, sb, lb = r
        all_f.append(fa); all_s.append(sb); all_l.append(lb)
        all_sc.append(np.full(len(fa), sc_id, dtype=np.int64))
        print(f"{os.path.basename(p)}: {len(fa)} samples, label={lab}, scenario={sc_id}")

    fa = np.concatenate(all_f); sb = np.concatenate(all_s); lb = np.concatenate(all_l)
    sc = np.concatenate(all_sc)
    np.save(os.path.join(args.out, "f_a.npy"), fa)
    np.save(os.path.join(args.out, "seq_b.npy"), sb)
    np.save(os.path.join(args.out, "labels.npy"), lb)
    np.save(os.path.join(args.out, "scenario.npy"), sc)
    print(f"\nsaved: {args.out}/f_a.npy [{fa.shape}] seq_b.npy [{sb.shape}] labels.npy [{lb.shape}] scenario.npy [{sc.shape}]")
    print("positive ratio:", float(lb.mean()) if len(lb) else 0.0)


if __name__ == "__main__":
    main()
