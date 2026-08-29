"""
pcap_to_records.py

Convert captured SOME/IP pcaps into the training arrays consumed by
someip_ids/dataset.py (View A features + View B payload tokens + labels).

Design:
  - Parse each pcap with scapy, keeping UDP packets (SOME/IP runs over UDP).
  - Group packets into flows by (srcIP, srcPort, dstIP, dstPort).
  - Per-packet View A features (reference the previous packet in the same flow):
      1) time-interval variation      dt  = t_i - t_{i-1}
      2) payload log-likelihood        sum log P_i(x_i)  (position model from benign)
      3) payload cross-entropy         -1/L sum log P_i(x_i)
      4) payload Hamming change        between consecutive payloads
      5) length change                 dL = L_i - L_{i-1}
    -> d_a = 5 features per packet.
  - View B: the UDP payload bytes (the SOME/IP message), mapped to tokens (byte+1, PAD=0),
    truncated/padded to LEN_B (default 128).
  - The position-wise byte model P is trained on the NORMAL pcaps ('normal' label).
  - Labels come from a manifest mapping each pcap file to 0 (normal) / 1 (attack), or from
    directory convention (normal/ -> 0, attack/ -> 1).

Usage (run in WSL where scapy is installed):
  python3 pcap_to_records.py --normal-dir pcaps/normal --attack-dir pcaps/attack \
      --out out/ --len-b 128
  python3 pcap_to_records.py --manifest manifest.json --out out/
"""

import argparse
import glob
import json
import os

import numpy as np

try:
    from scapy.all import rdpcap, IP, UDP
except Exception as e:  # pragma: no cover
    raise SystemExit("scapy is required:  pip install scapy")

LEN_B = 128          # View B token length
D_A = 5              # View A feature dim (per packet)
SOMEIP_HDR = 16      # SOME/IP fixed header length (bytes)


def _payload_model(normal_payloads, L, alpha=1.0):
    """Position-wise byte distribution from benign payloads (Laplace smoothing)."""
    counts = np.full((L, 256), 0.0)
    for p in normal_payloads:
        for i, b in enumerate(p[:L]):
            counts[i, int(b)] += 1.0
    counts += alpha
    return counts / counts.sum(axis=1, keepdims=True)


def _log_ll(payload, P):
    s = 0.0
    for i, b in enumerate(payload[:P.shape[0]]):
        p = float(P[i, int(b)])
        s += np.log(max(p, 1e-12))
    return s


def _hamming(a, b):
    n = min(len(a), len(b))
    return sum(bin(int(a[i]) ^ int(b[i])).count("1") for i in range(n))


def parse_pcap(path):
    """Return list of (ts, src, sp, dst, dp, payload bytes, ip_len) for UDP packets."""
    out = []
    for pkt in rdpcap(path):
        if IP in pkt and UDP in pkt:
            udp = pkt[UDP]
            payload = bytes(udp.payload)
            src = pkt[IP].src
            dst = pkt[IP].dst
            sp = int(udp.sport)
            dp = int(udp.dport)
            ts = float(pkt.time)
            ip_len = int(len(pkt))
            out.append((ts, src, sp, dst, dp, payload, ip_len))
    return out


def build_records(pcap_path, P, label, len_b=LEN_B):
    """Return (f_a[n,5], seq_b[n,len_b], labels[n]) for one pcap."""
    pkts = parse_pcap(pcap_path)
    # group into flows
    flows = {}
    for rec in pkts:
        ts, src, sp, dst, dp, payload, ip_len = rec
        key = (src, sp, dst, dp)
        flows.setdefault(key, []).append(rec)

    f_a, seq_b, labels = [], [], []
    for key, flow in flows.items():
        flow.sort(key=lambda r: r[0])  # by timestamp
        prev = None
        for rec in flow:
            ts, src, sp, dst, dp, payload, ip_len = rec
            # View A (per packet)
            feat = np.zeros(D_A, dtype=np.float32)
            if prev is not None:
                dt = ts - prev[0]
                dL = ip_len - prev[6]
                feat[0] = dt
                feat[4] = dL
                feat[3] = _hamming(prev[5], payload)
            feat[1] = _log_ll(payload, P)
            feat[2] = -_log_ll(payload, P) / max(min(len(payload), P.shape[0]), 1)
            # View B tokens
            toks = [int(b) + 1 for b in payload[:len_b]]
            if len(toks) < len_b:
                toks += [0] * (len_b - len(toks))
            f_a.append(feat)
            seq_b.append(np.array(toks, dtype=np.int64))
            labels.append(label)
            prev = rec

    if not f_a:
        return None
    return (np.stack(f_a), np.stack(seq_b), np.array(labels, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-dir", help="dir of normal pcaps (label 0)")
    ap.add_argument("--attack-dir", help="dir of attack pcaps (label 1)")
    ap.add_argument("--manifest", help="json: {pcap_path: 0/1}")
    ap.add_argument("--out", default="out")
    ap.add_argument("--len-b", type=int, default=LEN_B)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Collect (pcap, label) pairs
    pairs = []
    if args.manifest:
        pairs = [(os.path.abspath(p), int(l)) for p, l in json.load(open(args.manifest)).items()]
    else:
        if args.normal_dir:
            pairs += [(p, 0) for p in sorted(glob.glob(os.path.join(args.normal_dir, "*.pcap")))]
        if args.attack_dir:
            pairs += [(p, 1) for p in sorted(glob.glob(os.path.join(args.attack_dir, "*.pcap")))]
    if not pairs:
        raise SystemExit("no pcaps found -- provide --normal-dir/--attack-dir or --manifest")

    # Train position model on normal payloads
    normal_payloads = []
    for p, lab in pairs:
        if lab == 0:
            for rec in parse_pcap(p):
                normal_payloads.append(rec[5])
    if not normal_payloads:
        raise SystemExit("no normal pcaps to train position model")
    P = _payload_model(normal_payloads, LEN_B)
    print(f"position model trained on {len(normal_payloads)} benign payloads")

    # Build arrays
    all_f, all_s, all_l = [], [], []
    for p, lab in pairs:
        r = build_records(p, P, lab, len_b=args.len_b)
        if r is None:
            print(f"skip (no UDP packets): {p}")
            continue
        fa, sb, lb = r
        all_f.append(fa); all_s.append(sb); all_l.append(lb)
        print(f"{os.path.basename(p)}: {len(fa)} samples, label={lab}")

    fa = np.concatenate(all_f)
    sb = np.concatenate(all_s)
    lb = np.concatenate(all_l)
    np.save(os.path.join(args.out, "f_a.npy"), fa)
    np.save(os.path.join(args.out, "seq_b.npy"), sb)
    np.save(os.path.join(args.out, "labels.npy"), lb)
    print(f"\nsaved: {args.out}/f_a.npy [{fa.shape}] seq_b.npy [{sb.shape}] labels.npy [{lb.shape}]")
    print("positive ratio:", float(lb.mean()) if len(lb) else 0.0)


if __name__ == "__main__":
    main()
