"""
measure_jitter.py -- measure the REAL end-to-end jitter of this machine's physical
network path (NIC + campus Wi-Fi/AP + switches) via ICMP echo to the default gateway
and one external host. The resulting one-way delay distribution calibrates the
timestamp-perturbation study (jitter_robustness.py, level "measured").

Note: this host connects via Wi-Fi (SSID from netsh wlan show interfaces), so the
measured jitter is a wireless-path upper bound on the jitter of wired automotive
Ethernet switching -- a conservative stress test for timing features.

Usage:  python3 someip_ids/measure_jitter.py --count 1200 --interval 0.2
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

RTT_RE = re.compile(r"time=([\d.]+)\s*ms")


def ping_collect(target, count, interval, timeout=2.0):
    cmd = ["ping", "-c", str(count), "-i", str(interval), "-W", str(int(timeout)), target]
    rtts = []
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        m = RTT_RE.search(line)
        if m:
            rtts.append(float(m.group(1)))
    return np.array(rtts)


def summarize(name, rtts):
    if len(rtts) == 0:
        print(f"{name}: no samples")
        return None
    jitter = np.abs(np.diff(rtts))
    oneway = rtts / 2.0
    s = {
        "n": len(rtts),
        "rtt_med": np.median(rtts), "rtt_mean": rtts.mean(), "rtt_std": rtts.std(),
        "rtt_p95": np.percentile(rtts, 95), "rtt_p99": np.percentile(rtts, 99),
        "jit_med": np.median(jitter), "jit_p95": np.percentile(jitter, 95),
        "jit_p99": np.percentile(jitter, 99), "jit_max": jitter.max(),
        "ow_med": np.median(oneway), "ow_p95": np.percentile(oneway, 95),
    }
    print(f"{name}: n={s['n']}  RTT med={s['rtt_med']:.3f}ms std={s['rtt_std']:.3f}ms "
          f"p95={s['rtt_p95']:.3f}ms | jitter med={s['jit_med']:.3f}ms "
          f"p95={s['jit_p95']:.3f}ms p99={s['jit_p99']:.3f}ms | one-way med={s['ow_med']:.3f}ms")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--outdir", default="vs_ip_gen/results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    targets = [("gateway", "10.51.80.1"), ("external", "223.5.5.5")]
    for name, ip in targets:
        print(f"measuring {name} ({ip}) ...")
        rtts = ping_collect(ip, args.count, args.interval)
        np.savetxt(os.path.join(args.outdir, f"jitter_rtt_{name}.txt"), rtts,
                   header=f"ICMP RTT ms to {ip}")
        summarize(name, rtts)
    print(f"\nsamples saved -> {args.outdir}/jitter_rtt_*.txt")


if __name__ == "__main__":
    main()
