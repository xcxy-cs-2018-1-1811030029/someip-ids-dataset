#!/usr/bin/env bash
# test_one.sh -- ISOLATED single-run test. Run this in WSL BEFORE any full generation.
# It kills every stale signal_service/signal_client, starts exactly ONE tamper service and
# ONE client, captures ~20s, then reports whether tamper bytes actually reached the client.
# If test_cli.csv shows max_s1 > 165 (or test_svc.csv is non-empty), the capture is healthy.
set -u
BOOST="$HOME/boost_1_83_0"
export LD_LIBRARY_PATH=/usr/local/lib:"$BOOST/lib":${LD_LIBRARY_PATH:-}
CONFIG="$HOME/vsomeip/config/vsomeip-local.json"
cd "$(dirname "$0")"

echo "==> killing any stale service/client ..."
pkill -9 -f signal_service 2>/dev/null; pkill -9 -f signal_client 2>/dev/null; sleep 2
echo "==> remaining: $(pgrep -af 'signal_service|signal_client' | wc -l) procs"
pgrep -af 'signal_service|signal_client' || echo "   (clean)"

echo "==> starting tamper service (cycle=10ms, intensity=1.0, 20s) ..."
./signal_service --attack tamper --intensity 1.0 --cycle 10 --out test_svc.csv > test_svc.log 2>&1 &
SVC=$!
sleep 2
echo "==> starting client ..."
./signal_client --out test_cli.csv > test_cli.log 2>&1 &
CLI=$!
sleep 20
kill -9 "$CLI" "$SVC" 2>/dev/null || true
sleep 1

echo
echo "service payload log : test_svc.csv  -> $(wc -l < test_svc.csv 2>/dev/null || echo 0) lines"
echo "client payload log  : test_cli.csv  -> $(wc -l < test_cli.csv 2>/dev/null || echo 0) lines"
echo "client stdout       : test_cli.log  -> $(grep -c 'Received Event' test_cli.log 2>/dev/null || echo 0) events"
echo
echo "==> VERIFY (decode test_cli.csv only). Expect max_s1 > 165 => tamper reached client. =="
python3 - <<'PY'
import numpy as np
vals=[]
with open("test_cli.csv") as f:
    f.readline()
    for line in f:
        p=line.rstrip("\n").split(",",4)
        b=bytes(int(x,16) for x in p[-1].split()) if p[-1].split() else b""
        if len(b)>=12: vals.append(float(np.frombuffer(b,dtype=np.float32)[0]))
if not vals:
    print("NO parsed rows -> client got nothing; capture broken"); raise SystemExit(1)
vals=np.array(vals)
ok = vals.max() > 165
print(f"test_cli.csv: n={len(vals)}  max_s1={vals.max():.2f}  min_s1={vals.min():.2f}")
print("RESULT:", "PASS - tamper reached the client" if ok else "FAIL - traffic looks normal, capture still broken")
PY
