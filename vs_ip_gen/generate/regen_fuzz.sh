#!/usr/bin/env bash
# regen_fuzz.sh -- regenerate ONLY the fuzz scenario at the BENIGN cadence (10 ms, matching normal)
# so that fuzz becomes a pure payload anomaly with NO timing signature. Keeps all other scenarios.
# Run in WSL from this directory. ~5 min (30 s x 9 runs).
set -u
BOOST="$HOME/boost_1_83_0"
export LD_LIBRARY_PATH=/usr/local/lib:"$BOOST/lib":${LD_LIBRARY_PATH:-}
CONFIG="$HOME/vsomeip/config/vsomeip-local.json"
cd "$(dirname "$0")"

pkill -9 -f signal_service 2>/dev/null || true
pkill -9 -f signal_client  2>/dev/null || true
sleep 2
echo ">> stale procs: $(pgrep -af 'signal_service|signal_client' | wc -l)"

# remove old fuzz CSVs (leave every other scenario intact)
rm -f ./fuzz_i*.csv ./fuzz_i*.log
echo "removed old fuzz CSVs"

for I in 0.3 0.7 1.0; do
  for R in 1 2 3; do
    TAG="fuzz_i${I}_run${R}"
    echo ">> fuzz intensity=$I run=$R (cycle=10ms, 30s, pure payload) ..."
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=service-sample \
        ./signal_service --attack fuzz --intensity "$I" --cycle 10 --out "${TAG}_svc.csv" > "${TAG}_svc.log" 2>&1 &
    SVC=$!; sleep 1.5
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=client-sample \
        ./signal_client --out "${TAG}_cli.csv" > "${TAG}_cli.log" 2>&1 &
    CLI=$!; sleep 30
    kill -9 "$CLI" "$SVC" 2>/dev/null || true; sleep 0.5
    pkill -9 -f signal_service 2>/dev/null || true
    pkill -9 -f signal_client 2>/dev/null || true
    echo "  ${TAG}_cli.csv: $(wc -l < "${TAG}_cli.csv" 2>/dev/null || echo 0) lines"
  done
done
echo
echo "done. Verify + benchmark:"
echo "  python3 check_attacks.py ."
echo "  bash run_all.sh"
