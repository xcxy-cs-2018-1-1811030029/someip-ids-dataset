#!/usr/bin/env bash
# generate_data.sh
# Run the SOME/IP publish-subscribe generator for multiple SCENARIOS and MULTIPLE RUNS each,
# collect client CSV logs + a manifest mapping each CSV to a label/scenario/run id.
# Multi-run output enables run-independent / temporal split experiments.
# Run inside WSL from this directory (after build_samples.sh).
set -u
BOOST="$HOME/boost_1_83_0"
export LD_LIBRARY_PATH=/usr/local/lib:"$BOOST/lib":${LD_LIBRARY_PATH:-}
CONFIG="$HOME/vsomeip/config/vsomeip-local.json"
BASE="$(pwd)"
RUNS="${RUNS:-3}"            # runs per scenario (for run-independent splits)
DUR="${DUR:-30}"             # seconds per scenario-run (scale up for more data)

rm -f ./*_cli.csv ./*_svc.csv ./*.log

# IMPORTANT: a stale signal_service can keep offering 0x1234.0x5678 and silently
# serve NORMAL traffic to every new client, so all collected "attack" CSVs become
# normal traffic. Kill every leftover generator process before we start.
pkill -9 -f signal_service 2>/dev/null || true
pkill -9 -f signal_client 2>/dev/null || true
sleep 2
echo ">> stale procs remaining: $(pgrep -af 'signal_service|signal_client' | wc -l)"
pgrep -af 'signal_service|signal_client' || echo "   (clean)"

# scenario -> label
declare -A LABEL=( [normal]=0 [dos]=1 [fuzz]=1 [drop]=1 [slowslow]=1 [tamper]=1 [ctx_tamper]=1 )
# per-scenario cycle (ms) and runtime (s); smaller cycle + larger runtime = more messages
declare -A CYCLE=( [normal]=10 [dos]=1000 [fuzz]=10 [drop]=50 [slowslow]=1000 [tamper]=10 [ctx_tamper]=10 )
declare -A RUNTIME=( [normal]=120 [dos]=30 [fuzz]=90 [drop]=90 [slowslow]=90 [tamper]=120 [ctx_tamper]=120 )

SCENARIOS=(normal dos fuzz drop slowslow tamper ctx_tamper)

ENTRIES=()
INTENSITIES="${INTENSITIES:-1.0}"   # space-separated severity levels for attack scenarios (e.g. "0.3 0.7 1.0")

for S in "${SCENARIOS[@]}"; do
  if [ "$S" == "normal" ]; then INTS=("1.0"); else INTS=($INTENSITIES); fi
  for I in "${INTS[@]}"; do
  for R in $(seq 1 "$RUNS"); do
    TAG="${S}_i${I}_run${R}"
    echo ">> scenario=$S intensity=$I run=$R (label ${LABEL[$S]}, cycle=${CYCLE[$S]}ms, ${RUNTIME[$S]}s) ..."
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=service-sample \
        ./signal_service --attack "$S" --intensity "$I" --cycle "${CYCLE[$S]}" --out "${TAG}_svc.csv" > "${TAG}_svc.log" 2>&1 &
    SVC=$!
    sleep 1.5
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=client-sample \
        ./signal_client --out "${TAG}_cli.csv" > "${TAG}_cli.log" 2>&1 &
    CLI=$!
    sleep "${RUNTIME[$S]}"
    kill -9 "$CLI" "$SVC" 2>/dev/null || true
    sleep 0.5
    echo "  ${TAG}_cli.csv: $(wc -l < "${TAG}_cli.csv" 2>/dev/null || echo 0) lines"
    pkill -9 -f signal_service 2>/dev/null || true
    pkill -9 -f signal_client 2>/dev/null || true
    ENTRIES+=("  \"$BASE/${TAG}_cli.csv\": ${LABEL[$S]}")
  done
  done
done

{
  echo "{"
  for i in "${!ENTRIES[@]}"; do
    if [ "$i" -lt $(( ${#ENTRIES[@]} - 1 )) ]; then printf '%s,\n' "${ENTRIES[$i]}"; else printf '%s\n' "${ENTRIES[$i]}"; fi
  done
  echo "}"
} > manifest.json

echo
echo "manifest entries: ${#ENTRIES[@]}"
echo
echo "=============================================================="
echo "VERIFY ATTACKS PRESENT BEFORE CONVERTING (must NOT show 'NO (broken)'):"
echo "  python3 $BASE/check_attacks.py $BASE"
echo "=============================================================="
echo
echo "Convert: cd $BASE/../.. && python3 someip_ids/csv_to_records.py --manifest vs_ip_gen/generate/manifest.json --out data/ --len-b 128"
echo "Benchmark: cd $BASE/../.. && python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --epochs 30"
