#!/usr/bin/env bash
# generate_noisy.sh
# Regenerate normal / tamper / ctx_tamper with Gaussian SENSOR NOISE on the signal
# manifold (breaks the deterministic velocity->acceleration relation the reviewers
# flagged). Output goes to ./noisy/ so the released noiseless corpus stays untouched.
# Usage: NOISE=2.0 RUNS=3 bash generate_noisy.sh
set -u
BOOST="$HOME/boost_1_83_0"
export LD_LIBRARY_PATH=/usr/local/lib:"$BOOST/lib":${LD_LIBRARY_PATH:-}
CONFIG="$HOME/vsomeip/config/vsomeip-local.json"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="$HERE/noisy"
mkdir -p "$BASE"
cd "$BASE"

NOISE="${NOISE:-2.0}"
RUNS="${RUNS:-3}"

rm -f ./*_cli.csv ./*_svc.csv ./*.log
pkill -9 -f signal_service 2>/dev/null || true
pkill -9 -f signal_client 2>/dev/null || true
sleep 1

declare -A CYCLE=( [normal]=10 [tamper]=10 [ctx_tamper]=10 )
declare -A RUNTIME=( [normal]=120 [tamper]=120 [ctx_tamper]=120 )

for S in normal tamper ctx_tamper; do
  I=1.0
  for R in $(seq 1 "$RUNS"); do
    TAG="${S}_i${I}_run${R}"
    echo ">> scenario=$S noise=$NOISE run=$R (cycle=${CYCLE[$S]}ms, ${RUNTIME[$S]}s) ..."
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=service-sample \
        "$HERE/signal_service" --attack "$S" --intensity "$I" --noise "$NOISE" \
        --cycle "${CYCLE[$S]}" --out "${TAG}_svc.csv" > "${TAG}_svc.log" 2>&1 &
    SVC=$!
    sleep 1.5
    env VSOMEIP_CONFIGURATION="$CONFIG" VSOMEIP_APPLICATION_NAME=client-sample \
        "$HERE/signal_client" --out "${TAG}_cli.csv" > "${TAG}_cli.log" 2>&1 &
    CLI=$!
    sleep "${RUNTIME[$S]}"
    kill -9 "$CLI" "$SVC" 2>/dev/null || true
    sleep 0.5
    echo "  ${TAG}_cli.csv: $(wc -l < "${TAG}_cli.csv" 2>/dev/null || echo 0) lines"
    pkill -9 -f signal_service 2>/dev/null || true
    pkill -9 -f signal_client 2>/dev/null || true
  done
done

echo
echo "done. files in $BASE"
ls "$BASE"/*_cli.csv | wc -l
