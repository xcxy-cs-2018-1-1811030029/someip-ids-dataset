#!/usr/bin/env bash
# run_all.sh -- run the corrected benchmark across all three splits and save outputs.
# Run this AFTER generate_data.sh has produced the *cli.csv set.
# Each split runs GBM/RF/Semantic/Fused + GRU/TCN (30 epochs). Outputs land in
# /mnt/d/mlstart/someIP/vs_ip_gen/results/{run,random,temporal}.txt
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/someip_ids/benchmark.py"
GEN="$HERE"
OUT="$ROOT/vs_ip_gen/results"
mkdir -p "$OUT"

echo "==> using benchmark: $PY"
echo "==> generator dir  : $GEN"
echo "==> outputs        : $OUT"

for s in run random temporal; do
  echo
  echo "################ SPLIT = $s ################"
  python3 "$PY" --gen "$GEN" --split "$s" --epochs 30 --fpr-eval > "$OUT/${s}.txt" 2>&1
  echo "---- saved $OUT/${s}.txt ----"
  grep -A40 "SPLIT" "$OUT/${s}.txt" | head -60
done

echo
echo "ALL DONE. Results in: $OUT/{run,random,temporal}.txt"
echo "Quick sanity check (run split):"
grep -E "GBM|TCN" "$OUT/run.txt" | head -5
