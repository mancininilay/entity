#!/usr/bin/env bash
# The five checks of the shakedown on a pair of runs.
#   run_checks.sh <python with numpy> <bp2raw> <cache dir> <run without> <run with> <j_table.txt> <output dir>
# Each check writes <output dir>/checkN.txt; the exit codes are collected in summary.txt
# (0 pass, 1 fail, 2 no verdict). Nothing is retried.
set -uo pipefail
PY="$1"; BP2RAW="$2"; CACHE="$3"; OFF="$4"; LOW="$5"; JTAB="$6"; OUT="$7"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" "$CACHE"
: > "$OUT/summary.txt"
run() { # name script args...
  local name="$1"; shift
  "$PY" "$HERE/$@" > "$OUT/$name.txt" 2>&1
  echo "$name exit $?" >> "$OUT/summary.txt"
}
run check1 check1_counters.py "$BP2RAW" "$CACHE" "$LOW" "$OFF"
run check2 check2_paircount.py "$BP2RAW" "$CACHE" "$LOW" "$JTAB"
run check3 check3_compare.py "$BP2RAW" "$CACHE" "$OFF" "$LOW"
run check4 check4_gauss.py "$BP2RAW" "$CACHE" "$OFF" "$LOW"
"$PY" "$HERE/check5_maps.py" "$BP2RAW" "$CACHE" "$LOW" "$OUT" > "$OUT/check5.log" 2>&1
echo "check5 exit $?" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
