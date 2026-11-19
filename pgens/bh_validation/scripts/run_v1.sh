#!/usr/bin/env bash
# V1, Gauss's law: runs the matrix {single, double} x {leptons_first, ions_first}
# x {1, 2, 4 ranks} x {policy on, off}, the negative control (leptons first,
# previous-position write left out) and the stale-slot variants.
#   run_v1.sh <builds dir> <mpiexec> <output root> <make_input.py>
set -uo pipefail
BUILDS="$1"; MPIEXEC="$2"; ROOT="$3"; MK="$4"
mkdir -p "$ROOT"
run() { # name prec ranks decomposition key=value...
  local name="$1" prec="$2" ranks="$3" dec="$4"; shift 4
  local dir="$ROOT/$prec/r$ranks/$name"
  mkdir -p "$dir"
  python3 "$MK" "$dir/input.toml" name="$name" decomposition="$dec" "$@" > "$dir/make_input.log"
  ( cd "$dir" && OMP_PROC_BIND=false OMP_NUM_THREADS=$((8 / ranks)) \
      "$MPIEXEC" -n "$ranks" "$BUILDS/bhem-val-mpi-$prec/src/entity.xc" -input input.toml > run.log 2>&1 )
  echo "$prec r$ranks $name rc=$?"
}
for prec in single double; do
  for cfg in "1:" "2:2, 1" "4:2, 2"; do
    ranks="${cfg%%:*}"; dec="${cfg#*:}"
    run A_on  "$prec" "$ranks" "$dec" order=leptons_first policy=on
    run A_off "$prec" "$ranks" "$dec" order=leptons_first policy=off
    run B_on  "$prec" "$ranks" "$dec" order=ions_first policy=on
    run B_off "$prec" "$ranks" "$dec" order=ions_first policy=off
    run A_neg "$prec" "$ranks" "$dec" order=leptons_first policy=negative
    # unused lepton slots hold stale previous positions
    run A_on_stale  "$prec" "$ranks" "$dec" order=leptons_first policy=on stale_prev=true
    run B_on_stale  "$prec" "$ranks" "$dec" order=ions_first policy=on stale_prev=true
    run A_neg_stale "$prec" "$ranks" "$dec" order=leptons_first policy=negative stale_prev=true
  done
done
