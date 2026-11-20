#!/usr/bin/env bash
# Reruns of the two registered runs that did not start in the batch of 2026-11-19
# (V5 single: input not parsed; V7 largest step: refused at startup), with the
# lines of scripts/run_validation.sh as corrected.
set -uo pipefail
BUILDS="/Users/mancininilay/Desktop/Life/PhD/Research/P1.3 & P2/Entity Local Experiments/worktrees/builds/bhemv"; ROOT="/Users/mancininilay/Desktop/Life/PhD/Research/P1.3 & P2/Entity Local Experiments/p2-bh-emission-claude1-20261119/runs/final"; MK="/Users/mancininilay/Desktop/Life/PhD/Research/P1.3 & P2/Entity Local Experiments/worktrees/dev-bh-emission-validation/pgens/bh_validation/make_input.py"; J6=0.389958
COOL="fields=false neutralizer=false gauss=false recycle=true nx=32 ny=32 maxnpart_leptons=400000"
run() { local dir="$ROOT/$1" prec="$2"; shift 2; mkdir -p "$dir"; python3 "$MK" "$dir/input.toml" "$@" > "$dir/make_input.log"
  ( cd "$dir" && OMP_PROC_BIND=false OMP_NUM_THREADS=8 "$BUILDS/bhem-val-$prec/src/entity.xc" -input input.toml > run.log 2>&1 ); echo "$dir rc=$?"; }
run v5/g300_single single name=v5 order=ions_first $COOL n_ions=10000 gamma_p=300.0 gamma_max=0.0 eps0=0.05 dx=1.0 cfl=0.7071067811865476 tau=0.02827113042 steps=46000 diag_interval=23
for prec in single double; do
  pref=$(python3 -c "print(repr(0.05 / 1.00754 * 0.998 / 0.5))")
  run v7/p0.1_$prec $prec name=v7 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.02 J_ref=$J6 p_ref=$pref cfl=0.998 steps=20040 diag_interval=10
done
