#!/usr/bin/env bash
# Verdicts of the registered validation runs.
#   analyze_final.sh <output root> <scripts dir> <data dir> <bhv_spectra> <fork logs dir>
set -uo pipefail
ROOT="$1"; S="$2"; DATA="$3"; TOOL="$4"; FORK="$5"
D2="$DATA/bhref-0.2.0/moments_mono05.txt"
for prec in single double; do
  lim=1e-5; [ $prec = double ] && lim=1e-10
  echo "######## $prec"
  echo "---- V2 mean cooling, V3 straggling"
  python3 "$S/analyze_cooling.py" "$ROOT/v2/g6_$prec" --targets 0.9,0.7,0.5 --d2 "$D2" --tag "V2/V3 Gamma_p0=6 $prec"
  python3 "$S/analyze_cooling.py" "$ROOT/v2/g3_$prec" --targets 0.9,0.7,0.65 --d2 "$D2" --tag "V2/V3 Gamma_p0=3 $prec"
  echo "---- V4 pair spectra"
  for order in ions_first leptons_first; do
    e=2; p=3; [ $order = leptons_first ] && { e=1; p=2; }
    "$TOOL" "$DATA" 0.5 3 "$ROOT/v4/g3_${order}_$prec/bhv_species_${e}_rank_0.bin" "$ROOT/v4/g3_${order}_$prec/bhv_species_${p}_rank_0.bin" \
                        10 "$ROOT/v4/g10_${order}_$prec/bhv_species_${e}_rank_0.bin" "$ROOT/v4/g10_${order}_$prec/bhv_species_${p}_rank_0.bin" > "$ROOT/v4/spectra_${order}_$prec.txt" 2>&1
    echo "[$order] $(grep -E '^(PASS|FAIL) V4|^\(history\)' "$ROOT/v4/spectra_${order}_$prec.txt" | tr '\n' ' ')"
    grep -E "^  Gamma_p|^(PASS|FAIL) Gamma_p|^Gamma_p" "$ROOT/v4/spectra_${order}_$prec.txt" | sed 's/^/    /'
  done
  echo "---- V5 poster test"
  python3 "$S/analyze_poster.py" "$ROOT/v5/g300_$prec" "$DATA/poster/fig3_analytic_data.csv" --tag "Gamma_p0=300 $prec"
  echo "---- V6 energy"
  python3 "$S/analyze_energy.py" "$ROOT/v2/g6_$prec" --limit $lim --tag "fields off Gamma_p0=6 $prec"
  python3 "$S/analyze_energy.py" "$ROOT/v2/g3_$prec" --limit $lim --tag "fields off Gamma_p0=3 $prec"
  for th in 0.05 0.2; do
    python3 "$S/analyze_energy.py" "$ROOT/v6/b_theta${th}_$prec" --theta $th --tag "magnetic field $prec"
  done
  echo "---- V7 time step"
  python3 "$S/analyze_cooling.py" "$ROOT/v7/p0.0125_$prec" --targets 0.9,0.7,0.5 --tag "V7: V2 at the smallest step, $prec"
  python3 "$S/analyze_timestep.py" "$ROOT/v7/p0.0125_$prec" "$ROOT/v7/p0.025_$prec" "$ROOT/v7/p0.05_$prec" "$ROOT/v7/p0.1_$prec" --p 0.0125,0.025,0.05,0.1
  echo "---- V8 consistency with the event stage of dev/hadronic 72427ff6"
  python3 "$S/compare_fork.py" "$ROOT/v2/g6_$prec" "$FORK/e10_$prec.log"
  echo "---- V9 species order"
  python3 "$S/compare_runs.py" "$ROOT/v2/g6_$prec" "$ROOT/v9/g6_leptons_first_$prec" --label "V9 cooling (ions first vs leptons first)"
  python3 "$S/compare_spectra.py" "$ROOT/v4/spectra_ions_first_$prec.txt" "$ROOT/v4/spectra_leptons_first_$prec.txt"
  echo "---- V12 MPI"
  for r in 2 4; do
    python3 "$S/compare_runs.py" "$ROOT/v12/r1_$prec" "$ROOT/v12/r${r}_$prec" --label "V12 1 rank vs $r ranks"
  done
  for r in 1 2 4; do
    python3 "$S/analyze_energy.py" "$ROOT/v12/r${r}_$prec" --limit $lim --tag "$r ranks $prec"
    python3 "$S/analyze_cooling.py" "$ROOT/v12/r${r}_$prec" --targets 0.9,0.7,0.5 --tag "V12: V2 on $r ranks, $prec" | grep -E "PASS|FAIL|=="
  done
  echo "---- V13 pair weight"
  for a in 2 4; do
    python3 "$S/analyze_cooling.py" "$ROOT/v13/a${a}_$prec" --targets 0.9,0.7,0.5 --d2 "$D2" --tag "V13 pair_weight=$a $prec"
  done
done
