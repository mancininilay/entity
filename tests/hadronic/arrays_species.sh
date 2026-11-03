#!/usr/bin/env bash
# D2 fix (2026-11-03; Codex 1's closing validation, D2): the array closure of bh_proto after MPI migration. Runs entity.xc on
# the periodic box inputs/mpi/arrays_restframe_periodic.toml (32 cells: valid for 4 domains) on each rank count, restframe
# spectrum: the event mode on the mono bath (Codex 1's failing case) and the budget and poisson sources on the power-law bath
# of step 4 (eps_min 0.1, eps_max 1.5, index 2: eps differs from pair to pair, and the budget's weights do too, so a wrong
# partner or a doubled photon cannot hide); a delta control whose partners co-move; and at one rank an absorbing box where
# a pair can lose one lepton. arrays_species_check.py then reads the run's complete per-species dumps.
#
# Usage: arrays_species.sh <entity.xc> <inputs/mpi dir> <workdir> <single|double> "<rank counts>" [mpiexec]
# Without mpiexec the rank list must be "1" and entity.xc runs directly. Exit: 0 every check passed, 1 a check failed
# (FAIL lines), 2 a run or a parse failed (infrastructure).
source "$(dirname "$0")/common.sh"
EXE="$1"; INP="$2"; WD="$3"; PREC="$4"; RANKS="$5"; MPIEXEC="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -n "$RANKS" ] || { echo "usage: arrays_species.sh <entity.xc> <inputs/mpi> <workdir> <single|double> \"<ranks>\" [mpiexec]"; exit 2; }
mkdir -p "$WD"
n_bad=0; n_infra=0

make_case() { # make_case <dir> <mode> <spectrum> <ranks> <box: periodic|absorbing> <bath: mono|powerlaw>: the case's input.toml
  local d="$1" mode="$2" spec="$3" r="$4" box="$5" bath="$6" perk=$((16 / $4)) t="100.0"
  rm -rf "$d"; mkdir -p "$d"
  [ "$box" = "absorbing" ] && { perk=256; t="30.0"; }
  awk -v r="$r" -v perk="$perk" -v mode="$mode" -v spec="$spec" -v box="$box" -v bath="$bath" -v t="$t" '
    /^ *runtime *=/       { print "  runtime = " t; next }
    /^ *number *=/        { print "    number        = " r; next }
    /^ *decomposition *=/ { print "    decomposition = [" r "]"; next }
    /^ *ions\.per_K *=/   { print "  ions.per_K          = " perk; next }
    /^ *spectrum *=/      { print "  spectrum            = \"" spec "\""; next }
    /^ *photon_field\.type *=/ { if (bath == "powerlaw") { print "  photon_field.type   = \"powerlaw\""; print "  photon_field.eps_min = 0.1"
                                                           print "  photon_field.eps_max = 1.5"; print "  photon_field.index  = 2.0" } else print; next }
    /^ *photon_field\.eps0 *=/ { if (bath != "powerlaw") print; next }
    /^ *fields *= *\[\["PERIODIC"\]\]/    { if (box == "absorbing") print "    fields    = [[\"MATCH\", \"MATCH\"]]"; else print; next }
    /^ *particles *= *\[\["PERIODIC"\]\]/ { if (box == "absorbing") { print "    particles = [[\"ABSORB\", \"ABSORB\"]]"; print "    [grid.boundaries.match]"; print "      ds = [[2.0]]" } else print; next }
    /^ *mode *=/ {
      if (mode == "event") print
      else { print "  mode                = \"continuous\""; print "  source              = \"" mode "\""
             if (mode == "budget") { print "  r_th_factor         = 1.0"; print "  f_r                 = 0.002"; if (box != "absorbing") print "  t_latency           = 0.0" } }
      next }
    /^ *pair_weight *=/ { if (mode != "budget") print; next }
    /^ *f_max *=/       { if (mode == "event") print; next }
    { print }' "$INP/arrays_restframe_periodic.toml" > "$d/input.toml"
}

run_case() { # run_case <name> <mode> <spectrum> <ranks> <box> <bath> <expected old-paired verdict: pass|fail|none>
  local name="$1" mode="$2" spec="$3" r="$4" box="$5" bath="$6" old="$7" d="$WD/$1" rc crc
  make_case "$d" "$mode" "$spec" "$r" "$box" "$bath"
  if [ -n "$MPIEXEC" ]; then ( cd "$d" && "$MPIEXEC" -n "$r" "$EXE" -input input.toml > run.log 2>&1 ); rc=$?
  else ( cd "$d" && "$EXE" -input input.toml > run.log 2>&1 ); rc=$?; fi
  if [ $rc -ne 0 ]; then echo "INFRA: $name: entity.xc exit $rc (log $d/run.log)"; n_infra=$((n_infra + 1)); return; fi
  python3 "$HERE/arrays_species_check.py" --dir "$d" --name "$name" --precision "$PREC" --spectrum "$spec" --ranks "$r" --box "$box" --old "$old"
  crc=$?
  if [ $crc -eq 1 ]; then n_bad=$((n_bad + 1)); elif [ $crc -ne 0 ]; then n_infra=$((n_infra + 1)); fi
}

for r in $RANKS; do
  # the old index-paired sum: exact at one rank, broken as soon as the two species migrate apart
  old=fail; [ "$r" = "1" ] && old=pass
  run_case "event_restframe_n$r" event restframe "$r" periodic mono "$old"
  run_case "budget_restframe_powerlaw_n$r" budget restframe "$r" periodic powerlaw "$old"
  run_case "poisson_restframe_powerlaw_n$r" poisson restframe "$r" periodic powerlaw "$old"
  # the delta control: partners with one momentum co-move, so the old sum closes on any rank count (why G3 T14 is unaffected)
  run_case "poisson_delta_n$r" poisson delta "$r" periodic mono pass
done
# one partner lost: the absorbing box at one rank, where slot k of the two species still is a pair (the ground truth)
run_case event_restframe_absorbing_n1 event restframe 1 absorbing mono none
run_case budget_restframe_powerlaw_absorbing_n1 budget restframe 1 absorbing powerlaw none

echo "arrays_species summary: $n_bad cases with failed checks, $n_infra infrastructure failures"
[ $n_infra -eq 0 ] || exit 2
[ $n_bad -eq 0 ] || exit 1
exit 0
