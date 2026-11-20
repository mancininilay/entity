#!/usr/bin/env bash
# The mutation matrix of the charge test: every mutant is built in single and in double precision,
# as the shipped unit test (the verdict: its exit status) and as its diagnostic copy (every
# measurement), and run at 1, 2 and 8 OpenMP threads. The two guard mutants are built as the shipped
# test only and run at 8 threads. `none` is the unmutated policy under the diagnostic copy.
#   run_matrix.sh <master folder>/Entity Local Experiments <this directory>
set -uo pipefail
export E="$1" N="$2"
export BR="$E/worktrees/builds/corr" WT="$E/worktrees/dev-bh-emission"
POLICY="no_write integer_only fractional_only electrons_only positrons_only x_only y_only"
GUARD="no_dt_guard no_mass_guard"
mkdir -p "$N/bin" "$N/logs"
build_one() {
  local m="$1" prec="$2" var="$3" src overlay
  if [[ $var == shipped ]]; then src="$WT/tests/archetypes/bethe_heitler.cpp"; else src="$N/diagnostic/bethe_heitler_diagnostic.cpp"; fi
  if [[ $m == none ]]; then overlay=none; else overlay="$N/overlays/$m"; fi
  python3 "$N/build_mutant.py" "$BR/pr-tests-$prec" "$overlay" "$src" "$N/bin/$m-$prec-$var" > "$N/logs/build-$m-$prec-$var.log" 2>&1
  echo "BUILD $m $prec $var rc=$?"
}
run_one() {
  local m="$1" prec="$2" var="$3" t="$4"
  local log="$N/logs/run-$m-$prec-$var-t$t.log"
  ( cd "$N/bin" && OMP_NUM_THREADS=$t OMP_PROC_BIND=false "./$m-$prec-$var" ) > "$log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$log"
  echo "RUN $m $prec $var threads=$t rc=$rc"
}
export -f build_one run_one
{
  for prec in single double; do
    for m in $POLICY; do echo "$m $prec shipped"; echo "$m $prec diagnostic"; done
    for m in $GUARD; do echo "$m $prec shipped"; done
    echo "none $prec diagnostic"
  done
} > "$N/logs/builds.txt"
xargs -P 5 -L 1 bash -c 'build_one "$0" "$1" "$2"' < "$N/logs/builds.txt"
{
  for prec in single double; do
    for m in $POLICY; do for var in shipped diagnostic; do for t in 8 2 1; do echo "$m $prec $var $t"; done; done; done
    for m in $GUARD; do echo "$m $prec shipped 8"; done
    for t in 8 2 1; do echo "none $prec diagnostic $t"; done
  done
} > "$N/logs/runs.txt"
xargs -P 4 -L 1 bash -c 'run_one "$0" "$1" "$2" "$3"' < "$N/logs/runs.txt"
echo "MATRIX DONE $(date +%H:%M:%S)"
