#!/usr/bin/env bash
# The runs of M1 to M3 on the Mac (MPICH, OpenMP), for the local check of the
# generator and of the analysis scripts before the GPU runs. Not the registered runs.
#   run_mac.sh <builds dir with val-mpi-{single,double}> <mpiexec> <output root> <groups: m1 m2 m3> [precisions]
set -uo pipefail
BUILDS="$1"; MPIEXEC="$2"; ROOT="$3"; GROUPS_="$4"; PRECS="${5:-single double}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MK="$HERE/../make_input.py"
run() { # group name prec ranks decomposition key=value...
  local grp="$1" name="$2" prec="$3" ranks="$4" dec="$5"; shift 5
  local dir="$ROOT/$grp/$prec/r$ranks/$name"
  mkdir -p "$dir"
  python3 "$MK" "$dir/input.toml" name="$name" decomposition="$dec" "$@" > "$dir/make_input.log" || { echo "INPUT FAILED $grp/$prec/r$ranks/$name"; return 0; }
  local rc=0
  ( cd "$dir" && OMP_PROC_BIND=false OMP_NUM_THREADS=$((8 / ranks)) "$MPIEXEC" -n "$ranks" "$BUILDS/val-mpi-$prec/src/entity.xc" -input input.toml > run.log 2>&1 ) || rc=$?
  echo "$rc" > "$dir/rc"
  echo "$(date +%H:%M:%S) $grp $prec r$ranks $name rc=$rc"
}
source "$HERE/cases.sh"
for prec in $PRECS; do
  for g in $GROUPS_; do "${g}_cases" "$prec"; done
done
