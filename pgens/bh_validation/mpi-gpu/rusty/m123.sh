#!/usr/bin/env bash
# The runs of M1, M2 and M3 on the four A100s of one node, one GPU per rank, with
# the executables of the build marked usable; then the analysis scripts on the
# CPU cores of the job. Every run is attempted once under a time cap; nothing is
# retried and no parameter is changed. M3 comes last: it is expected to stop in
# the engine (PREREGISTRATION.md).
# First, the part of M0 that build job 7523472 could not do: the unit test in the
# tree with bounds-checked views, on one rank (that tree and val-single-bounds were
# stopped by the build driver's own time cap two objects short and finished on a
# login node). A bounds-checked tree that does not exist is reported and skipped.
# BHM_BUDGET is the time for the runs in seconds (default 2400); BHM_ROOT,
# BHM_GROUPS and M1_N_IONS exist only for the dry run.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHM_ROOT:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}"
bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
V="$R/src-val/pgens/bh_validation"
MK="$V/make_input.py"
ROOT="$R/runs/m123-$SLURM_JOB_ID"
mkdir -p "$R/runs" && mkdir "$ROOT" || exit 1
exec > "$ROOT/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$ROOT/exit-status"' EXIT
T0=$SECONDS
BUDGET="${BHM_BUDGET:-2400}"      # seconds for the runs; the analysis takes the rest
hostname; date -u
module list 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
grep Cpus_allowed_list /proc/self/status
want=$(cat "$R/COMMIT-val") || exit 1
[[ $(git -C "$R/src-val" rev-parse HEAD) == "$want" && -z $(git -C "$R/src-val" status --porcelain) ]] || { echo "src-val is not a clean $want"; exit 1; }
echo "source: $R/src-val at $want"; echo "build: $bld"
usable() { # tree executable
  [[ -f "$bld/$1/BUILD_OK" && -x "$bld/$1/$2" && $(sha256sum "$bld/$1/$2" | cut -d' ' -f1) == $(cat "$bld/$1/BUILD_OK") ]]
}
for t in val-single val-double; do
  usable "$t" src/entity.xc || { echo "$t is not the executable marked usable"; exit 1; }
  echo "$t sha256 $(cat "$bld/$t/BUILD_OK")"
done
bounds=0
if usable val-single-bounds src/entity.xc; then
  bounds=1; echo "val-single-bounds sha256 $(cat "$bld/val-single-bounds/BUILD_OK")"
else
  echo "NOT BUILT val-single-bounds: the bounds-checked runs of M3 are skipped"
fi
IFS=, read -ra devs <<< "${CUDA_VISIBLE_DEVICES:-}"
ngpu=${#devs[@]}
status=0
run() { # group name precision ranks decomposition key=value...
  local grp="$1" name="$2" prec="$3" ranks="$4" dec="$5"; shift 5
  local dir="$ROOT/$grp/$prec/r$ranks/$name" tree="val-$prec" cap=300
  [[ "$grp" == *-bounds ]] && tree="val-$prec-bounds"
  local rem=$(( BUDGET - (SECONDS - T0) ))
  if (( rem < 60 )); then
    echo "SKIPPED $grp/$prec/r$ranks/$name: no time left"; status=1; return 0
  fi
  (( cap > rem )) && cap=$rem
  if (( ngpu < ranks )) && [[ -z "${BHM_SHARE_GPU:-}" ]]; then
    echo "SKIPPED $grp/$prec/r$ranks/$name: $ngpu GPUs for $ranks ranks"; status=1; return 0
  fi
  mkdir -p "$dir"
  if ! python3 "$MK" "$dir/input.toml" name="$name" decomposition="$dec" "$@" > "$dir/make_input.log" 2>&1; then
    echo "INPUT FAILED $grp/$prec/r$ranks/$name"; status=1; return 0
  fi
  local t=$SECONDS rc=0
  ( cd "$dir" && BHM_GPU_LOG="$dir/gpu" timeout --kill-after=20 "$cap" mpirun --bind-to none --map-by "ppr:$ranks:node" -n "$ranks" \
      "$HERE/rank-gpu.sh" "$bld/$tree/src/entity.xc" -input input.toml > run.log 2>&1 ) || rc=$?
  echo "$rc" > "$dir/rc"
  echo "RUN $grp/$prec/r$ranks/$name rc=$rc seconds=$((SECONDS - t)) cap=$cap gpus=$(cat "$dir"/gpu.* 2>/dev/null | awk '{print $9}' | sort -u | wc -l)"
  [[ $rc == 0 || "$grp" == m3* ]] || status=1
}
echo "==== M0: the unit test ARCHETYPES::bethe_heitler with bounds-checked views, one rank"
T="tests/archetypes/test-archetypes-bethe_heitler.xc"
if usable tests-single-bounds "$T"; then
  echo "tests-single-bounds sha256 $(cat "$bld/tests-single-bounds/BUILD_OK")"
  d="$ROOT/m0/tests-single-bounds"; mkdir -p "$d"
  t=$SECONDS; rc=0
  ( cd "$d" && BHM_GPU_LOG="$d/gpu" timeout --kill-after=20 900 mpirun --bind-to none -n 1 "$HERE/rank-gpu.sh" "$bld/tests-single-bounds/$T" > run.log 2>&1 ) || rc=$?
  echo "$rc" > "$d/rc"
  echo "RUN m0/tests-single-bounds rc=$rc seconds=$((SECONDS - t)) cap=900 ranks=1"
  grep -v -E 'oob_tcp|safely ignore|^-+$|UCX  WARN' "$d/run.log" | tail -12
  [[ $rc == 0 ]] || status=1
else
  echo "NOT BUILT tests-single-bounds: unit test not run"
fi
source "$V/mpi-gpu/cases.sh"
for g in ${BHM_GROUPS:-m1 m2 m3}; do
  echo "==== $g"
  for prec in single double; do "${g}_cases" "$prec"; done
  if [[ $g == m3 && $bounds == 1 ]]; then m3_cases single -bounds; fi
done
echo "runs finished after $((SECONDS - T0)) s, status $status"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

echo "==== analysis"
{
  timeout 600 python3 "$V/mpi-gpu/analyze_m1.py" "$ROOT" "$V/mpi-gpu/reference/points.txt"; echo "analyze_m1 exit $?"
  timeout 300 python3 "$V/mpi-gpu/analyze_m2.py" "$ROOT"; echo "analyze_m2 exit $?"
  timeout 300 python3 "$V/mpi-gpu/analyze_m3.py" "$ROOT"; echo "analyze_m3 exit $?"
} > "$ROOT/analysis.txt" 2>&1
cat "$ROOT/analysis.txt"
( cd "$ROOT" && find . -name bhv_tables.txt -exec sha256sum {} + | sort -k2 > tables.sha256 )
( cd "$ROOT" && find . -type f -printf '%P %s\n' | sort > files.txt )
date -u
echo "job finished after $((SECONDS - T0)) s, status $status"
exit "$status"
