#!/usr/bin/env bash
# Job 1 of the Rusty proposal (report of 2026-11-19, section 8): CUDA builds of
# the Bethe-Heitler emission policy. Compile only: nothing is run on the GPU.
#   test-archetypes-bethe_heitler.xc    dev/bh-emission 66aa6fb6             single, double
#   entity.xc with pgens/bh_validation  dev/bh-emission-validation a8a6f399  single, double
# Every build is attempted; a tree that built completely gets the marker BUILD_OK.
set -uo pipefail
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
B="$HOME/ceph/entity-builds/bh-emission-claude1-20261119"
O="$HOME/ceph/bh-emission-claude1-20261119"
cuda="$B/cuda-build-$SLURM_JOB_ID"
log="$O/build-$SLURM_JOB_ID"
mkdir -p "$log" || exit 1
mkdir "$cuda" || exit 1
exec > >(tee "$log/driver.log") 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$log/exit-status"' EXIT
hostname
date -u
module list 2>&1
g++ --version | head -1
nvcc --version | tail -2
cmake --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
pr=66aa6fb669fcc564a5748e876c972833e1559dcc
val=a8a6f39946657819e6f021343203c0422644df1d
[[ $(git -C "$B/src-pr" rev-parse HEAD) == "$pr" && -z $(git -C "$B/src-pr" status --porcelain) ]] || { echo "src-pr is not a clean $pr"; exit 1; }
[[ $(git -C "$B/src" rev-parse HEAD) == "$val" && -z $(git -C "$B/src" status --porcelain) ]] || { echo "src is not a clean $val"; exit 1; }
echo "Kokkos: $Kokkos_DIR"
echo "CXX: $CXX"
jobs=${SLURM_CPUS_PER_TASK:-8}
status=0
build() { # name source precision target cmake-args...
  local name="$1" src="$2" prec="$3" target="$4"; shift 4
  local dir="$cuda/$name" rc=0 t0=$SECONDS
  cmake -S "$src" -B "$dir" -G Ninja \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DKokkos_DIR="$Kokkos_DIR" \
    -Dmpi=OFF -Doutput=OFF -Dprecision="$prec" "$@" > "$log/configure-$name.log" 2>&1 || rc=$?
  if [[ $rc == 0 ]]; then
    cmake --build "$dir" -j "$jobs" --target "$target" > "$log/build-$name.log" 2>&1 || rc=$?
  fi
  local nerr=0 nwarn=0
  if [[ -f "$log/build-$name.log" ]]; then
    nerr=$(grep -c -E '(^|[^a-zA-Z_])error([^a-zA-Z_]|$)' "$log/build-$name.log")
    nwarn=$(grep -c -i 'warning' "$log/build-$name.log")
  fi
  echo "BUILD $name rc=$rc seconds=$((SECONDS - t0)) error_lines=$nerr warning_lines=$nwarn"
  if [[ $rc == 0 ]]; then
    touch "$dir/BUILD_OK"
  else
    status=1
    tail -40 "$log/build-$name.log" 2>/dev/null || tail -40 "$log/configure-$name.log"
  fi
}
build pr-single  "$B/src-pr" single test-archetypes-bethe_heitler.xc -DTESTS=ON
build val-single "$B/src"    single entity.xc -Dpgen=bh_validation
build pr-double  "$B/src-pr" double test-archetypes-bethe_heitler.xc -DTESTS=ON
build val-double "$B/src"    double entity.xc -Dpgen=bh_validation
for x in "$cuda"/pr-*/tests/archetypes/test-archetypes-bethe_heitler.xc "$cuda"/val-*/src/entity.xc; do
  [[ -f "$x" ]] || continue
  ls -la "$x"
  sha256sum "$x"
  echo "unresolved libraries: $(ldd "$x" | grep -c 'not found')"
done
for name in pr-single val-single pr-double val-double; do
  grep -E 'Git hash|Enabled Kokkos devices' "$log/configure-$name.log" 2>/dev/null | sed "s/^/$name: /"
done
date -u
echo "CUDA builds finished, status $status"
exit "$status"
