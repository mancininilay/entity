#!/usr/bin/env bash
# The build job of the resolution pair (Claude 1, 2026-11-23): Entity with CUDA, MPI (GPU-aware
# MPI off) and output, single precision, from the clean clone `src` at `COMMIT`, two trees side by
# side:
#   rec-single   entity.xc with pgen reconnection_bh (the executable of every run but r-low-1card)
#   tests        the engine test of the capacity of the particle exchange (comm-capacity)
# and bp2raw. Then, on the one GPU of the job: the four cases of comm-capacity on one and two
# ranks (the fix of upstream issue U19 with CUDA and Open MPI), and the toy copy of the input of
# the pair (`series/toy-s1.toml`, 200 steps) on four ranks sharing the GPU, as a smoke run. The
# tree is marked usable (`BUILD`) only if everything ends as it should. Nothing is retried.
# BHP_ROOT and BHP_STUB exist only for the dry run (stub executables, no compiler).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHP_ROOT:-$HOME/ceph/bh-resolution-pair-claude1-20261123}"
bld="$R/build-$SLURM_JOB_ID"
mkdir "$bld" || exit 1
exec > "$bld/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$bld/exit-status"' EXIT
T0=$SECONDS
hostname; date -u
module list 2>&1
g++ --version | head -1; nvcc --version | tail -2; cmake --version | head -1; mpirun --version | head -1
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
grep Cpus_allowed_list /proc/self/status
src="$R/src"
want=$(cat "$R/COMMIT") || exit 1
[[ $(git -C "$src" rev-parse HEAD) == "$want" && -z $(git -C "$src" status --porcelain) ]] || { echo "src is not a clean $want"; exit 1; }
echo "$want" > "$bld/COMMIT"
echo "source: $src at $want"
echo "Kokkos: ${Kokkos_DIR:-}"; echo "ADIOS2: ${adios2_DIR:-}"; echo "CXX: ${CXX:-}"; echo "mpicxx: $(command -v mpicxx)"
MPIOPT=(-Dmpi=ON -Dgpu_aware_mpi=OFF "-DMPI_CXX_COMPILER=$(command -v mpicxx)" "-DMPI_C_COMPILER=$(command -v mpicc)")
jobs="${BHP_JOBS:-8}"

build() { # tree target cmake-args...
  local name="$1" target="$2"; shift 2
  local dir="$bld/$name" rc=0 t0=$SECONDS
  if [[ -n "${BHP_STUB:-}" ]]; then
    mkdir -p "$(dirname "$dir/$target")"
    cp "$BHP_STUB/$(basename "$target")" "$dir/$target" && chmod +x "$dir/$target"
    echo stub > "$bld/build-$name.log"; echo stub > "$bld/configure-$name.log"
  else
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
      -DKokkos_DIR="$Kokkos_DIR" -Dadios2_DIR="$adios2_DIR" "${MPIOPT[@]}" -Doutput=ON -Dprecision=single "$@" \
      > "$bld/configure-$name.log" 2>&1 || rc=$?
    if [[ $rc == 0 ]]; then
      timeout --kill-after=20 2100 cmake --build "$dir" -j "$jobs" --target "$(basename "$target")" > "$bld/build-$name.log" 2>&1 || rc=$?
    fi
  fi
  if [[ $rc == 0 && -x "$dir/$target" ]]; then
    local missing=0
    [[ -n "${BHP_STUB:-}" ]] || missing=$(ldd "$dir/$target" | grep -c 'not found')
    if [[ $missing == 0 ]]; then sha256sum "$dir/$target" | cut -d' ' -f1 > "$dir/BUILD_OK"; else rc=1; fi
  else
    rc=1
  fi
  echo "BUILD $name rc=$rc seconds=$((SECONDS - t0))" > "$bld/status-$name"
}
T="tests/framework/test-framework-comm-capacity.xc"
build rec-single src/entity.xc -Dpgen=reconnection_bh &
build tests "$T" -Dpgen=reconnection_bh -DTESTS=ON &
wait
mkdir -p "$bld/tools"
if [[ -n "${BHP_STUB:-}" ]]; then
  printf '#!/bin/bash\necho stub bp2raw\n' > "$bld/tools/bp2raw"; chmod +x "$bld/tools/bp2raw"
else
  g++ -std=c++17 -O2 "$src/pgens/reconnection_bh/tools/bp2raw.cpp" \
    $("$ENTITY_RUSTY_ROOT/install/adios2/bin/adios2-config" -s --cxx-flags --cxx-libs) -o "$bld/tools/bp2raw" > "$bld/build-bp2raw.log" 2>&1 \
    || { echo "bp2raw did not build"; cat "$bld/build-bp2raw.log"; }
fi
status=0
for name in rec-single tests; do
  cat "$bld/status-$name"
  if [[ -f "$bld/$name/BUILD_OK" ]]; then
    grep -E 'Git hash|Enabled Kokkos devices|MPI|gpu_aware' "$bld/configure-$name.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -6
  else
    status=1
    tail -30 "$bld/build-$name.log" 2>/dev/null || tail -30 "$bld/configure-$name.log"
  fi
done
echo "builds finished after $((SECONDS - T0)) s, status $status"

launch() { # directory cap ranks command...
  local dir="$1" cap="$2" ranks="$3"; shift 3
  mkdir -p "$dir"
  local t=$SECONDS rc=0
  ( cd "$dir" && BHM_GPU_LOG="$dir/gpu" timeout --kill-after=20 "$cap" mpirun --bind-to none -n "$ranks" "$HERE/rank-gpu.sh" "$@" > run.log 2>&1 ) || rc=$?
  echo "$rc" > "$dir/rc"
  echo "RUN ${dir#$bld/} rc=$rc seconds=$((SECONDS - t)) cap=$cap ranks=$ranks"
  return $rc
}

echo "==== the capacity test of the particle exchange (U19), on the GPU of the job"
if [[ -f "$bld/tests/BUILD_OK" ]]; then
  for c in full fill holes; do
    for n in 1 2; do
      launch "$bld/u19/$c-$n" 300 "$n" "$bld/tests/$T" "$c"
      grep -q "^OK $c on $n rank" "$bld/u19/$c-$n/run.log" && echo "u19 $c on $n: ran to the end" || { echo "u19 $c on $n: FAILED"; status=1; }
    done
  done
  launch "$bld/u19/over-2" 300 2 "$bld/tests/$T" over
  if grep -q "Too many particles to receive" "$bld/u19/over-2/run.log" && ! grep -q "OVERFLOW NOT DETECTED" "$bld/u19/over-2/run.log"; then
    echo "u19 over on 2: stopped, as it must"
  else
    echo "u19 over on 2: NOT stopped"; status=1
  fi
else
  echo "the test tree did not build"; status=1
fi

echo "==== smoke: the toy copy of the input of the pair on four ranks sharing the GPU, 200 steps"
if [[ -f "$bld/rec-single/BUILD_OK" ]]; then
  d="$bld/smoke/toy-s1"; mkdir -p "$d"
  cp "$src/pgens/reconnection_bh/series/toy-s1.toml" "$d/" || status=1
  launch "$d" 900 4 "$bld/rec-single/src/entity.xc" -input toy-s1.toml || status=1
  tail -5 "$d/run.log"
  for f in bh_counters.txt bh_gauss.txt bh_ranks.txt bh_profile.txt; do
    [[ -s "$d/$f" ]] || { echo "smoke: no $f"; status=1; }
  done
  n=$(grep -c -E 'FATAL|ERROR' "$d/run.log"); echo "smoke: lines with FATAL or ERROR: $n"; [[ $n == 0 ]] || status=1
  for k in fields particles spectra; do echo "smoke: $(ls "$d/toy-s1/$k" 2>/dev/null | wc -l) outputs of $k"; done
  tail -1 "$d/bh_counters.txt" 2>/dev/null | cut -c1-300
  tail -1 "$d/bh_gauss.txt" 2>/dev/null
  tail -4 "$d/bh_ranks.txt" 2>/dev/null
  cat "$d"/gpu.* 2>/dev/null
else
  status=1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
if [[ $status == 0 ]]; then
  echo "$bld" > "$R/BUILD"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) build $bld of $want" >> "$R/BUILD.history"
  echo "both trees built, the capacity test and the smoke run ended as they should; $R/BUILD points to $bld"
else
  echo "build NOT marked usable"
fi
date -u
echo "total seconds: $((SECONDS - T0)), status $status"
exit "$status"
