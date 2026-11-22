#!/usr/bin/env bash
# M0, the build job: Entity with CUDA, MPI (GPU-aware MPI off) and output, from
# two clean clones: `src-val` (dev/bh-mpi-gpu: the policy at 43d68f45 with the
# validation generator) and `src-rec` (dev/bh-reconnection-mpi: the generator copy
# of the reconnection runs). Seven trees, built side by side:
#   tests-single, tests-double, tests-single-bounds   the unit test ARCHETYPES::bethe_heitler
#   val-single, val-double, val-single-bounds         entity.xc with pgen bh_validation
#   rec-single                                        entity.xc with pgen reconnection_bh, and bp2raw
# ("bounds": Kokkos views with bounds checking). Then, on the one GPU of the job:
# the unit test on one rank in the three test trees (M0); one V1 run on two ranks
# and the toy reconnection input on four ranks, the ranks sharing the GPU (smoke
# runs: they show that the launcher works and give output for the reader gate;
# no verdict of M1 to M4 comes from them). Nothing is retried.
# BHM_ROOT and BHM_STUB exist only for the dry run on the login node: with
# BHM_STUB set no compiler is called and stub executables are written.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHM_ROOT:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}"
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
for s in val rec; do
  want=$(cat "$R/COMMIT-$s") || exit 1
  [[ $(git -C "$R/src-$s" rev-parse HEAD) == "$want" && -z $(git -C "$R/src-$s" status --porcelain) ]] || { echo "src-$s is not a clean $want"; exit 1; }
  echo "source: $R/src-$s at $want"
done
echo "Kokkos: $Kokkos_DIR"; echo "ADIOS2: $adios2_DIR"; echo "CXX: $CXX"; echo "mpicxx: $(command -v mpicxx)"
MPIOPT=(-Dmpi=ON -Dgpu_aware_mpi=OFF "-DMPI_CXX_COMPILER=$(command -v mpicxx)" "-DMPI_C_COMPILER=$(command -v mpicc)")
BOUNDS="-DCMAKE_CXX_FLAGS=-DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK"
jobs="${BHM_JOBS:-6}"

build() { # tree source precision target cmake-args...
  local name="$1" src="$2" prec="$3" target="$4"; shift 4
  local dir="$bld/$name" rc=0 t0=$SECONDS
  if [[ -n "${BHM_STUB:-}" ]]; then
    # stub of an executable for the dry run: writes what the drivers look at
    mkdir -p "$(dirname "$dir/$target")"
    cat > "$dir/$target" <<'STUB'
#!/bin/bash
echo "stub $0 $*"
if [[ "${1:-}" == -input ]] && grep -q rank_interval "$2"; then
  name=$(grep -m1 'name ' "$2" | cut -d'"' -f2)
  mkdir -p "$name/fields" "$name/particles" "$name/spectra"
  touch "$name/fields/fields.00000001.bp" "$name/particles/particles.00000001.bp" "$name/spectra/spectra.00000001.bp"
  printf '# steps time pairs\n1 0.1 0\n58 5.0 7\n' > bh_counters.txt
  printf '# steps time rank\n58 5.0 0\n' > bh_ranks.txt
elif [[ "${1:-}" == -input && "${OMPI_COMM_WORLD_RANK:-0}" == 0 ]]; then
  echo "BHV step 0 time 0 pairs 1 gauss_r 1e-6"
  echo "BHVR step 0 rank 0 pairs 1"
fi
exit 0
STUB
    chmod +x "$dir/$target"
    echo "stub" > "$bld/build-$name.log"; echo "stub" > "$bld/configure-$name.log"
  else
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
      -DKokkos_DIR="$Kokkos_DIR" -Dadios2_DIR="$adios2_DIR" "${MPIOPT[@]}" -Doutput=ON -Dprecision="$prec" "$@" \
      > "$bld/configure-$name.log" 2>&1 || rc=$?
    if [[ $rc == 0 ]]; then
      timeout --kill-after=20 1800 cmake --build "$dir" -j "$jobs" --target "$(basename "$target")" > "$bld/build-$name.log" 2>&1 || rc=$?
    fi
  fi
  local nerr=0 nwarn=0
  if [[ -f "$bld/build-$name.log" ]]; then
    nerr=$(grep -c -E '(^|[^a-zA-Z_])error([^a-zA-Z_]|$)' "$bld/build-$name.log")
    nwarn=$(grep -c -i 'warning' "$bld/build-$name.log")
  fi
  if [[ $rc == 0 && -x "$dir/$target" ]]; then
    local missing=0
    [[ -n "${BHM_STUB:-}" ]] || missing=$(ldd "$dir/$target" | grep -c 'not found')
    if [[ $missing == 0 ]]; then
      sha256sum "$dir/$target" | cut -d' ' -f1 > "$dir/BUILD_OK"
    else
      rc=1
    fi
  else
    rc=1
  fi
  echo "BUILD $name rc=$rc seconds=$((SECONDS - t0)) error_lines=$nerr warning_lines=$nwarn" > "$bld/status-$name"
}
T="tests/archetypes/test-archetypes-bethe_heitler.xc"
build tests-single        "$R/src-val" single "$T" -DTESTS=ON &
build tests-double        "$R/src-val" double "$T" -DTESTS=ON &
build tests-single-bounds "$R/src-val" single "$T" -DTESTS=ON "$BOUNDS" &
build val-single          "$R/src-val" single src/entity.xc -Dpgen=bh_validation &
build val-double          "$R/src-val" double src/entity.xc -Dpgen=bh_validation &
build val-single-bounds   "$R/src-val" single src/entity.xc -Dpgen=bh_validation "$BOUNDS" &
build rec-single          "$R/src-rec" single src/entity.xc -Dpgen=reconnection_bh &
wait
mkdir -p "$bld/tools"
if [[ -n "${BHM_STUB:-}" ]]; then
  printf '#!/bin/bash\necho stub bp2raw\n' > "$bld/tools/bp2raw"; chmod +x "$bld/tools/bp2raw"
else
  g++ -std=c++17 -O2 "$R/src-rec/pgens/reconnection_bh/tools/bp2raw.cpp" \
    $("$ENTITY_RUSTY_ROOT/install/adios2/bin/adios2-config" -s --cxx-flags --cxx-libs) -o "$bld/tools/bp2raw" > "$bld/build-bp2raw.log" 2>&1 \
    || { echo "bp2raw did not build"; cat "$bld/build-bp2raw.log"; }
fi
status=0
for name in tests-single tests-double tests-single-bounds val-single val-double val-single-bounds rec-single; do
  cat "$bld/status-$name"
  if [[ -f "$bld/$name/BUILD_OK" ]]; then
    grep -i 'warning' "$bld/build-$name.log" | sed 's/\x1b\[[0-9;]*m//g' | sort | uniq -c | head -8
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
  [[ $rc == 0 ]] || status=1
}

echo "==== M0: the unit test ARCHETYPES::bethe_heitler on one rank"
for name in tests-single tests-double tests-single-bounds; do
  if [[ -f "$bld/$name/BUILD_OK" ]]; then
    launch "$bld/m0/$name" 900 1 "$bld/$name/$T"
    tail -40 "$bld/m0/$name/run.log"
  else
    echo "NOT BUILT $name: unit test not run"; status=1
  fi
done

echo "==== smoke: V1, leptons first, on two ranks (sharing the GPU)"
if [[ -f "$bld/val-single/BUILD_OK" ]]; then
  d="$bld/smoke/val-r2"; mkdir -p "$d"
  python3 "$R/src-val/pgens/bh_validation/make_input.py" "$d/input.toml" name=smoke decomposition="2, 1" order=leptons_first policy=on rank_lines=true > "$d/make_input.log" 2>&1
  launch "$d" 300 2 "$bld/val-single/src/entity.xc" -input input.toml
  grep -c '^BHV step' "$d/run.log"; grep '^BHV step' "$d/run.log" | tail -1 | cut -c1-400
  grep '^BHVR step' "$d/run.log" | tail -2 | cut -c1-260
  cat "$d"/gpu.* 2>/dev/null
fi

echo "==== smoke: the toy reconnection input on four ranks (sharing the GPU), runtime 5"
if [[ -f "$bld/rec-single/BUILD_OK" ]]; then
  d="$bld/smoke/rec-r4"; mkdir -p "$d"
  sed 's/^  runtime = 500.0$/  runtime = 5.0/' "$R/src-rec/pgens/reconnection_bh/mpi/toy-low-mpi.toml" > "$d/toy-low-mpi.toml"
  grep -q '^  runtime = 5.0$' "$d/toy-low-mpi.toml" || { echo "smoke input not made"; status=1; }
  launch "$d" 600 4 "$bld/rec-single/src/entity.xc" -input toy-low-mpi.toml
  tail -5 "$d/run.log"
  for f in bh_counters.txt bh_ranks.txt; do
    [[ -s "$d/$f" ]] || { echo "smoke: no $f"; status=1; }
  done
  for k in fields particles spectra; do
    n=$(ls "$d/toy-low-mpi/$k" 2>/dev/null | wc -l); echo "smoke: $n outputs of $k"
    [[ $n -ge 1 ]] || status=1
  done
  tail -1 "$d/bh_counters.txt" 2>/dev/null | cut -c1-300
  tail -4 "$d/bh_ranks.txt" 2>/dev/null
  grep -h 'Bethe-Heitler emission' "$d/run.log" | head -2
  cat "$d"/gpu.* 2>/dev/null
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
if [[ $status == 0 ]]; then
  echo "$bld" > "$R/BUILD"
  echo "every tree built, the unit tests and the smoke runs ended with status 0; $R/BUILD points to $bld"
else
  echo "build NOT marked usable"
fi
date -u
echo "total seconds: $((SECONDS - T0)), status $status"
exit "$status"
