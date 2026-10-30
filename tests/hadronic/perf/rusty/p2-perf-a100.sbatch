#!/bin/bash -l
#SBATCH --job-name=p2-perf-a100
#SBATCH --account=cca
#SBATCH --partition=gpu
#SBATCH --constraint=a100&rocky9&nvlink
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/nmancini/ceph/entity-validation/p2-perf-%j/slurm-%j.out
#SBATCH --error=/mnt/home/nmancini/ceph/entity-validation/p2-perf-%j/slurm-%j.err
# P2 step 7: the scaling measurement of the Bethe-Heitler stage on one A100 (contract
# Resources/p2-hadronic/notes/2026-10-30-claude1-scaling-harness-contract.md), plus Codex 2's F8 GPU tracer
# checks (weights audit §F8). ONE job, <= 1 hour, chained invocations each under its own `timeout`, fresh output
# root, the build gate separate from the measurements. Written by Claude 1 (2026-10-30), NOT SUBMITTED: Codex 1
# submits through Nilay's SSH session after Nilay's explicit authorisation. Partition, QoS, account and modules are
# those of the 2026-10-03 bootstrap and the G3 jobs (`rusty-environment.md`); Codex 1 re-checks them live before
# submission (Codex 2's caution in audit §F8).
#
# Parameters (environment, with sbatch --export=ALL,ENTITY_COMMIT=...):
#   ENTITY_COMMIT  the merged head to measure (event mode + Born tables)          [required]
#   ENTITY_REPO    fork URL                               [git@github.com:mancininilay/entity.git]
#   PERF_N         ion counts                             [262144 1048576 4194304 16777216]
#   PERF_N_M3      ion count for M3 (nsys/ncu) and M6     [4194304]
#   PERF_REPEATS   [3]   PERF_WARMUP [100]   PERF_MEASURED [500]   PERF_DIAG [10]
#   DRY_RUN=1      print every step instead of executing (the local dry run handed to the referee)
#
# Steps, in priority order, with their time budgets (the job records a step as SKIPPED when the remaining
# allocation is below its budget; nothing is retried):
#   0  provenance and fresh checkout at ENTITY_COMMIT                                   2 min
#   1  build gate: Entity single precision, CUDA AMPERE80, pgen bh_proto, tests, output, MPI   15 min
#   2  M1 + M2 + M5: perf_driver.sh (every mode, warm and cold, three repeats, M5 runs)       20 min
#   3  M6 stride and M3 variants: bh_perf.xc at PERF_N_M3                                      6 min
#   4  M4: bh_reduction_timing.xc with BH_REDUCTION_N = PERF_N_M3                               2 min
#   5  M3 profiles: nsys (timeline) and ncu (kernel, limited launch count) on bh_perf.xc variants  7 min
#   6  F8 GPU tracers, single precision: stock sort / deposit / checkpoint / spectra ctest entries   4 min
#   7  F8 double precision: a second build of the test targets only, then the same entries         (if >= 14 min remain)
set -uo pipefail
DRY_RUN="${DRY_RUN:-0}"
ENTITY_COMMIT="${ENTITY_COMMIT:?set ENTITY_COMMIT to the merged head}"
ENTITY_REPO="${ENTITY_REPO:-git@github.com:mancininilay/entity.git}"
PERF_N="${PERF_N:-262144 1048576 4194304 16777216}"
PERF_N_M3="${PERF_N_M3:-4194304}"
export PERF_REPEATS="${PERF_REPEATS:-3}" PERF_WARMUP="${PERF_WARMUP:-100}" PERF_MEASURED="${PERF_MEASURED:-500}" PERF_DIAG="${PERF_DIAG:-10}"
JOB="${SLURM_JOB_ID:-dryrun}"
ROOT="${PERF_ROOT:-$HOME/ceph/entity-validation/p2-perf-$JOB}"
SRC="$ROOT/src"; BLD="$ROOT/build-single"; BLD2="$ROOT/build-double"; OUT="$ROOT/out"; LOG="$ROOT/logs"
T_START=$(date +%s)
BUDGET_S=$((60 * 60))

say() { printf '[%s +%ss] %s\n' "$(date -u +%H:%M:%S)" "$(( $(date +%s) - T_START ))" "$*"; }
remaining() { echo $(( BUDGET_S - ( $(date +%s) - T_START ) )); }
run() {
  # run <step name> <budget seconds> <command...>: skipped when the remaining time is below the budget
  local name="$1" budget="$2"; shift 2
  if [ "$(remaining)" -lt "$budget" ]; then say "SKIPPED $name: $(remaining) s remain, budget $budget s"; echo "$name SKIPPED" >> "$LOG/steps.txt"; return 0; fi
  say "START $name (budget $budget s): $*"
  if [ "$DRY_RUN" = "1" ]; then echo "$name DRY" >> "$LOG/steps.txt"; return 0; fi
  local rc=0
  timeout --kill-after=10 "$budget" "$@" > "$LOG/$name.log" 2>&1 || rc=$?
  say "END $name: exit $rc"; echo "$name $rc" >> "$LOG/steps.txt"
  return 0
}

mkdir -p "$ROOT" "$OUT" "$LOG"
say "p2-perf-a100 job $JOB on $(hostname); commit $ENTITY_COMMIT; root $ROOT; dry run $DRY_RUN"
if [ "$DRY_RUN" != "1" ]; then
  export ENTITY_BACKEND=cuda
  source "$HOME/work/env_rusty_entity.sh" || { say "environment failed"; exit 2; }
  module list 2>&1 | tee "$LOG/modules.txt"; nvidia-smi | tee "$LOG/nvidia-smi.txt"; nvcc --version | tail -1 | tee "$LOG/nvcc.txt"
fi

# ---- step 0: fresh checkout at the exact commit
run step0-checkout 120 bash -c "git clone -q '$ENTITY_REPO' '$SRC' && git -C '$SRC' checkout -q '$ENTITY_COMMIT' && git -C '$SRC' rev-parse HEAD > '$LOG/commit.txt' && git -C '$SRC' status --porcelain > '$LOG/status.txt'"
if [ "$DRY_RUN" != "1" ] && [ "$(cat "$LOG/commit.txt" 2>/dev/null)" != "$ENTITY_COMMIT" ]; then say "checkout did not land on $ENTITY_COMMIT: abort"; exit 3; fi

# ---- step 1: the build gate (separate from every measurement): single precision, CUDA, tests on
ENTITY_CMAKE=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DTESTS=ON -Dpgen=bh_proto -Dmpi=ON -Doutput=ON -Dgpu_aware_mpi=OFF
              -DKokkos_DIR="${Kokkos_DIR:-/unset}" -Dadios2_DIR="${adios2_DIR:-/unset}")
run step1-build 900 bash -c "cmake -S '$SRC' -B '$BLD' ${ENTITY_CMAKE[*]} -Dprecision=single && cmake --build '$BLD' -j ${SLURM_CPUS_PER_TASK:-8} --target entity.xc bh_check test-kernels-bh_perf.xc test-kernels-bh_reduction_timing.xc"
if [ "$DRY_RUN" != "1" ] && [ ! -x "$BLD/src/entity.xc" ]; then say "build gate failed: no entity.xc; the measurements are not run"; exit 4; fi
# MPI build: entity.xc runs under one MPI rank (the G3 pattern); the driver takes a launcher wrapper
cat > "$ROOT/entity-1rank.sh" <<WRAP
#!/bin/bash
exec mpirun -n 1 --bind-to none "$BLD/src/entity.xc" "\$@"
WRAP
chmod +x "$ROOT/entity-1rank.sh"

# ---- step 2: M1, M2, M5 through the engine (Entity's timers)
run step2-m1-m2-m5 1200 env PERF_N="$PERF_N" PERF_M5=1 PERF_LABEL="a100-$JOB-$ENTITY_COMMIT" OMP_NUM_THREADS=1 \
  bash "$SRC/tests/hadronic/perf/perf_driver.sh" "$ROOT/entity-1rank.sh" "$BLD/tests/hadronic/bh_check" "$SRC/tests/hadronic/perf/inputs" "$OUT/m1"

# ---- step 3: M6 stride and M3 variants at the kernel level
run step3-stride 180 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" stride
run step3-variants 180 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants

# ---- step 4: M4 reducer vs atomics vs per-team scratch at the contract's N
run step4-reduction 120 env BH_REDUCTION_N="$PERF_N_M3" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc"

# ---- step 5: profiles for M3 (the variants binary: production kernel vs the two variants, labelled launches)
run step5-nsys 180 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 nsys profile --trace=cuda,nvtx --stats=true -o "$OUT/nsys-variants" \
  "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants
run step5-ncu 240 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 ncu --launch-skip 20 --launch-count 12 --kernel-name regex:bh_perf --set basic \
  -o "$OUT/ncu-variants" "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants

# ---- step 6: Codex 2's F8 GPU tracers, single precision (the stock sort, deposit, checkpoint and spectra entries)
run step6-f8-single 240 bash -c "cd '$BLD' && ctest -R 'particles_sort|sort_by_key|deposit|checkpoint|spectra|injectors_global_weights|reduced_stats_weights' --output-on-failure --timeout 200 -j 1"

# ---- step 7: F8 in double precision (test targets only), if the time allows
run step7-f8-double 840 bash -c "cmake -S '$SRC' -B '$BLD2' ${ENTITY_CMAKE[*]} -Dprecision=double && cmake --build '$BLD2' -j ${SLURM_CPUS_PER_TASK:-8} --target test-framework-particles_sort.xc test-framework-sort_by_key.xc test-kernels-deposit.xc test-kernels-deposit_tiled.xc test-framework-checkpoint-data.xc test-framework-spectra.xc test-kernels-injectors_global_weights.xc test-kernels-reduced_stats_weights.xc && cd '$BLD2' && ctest -R 'particles_sort|sort_by_key|deposit|checkpoint|spectra|injectors_global_weights|reduced_stats_weights' --output-on-failure --timeout 200 -j 1"

say "DONE; steps:"; cat "$LOG/steps.txt" 2>/dev/null
[ "$DRY_RUN" = "1" ] || { cd "$ROOT" && find out logs -type f | sort > "$LOG/files.txt"; sha256sum "$LOG/commit.txt" "$OUT"/m1/*.csv 2>/dev/null > "$LOG/SHA256SUMS" || true; }
exit 0
