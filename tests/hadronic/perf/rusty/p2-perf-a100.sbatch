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
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/nmancini/ceph/entity-validation/p2-perf-logs/slurm-%j.out
#SBATCH --error=/mnt/home/nmancini/ceph/entity-validation/p2-perf-logs/slurm-%j.err
# P2 step 7: the scaling measurement of the Bethe-Heitler stage on one A100 (contract
# Resources/p2-hadronic/notes/2026-10-30-claude1-scaling-harness-contract.md), plus Codex 2's F8 GPU tracer
# checks (weights audit §F8). ONE job, <= 1 hour, chained invocations each under its own `timeout`, a fresh
# output root created exclusively, the checkout and build gates separate from the measurements. Written by Claude 1
# (2026-10-30; round 2 after Codex 1's review of 2026-10-30, F2/F3/F6/F8/F9), NOT SUBMITTED: Codex 1 submits
# through Nilay's SSH session after Nilay's explicit authorisation.
#
# PRECONDITION (Codex 1's F2.1): Slurm opens the stdout/stderr files before this script runs, so their parent
#   directory must exist beforehand: `mkdir -p ~/ceph/entity-validation/p2-perf-logs` (rusty/preflight.sh does it
#   and then runs `sbatch --test-only`). The per-job run root is created by this script, exclusively.
#
# Parameters (environment, with sbatch --export=ALL,ENTITY_COMMIT=...):
#   ENTITY_COMMIT  the merged head to measure (event mode + Born tables)          [required]
#   ENTITY_REPO    fork URL                               [git@github.com:mancininilay/entity.git]
#   PERF_ROOT      run root (must NOT exist)              [~/ceph/entity-validation/p2-perf-<jobid>]
#   PERF_N         ion counts                             [262144 1048576 4194304 16777216]
#   PERF_N_M3      ion count for M3 (nsys/ncu) and M6     [4194304, fixed for this authorisation]
#   PERF_REPEATS   [3]   PERF_REPEATS_TOP [2: the F3 cut, repeats at 2^24 only]   PERF_WARMUP [50: the F3 cut]
#   PERF_MEASURED  [500]  PERF_DIAG [1: every measured step is summed, F4]
#   PERF_QK_SOURCE [born] the q_k source of the ENGINE timing runs (the production default). Every bh_proto process
#                  builds the Born node table at startup (~5 s on 10 host threads on the Mac; Codex 1 measured 8 to
#                  20 s under load): with ~140 engine processes that is the largest single item of the hour. "fit"
#                  (the step-1 tables, same table layout, same kernel work, no startup build) is PROPOSED in the
#                  round-2 report §F3 for the timing runs; the kernel-level groups (M3, M6) run on the Born tables in
#                  either case (BH_QK_SOURCE=born, one table build per process). Nilay decides; the default stays born.
#   DRY_RUN=1      print every step instead of executing (the local dry run handed to the referee)
#
# Step budgets (ceilings; the estimate and its basis are in the round-2 report §F3). Sum of the ceilings with
# PERF_QK_SOURCE=born: 3630 s (the double-precision F8 step is the one at risk and the status file says so);
# with PERF_QK_SOURCE=fit: 2970 s, margin 630 s. Priority order = execution order; a step that no longer fits
# is SKIPPED and the job ends incomplete_coverage.
#   0  checkout                                                                  120 s
#   1  build single (entity.xc, bh_check, bh_perf, bh_reduction_timing, f8_tracers)      780 s
#   2  M1 + M2 + M5 (+ the M5 restart runs on the end-loaded populations, F8)   1260 s (600 s with fit)
#   3  M6 stride at PERF_N_M3 (event, none, poisson, budget; n = 1, 4, 16)        240 s
#      M3 variants at PERF_N_M3 (matched work)                                   180 s
#   4  M4 reducer vs atomics vs per-team, equal boundaries                        90 s
#   5  profiles: nsys and ncu per configuration, with content assertions         300 s
#   6  F8 single: FRAMEWORK::f8_tracers + the stock sort/deposit/checkpoint/spectra entries      90 s
#   7  F8 double: build of the test targets (test-framework-f8_tracers.xc and the stock entries)  480 s
#      + run                                                                      90 s
#
# Outcome (Codex 1's F2.2): every step is recorded in logs/steps.tsv (name, kind, budget, seconds, exit, class)
#   and the job ends with logs/status.json and the exit code of its overall class:
#     complete                  every step ran and passed                                  exit 0
#     scientific_check_failure  a measurement step's own check failed (ledger identity, the M6 4 s.e. cooling
#                               check, an F8 assertion): the step exited 1 and its log carries a FAIL line        exit 10
#     infrastructure_failure    checkout/build gate, timeout (124), kill (137), exception (2), missing binary
#                               or missing profile content                                                 exit 20
#     incomplete_coverage       a step was SKIPPED for lack of time or BLOCKED by a failed gate            exit 30
#   Precedence of the overall class: infrastructure > incomplete > scientific > complete; the three flags are
#   written separately so nothing is hidden. Independent steps continue after a failure; nothing is retried.
set -uo pipefail
DRY_RUN="${DRY_RUN:-0}"
ENTITY_COMMIT="${ENTITY_COMMIT:?set ENTITY_COMMIT to the merged head}"
ENTITY_REPO="${ENTITY_REPO:-git@github.com:mancininilay/entity.git}"
PERF_N="${PERF_N:-262144 1048576 4194304 16777216}"
PERF_N_M3="${PERF_N_M3:-4194304}"
export PERF_REPEATS="${PERF_REPEATS:-3}" PERF_REPEATS_TOP="${PERF_REPEATS_TOP:-2}" PERF_WARMUP="${PERF_WARMUP:-50}" PERF_MEASURED="${PERF_MEASURED:-500}" PERF_DIAG="${PERF_DIAG:-1}"
export PERF_QK_SOURCE="${PERF_QK_SOURCE:-born}"
JOB="${SLURM_JOB_ID:-dryrun-$$}"
ROOT="${PERF_ROOT:-$HOME/ceph/entity-validation/p2-perf-$JOB}"
SRC="$ROOT/src"; BLD="$ROOT/build-single"; BLD2="$ROOT/build-double"; OUT="$ROOT/out"; LOG="$ROOT/logs"
T_START=$(date +%s)
BUDGET_S=$((60 * 60))
GATE_OK=0          # 1 once the checkout and the single build gate passed
ANY_INFRA=0; ANY_SCI=0; ANY_SKIP=0

say() { printf '[%s +%ss] %s\n' "$(date -u +%H:%M:%S)" "$(( $(date +%s) - T_START ))" "$*"; }
remaining() { echo $(( BUDGET_S - ( $(date +%s) - T_START ) )); }
record() { # record <name> <kind> <budget> <seconds> <exit> <class>
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$LOG/steps.tsv"
  case "$6" in infrastructure_failure) ANY_INFRA=1 ;; scientific_check_failure) ANY_SCI=1 ;; skipped|blocked) ANY_SKIP=1 ;; esac
}
# run <step name> <kind: infra|sci> <budget seconds> <command...>
#   skipped when the remaining time is below the budget (class skipped); blocked when the gate has not passed
#   (class blocked); otherwise run under timeout and classified: exit 0 -> ok; for kind sci, exit 1 with a FAIL
#   line in the log -> scientific_check_failure; any other non-zero exit (timeout 124, kill 137, exception 2,
#   missing binary 126/127) -> infrastructure_failure. Returns the command's exit code.
run() {
  local name="$1" kind="$2" budget="$3"; shift 3
  local t0 rc=0 cls
  if [ "$kind" != "gate" ] && [ "$GATE_OK" -ne 1 ]; then
    say "BLOCKED $name: the checkout/build gate did not pass"; record "$name" "$kind" "$budget" 0 "-" blocked; return 99
  fi
  if [ "$(remaining)" -lt "$budget" ]; then
    say "SKIPPED $name: $(remaining) s remain, budget $budget s"; record "$name" "$kind" "$budget" 0 "-" skipped; return 98
  fi
  say "START $name [$kind] (budget $budget s): $*"
  if [ "$DRY_RUN" = "1" ]; then record "$name" "$kind" "$budget" 0 0 dry; return 0; fi
  t0=$(date +%s)
  timeout --kill-after=10 "$budget" "$@" > "$LOG/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then cls=ok
  elif [ "$kind" = "sci" ] && [ $rc -eq 1 ] && grep -q "FAIL" "$LOG/$name.log"; then cls=scientific_check_failure
  else cls=infrastructure_failure; fi
  say "END $name: exit $rc ($cls, $(( $(date +%s) - t0 )) s)"
  record "$name" "$kind" "$budget" $(( $(date +%s) - t0 )) "$rc" "$cls"
  return $rc
}
# require_files <step name> <file...>: an infrastructure failure when any listed file is missing or empty
require_files() {
  local name="$1"; shift
  local f missing=0
  for f in "$@"; do [ -s "$f" ] || { say "MISSING for $name: $f"; missing=1; }; done
  [ $missing -eq 0 ] || record "$name-inventory" infra 0 0 "-" infrastructure_failure
  return $missing
}
finish() {
  local overall rc
  if [ "$ANY_INFRA" -eq 1 ]; then overall=infrastructure_failure; rc=20
  elif [ "$ANY_SKIP" -eq 1 ]; then overall=incomplete_coverage; rc=30
  elif [ "$ANY_SCI" -eq 1 ]; then overall=scientific_check_failure; rc=10
  else overall=complete; rc=0; fi
  {
    printf '{\n  "job": "%s",\n  "commit": "%s",\n  "root": "%s",\n  "dry_run": %s,\n' "$JOB" "$ENTITY_COMMIT" "$ROOT" "$DRY_RUN"
    printf '  "overall": "%s",\n  "any_infrastructure_failure": %s,\n  "any_scientific_check_failure": %s,\n  "any_skipped_or_blocked": %s,\n' \
      "$overall" "$ANY_INFRA" "$ANY_SCI" "$ANY_SKIP"
    printf '  "elapsed_s": %s,\n  "steps": [\n' "$(( $(date +%s) - T_START ))"
    awk -F'\t' 'NR > 1 { printf ",\n" } { printf "    {\"name\": \"%s\", \"kind\": \"%s\", \"budget_s\": %s, \"seconds\": %s, \"exit\": \"%s\", \"class\": \"%s\"}", $1, $2, $3, $4, $5, $6 }' "$LOG/steps.tsv"
    printf '\n  ]\n}\n'
  } > "$ROOT/status.json"
  say "DONE: overall $overall (exit $rc); steps:"; cat "$LOG/steps.tsv"
  if [ "$DRY_RUN" != "1" ]; then
    ( cd "$ROOT" && find out logs -type f | sort > "$LOG/files.txt" && sha256sum "$LOG/commit.txt" "$ROOT/status.json" "$OUT"/m1/*.csv > "$LOG/SHA256SUMS" 2>/dev/null ) || true
  fi
  exit $rc
}

# ---- PERF_SELFTEST=1: the negative probes of Codex 1's F2.2 on this script's own functions (no checkout, build or
#      measurement): an injected failure, a step skipped for time, a sci step exiting 1 with a FAIL line, a blocked
#      step; prints steps.tsv and status.json and exits with the overall class (expected: infrastructure_failure, 20)
if [ "${PERF_SELFTEST:-0}" = "1" ]; then
  ROOT="${PERF_ROOT:-$(mktemp -d /tmp/p2perf-selftest.XXXXXX)/root}"; LOG="$ROOT/logs"; OUT="$ROOT/out"
  mkdir -p "$LOG" "$OUT" "$OUT/m1"; : > "$LOG/steps.tsv"; : > "$LOG/commit.txt"
  GATE_OK=1
  run probe-ok sci 60 /usr/bin/true;                          echo "probe-ok returned $?"
  run probe-injected-failure sci 60 /usr/bin/false;           echo "probe-injected-failure returned $? (expected 1, class infrastructure_failure: exit 1 without a FAIL line)"
  run probe-sci-fail sci 60 bash -c 'echo "FAIL: injected scientific check"; exit 1'; echo "probe-sci-fail returned $? (expected 1, class scientific_check_failure)"
  run probe-timeout infra 1 sleep 5;                          echo "probe-timeout returned $? (expected 124, class infrastructure_failure)"
  BUDGET_S=0; run probe-skipped sci 60 /usr/bin/true;         echo "probe-skipped returned $? (expected 98, class skipped)"; BUDGET_S=3600
  GATE_OK=0; run probe-blocked sci 60 /usr/bin/true;          echo "probe-blocked returned $? (expected 99, class blocked)"; GATE_OK=1
  require_files probe-inventory "$ROOT/does-not-exist";       echo "require_files returned $? (expected 1)"
  DRY_RUN=1
  finish
fi

# ---- the run root: exclusive creation (Codex 1's F2.3: an existing root would let the driver overwrite CSVs)
if [ -e "$ROOT" ]; then echo "run root $ROOT exists: refusing to reuse it (fresh root per job)"; exit 5; fi
mkdir "$ROOT" || { echo "cannot create the run root $ROOT"; exit 5; }
mkdir -p "$OUT" "$LOG"
: > "$LOG/steps.tsv"
say "p2-perf-a100 job $JOB on $(hostname); commit $ENTITY_COMMIT; root $ROOT; dry run $DRY_RUN; budget $BUDGET_S s"
if [ "$DRY_RUN" != "1" ]; then
  export ENTITY_BACKEND=cuda
  source "$HOME/work/env_rusty_entity.sh" || { say "environment failed"; record environment infra 0 0 2 infrastructure_failure; finish; }
  module list 2>&1 | tee "$LOG/modules.txt"; nvidia-smi | tee "$LOG/nvidia-smi.txt"; nvcc --version | tail -1 | tee "$LOG/nvcc.txt"
fi

# ---- step 0: fresh checkout at the exact commit (gate)
GATE_OK=1
run step0-checkout gate 120 bash -c "git clone -q '$ENTITY_REPO' '$SRC' && git -C '$SRC' checkout -q '$ENTITY_COMMIT' && git -C '$SRC' rev-parse HEAD > '$LOG/commit.txt' && git -C '$SRC' status --porcelain > '$LOG/status.txt'"
rc=$?
if [ "$DRY_RUN" != "1" ]; then
  if [ $rc -ne 0 ] || [ "$(cat "$LOG/commit.txt" 2>/dev/null)" != "$ENTITY_COMMIT" ] || [ -s "$LOG/status.txt" ]; then
    say "checkout gate failed (exit $rc, head '$(cat "$LOG/commit.txt" 2>/dev/null)', dirty '$(wc -l < "$LOG/status.txt" 2>/dev/null)'): every later step is BLOCKED"
    [ $rc -eq 0 ] && record step0-checkout-head gate 0 0 "-" infrastructure_failure
    GATE_OK=0
  fi
fi

# ---- step 1: the build gate (separate from every measurement): single precision, CUDA, tests on; the gate is the
#      build command's success AND the inventory of every binary the later steps need
ENTITY_CMAKE=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DTESTS=ON -Dpgen=bh_proto -Dmpi=ON -Doutput=ON -Dgpu_aware_mpi=OFF
              -DKokkos_DIR="${Kokkos_DIR:-/unset}" -Dadios2_DIR="${adios2_DIR:-/unset}")
# F8 (Codex 2 §F8, Codex 1 F3): the tracer executable tests/framework/f8_tracers.cpp (sort, next-step deposition, split,
# checkpoint round trip, 1D/2D/3D, enumerated assertions) plus the stock entries that exist in a flat-deposit build
# (particles_sort, deposit, checkpoint-data, checkpoint-extents, spectra, injectors_global_weights, reduced_stats_weights;
# sort_by_key and deposit_tiled exist only with tiled_deposit=ON, which is not this build: see the report's proposal)
F8_TARGETS="test-framework-f8_tracers.xc test-framework-particles_sort.xc test-kernels-deposit.xc test-framework-checkpoint-data.xc test-framework-checkpoint-extents.xc test-framework-spectra.xc test-kernels-injectors_global_weights.xc test-kernels-reduced_stats_weights.xc"
F8_NAMES="FRAMEWORK::f8_tracers FRAMEWORK::particles_sort KERNELS::deposit FRAMEWORK::checkpoint-data FRAMEWORK::checkpoint-extents FRAMEWORK::spectra KERNELS::injectors_global_weights KERNELS::reduced_stats_weights"
SINGLE_TARGETS="entity.xc bh_check test-kernels-bh_perf.xc test-kernels-bh_reduction_timing.xc $F8_TARGETS"
SINGLE_BINARIES=("$BLD/src/entity.xc" "$BLD/tests/hadronic/bh_check" "$BLD/tests/kernels/test-kernels-bh_perf.xc" "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc" "$BLD/tests/framework/test-framework-f8_tracers.xc")
run step1-build gate 780 bash -c "cmake -S '$SRC' -B '$BLD' ${ENTITY_CMAKE[*]} -Dprecision=single && cmake --build '$BLD' -j ${SLURM_CPUS_PER_TASK:-8} --target $SINGLE_TARGETS"
rc=$?
if [ "$DRY_RUN" != "1" ] && [ "$GATE_OK" -eq 1 ]; then
  if [ $rc -ne 0 ] || ! require_files step1-build "${SINGLE_BINARIES[@]}"; then
    say "build gate failed: every measurement is BLOCKED"; GATE_OK=0
  fi
fi
# MPI build: entity.xc runs under one MPI rank (the G3 pattern); the driver takes a launcher wrapper
cat > "$ROOT/entity-1rank.sh" <<WRAP
#!/bin/bash
exec mpirun -n 1 --bind-to none "$BLD/src/entity.xc" "\$@"
WRAP
chmod +x "$ROOT/entity-1rank.sh"

# ---- step 2: M1, M2, M5 through the engine (Entity's timers)
STEP2_BUDGET=1260; [ "$PERF_QK_SOURCE" = "fit" ] && STEP2_BUDGET=600
run step2-m1-m2-m5 sci "$STEP2_BUDGET" env PERF_N="$PERF_N" PERF_M5=1 PERF_LABEL="a100-$JOB-$ENTITY_COMMIT" OMP_NUM_THREADS=1 \
  bash "$SRC/tests/hadronic/perf/perf_driver.sh" "$ROOT/entity-1rank.sh" "$BLD/tests/hadronic/bh_check" "$SRC/tests/hadronic/perf/inputs" "$OUT/m1"

# ---- step 3: M6 stride and M3 variants at the kernel level
run step3-stride sci 240 env BH_PERF_N="$PERF_N_M3" BH_QK_SOURCE=born OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" stride
run step3-variants sci 180 env BH_PERF_N="$PERF_N_M3" BH_QK_SOURCE=born OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants

# ---- step 4: M4 reducer vs atomics vs per-team scratch at the contract's N
run step4-reduction sci 90 env BH_REDUCTION_N="$PERF_N_M3" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc"

# ---- step 5: profiles for M3 (the variants binary: production kernel vs the two variants, labelled launches)
run step5-nsys infra 180 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 nsys profile --trace=cuda,nvtx --stats=true -o "$OUT/nsys-variants" \
  "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants
run step5-ncu infra 240 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 ncu --launch-skip 20 --launch-count 12 --kernel-name regex:bh_perf --set basic \
  -o "$OUT/ncu-variants" "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants

# ---- step 6: Codex 2's F8 GPU tracers, single precision: the tracer executable and the stock entries; the selection
#      must be non-empty and complete (F8_EXPECTED entries run), else the step is an infrastructure failure
f8_ctest() { # f8_ctest <build dir>: every named entry must exist (exact name) and is run on its own; the step fails on a
             # missing entry (selection gate, exit 3) and on any failed entry (FAIL line, exit 1)
  cd "$1" || return 2
  local name rc=0 n_run=0 n_fail=0
  for name in $F8_NAMES; do
    if ! ctest -N | grep -Eq "Test +#[0-9]+: ${name}\$"; then echo "F8 selection gate: entry $name is not registered in $1"; return 3; fi
    if ctest -R "^${name}\$" --output-on-failure --timeout 80; then n_run=$((n_run + 1)); else n_run=$((n_run + 1)); n_fail=$((n_fail + 1)); echo "FAIL: F8 entry $name"; fi
  done
  echo "F8: $n_run entries run, $n_fail failed"
  [ $n_fail -eq 0 ] || rc=1
  return $rc
}
export -f f8_ctest; export F8_NAMES
run step6-f8-single sci 90 bash -c "f8_ctest '$BLD'"

# ---- step 7: F8 in double precision: the test targets only (the libraries they need are the bulk of the build), then
#      the same entries; the step runs last so that the registered measurements come first, and it is SKIPPED (status
#      incomplete_coverage) if the hour does not hold it
run step7-f8-double-build infra 480 bash -c "cmake -S '$SRC' -B '$BLD2' ${ENTITY_CMAKE[*]} -Dprecision=double && cmake --build '$BLD2' -j ${SLURM_CPUS_PER_TASK:-8} --target $F8_TARGETS"
if [ "$DRY_RUN" = "1" ] || [ -x "$BLD2/tests/framework/test-framework-f8_tracers.xc" ]; then
  run step7-f8-double sci 90 bash -c "f8_ctest '$BLD2'"
else
  say "step7-f8-double BLOCKED: the double build did not produce the tracer executable"; record step7-f8-double sci 90 0 "-" blocked
fi

finish
