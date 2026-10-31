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
# (2026-10-30; round 2 after Codex 1's review of 2026-10-30, F2/F3/F6/F8/F9; round 3 after Codex 1's closure of
# 2026-10-31, C1/C4), NOT SUBMITTED: Codex 1 submits through Nilay's SSH session after Nilay's explicit authorisation.
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
#   PERF_QK_SOURCE [fit] the q_k source of the ENGINE timing runs. DECIDED by Nilay on 2026-10-31 ("fit for engine
#                  runs", decisions.md): the step-1 tables for the 143 engine processes of step 2, because every
#                  bh_proto process would otherwise build the Born node table at startup (6.5 s on 10 Mac threads,
#                  8 to 20 s under load), the largest single item of the hour, while the table layout and the
#                  kernel work per step are identical. The kernel-level groups (M3, M6, the profiles) run on the
#                  Born tables (BH_QK_SOURCE=born, one table build per process). "born" remains selectable.
#   DRY_RUN=1      print every step instead of executing (the local dry run handed to the referee)
#   PERF_SELFTEST=1  run the script's own negative controls (no checkout, build or measurement) and exit 0 when every
#                  control gives its expected class (registered as HADRONIC::perf_job_selftest)
#
# Step budgets (ceilings; the estimate and its basis are in the round-2 report §7.3). Sum of the ceilings with the
# decided PERF_QK_SOURCE=fit: 3020 s, margin 580 s (with born: 3680 s, the double-precision F8 step at risk).
# Priority order = execution order; a step that no longer fits is SKIPPED and the job ends incomplete_coverage.
#   0  checkout                                                                  120 s
#   1  build single (entity.xc, bh_check, bh_perf, bh_reduction_timing, the eight F8 executables)     780 s
#   2  M1 + M2 + M5 (+ the M5 restart runs on the end-loaded populations, F8)    600 s (fit; 1260 s with born)
#   3  M6 stride at PERF_N_M3 (event, none, poisson, budget; n = 1, 4, 16)        240 s
#      M3 variants at PERF_N_M3 (matched work)                                   180 s
#   4  M4 reducer vs atomics vs per-team, equal boundaries                        90 s
#   5  profiles: nsys + ncu per configuration (event: production, const, float; continuous: production)
#      with content assertions, and the RNG-pool measurement                     350 s
#   6  F8 single: FRAMEWORK::f8_tracers + the stock sort/deposit/checkpoint/spectra entries      90 s
#   7  F8 double: build of the eight F8 test targets                                           480 s
#      + run, only when that build exited 0 AND produced its complete inventory    90 s
#
# Outcome (Codex 1's F2.2, C1): every step is recorded in logs/steps.tsv (name, kind, budget, seconds, exit, class)
#   and the job ends with status.json at the run ROOT and the exit code of its overall class:
#     complete                  every step ran and passed                                  exit 0
#     scientific_check_failure  an EXECUTED check failed (ledger identity, the M6 event 4 s.e. cooling check, an M3
#                               identity, an F8 assertion): the program exited 1 and printed a line starting with
#                               "FAIL: "                                                                     exit 10
#     infrastructure_failure    checkout/build gate, incomplete binary inventory, timeout (124), kill (137), exception
#                               (2, signals), missing binary (126/127), a CTest entry Not Run / Timeout / Exception,
#                               an MPI launcher failure, missing or malformed timer data, missing profile content,
#                               a refused configuration: any non-zero exit other than (1 + a "FAIL: " line)       exit 20
#     incomplete_coverage       a step was SKIPPED for lack of time or BLOCKED by a failed gate            exit 30
#   Precedence of the overall class: infrastructure > incomplete > scientific > complete; the three flags are
#   written separately so nothing is hidden: a step classed infrastructure_failure whose log ALSO carries "FAIL: "
#   lines (executed checks that failed before the infrastructure fault) gets a second row <step>-scientific.
#   Independent steps continue after a failure; nothing is retried.
set -uo pipefail
DRY_RUN="${DRY_RUN:-0}"
ENTITY_REPO="${ENTITY_REPO:-git@github.com:mancininilay/entity.git}"
PERF_N="${PERF_N:-262144 1048576 4194304 16777216}"
PERF_N_M3="${PERF_N_M3:-4194304}"
export PERF_REPEATS="${PERF_REPEATS:-3}" PERF_REPEATS_TOP="${PERF_REPEATS_TOP:-2}" PERF_WARMUP="${PERF_WARMUP:-50}" PERF_MEASURED="${PERF_MEASURED:-500}" PERF_DIAG="${PERF_DIAG:-1}"
export PERF_QK_SOURCE="${PERF_QK_SOURCE:-fit}"
JOB="${SLURM_JOB_ID:-dryrun-$$}"
ROOT="${PERF_ROOT:-$HOME/ceph/entity-validation/p2-perf-$JOB}"
SRC="$ROOT/src"; BLD="$ROOT/build-single"; BLD2="$ROOT/build-double"; OUT="$ROOT/out"; LOG="$ROOT/logs"
F8_TOOL="$SRC/tests/hadronic/perf/rusty/f8_ctest.py"
T_START=$(date +%s)
BUDGET_S=$((60 * 60))
GATE_OK=0          # 1 once the checkout and the single build gate passed
ANY_INFRA=0; ANY_SCI=0; ANY_SKIP=0

# F8 (Codex 2 §F8, Codex 1 F3/C1): the tracer executable tests/framework/f8_tracers.cpp (sort, next-step deposition, split,
# checkpoint round trip, 1D/2D/3D, enumerated assertions) plus the stock entries that exist in a flat-deposit build
# (particles_sort, deposit, checkpoint-data, checkpoint-extents, spectra, injectors_global_weights, reduced_stats_weights;
# sort_by_key and deposit_tiled exist only with tiled_deposit=ON, which is not this build: see the report's proposal)
F8_TARGETS="test-framework-f8_tracers.xc test-framework-particles_sort.xc test-kernels-deposit.xc test-framework-checkpoint-data.xc test-framework-checkpoint-extents.xc test-framework-spectra.xc test-kernels-injectors_global_weights.xc test-kernels-reduced_stats_weights.xc"
F8_NAMES="FRAMEWORK::f8_tracers FRAMEWORK::particles_sort KERNELS::deposit FRAMEWORK::checkpoint-data FRAMEWORK::checkpoint-extents FRAMEWORK::spectra KERNELS::injectors_global_weights KERNELS::reduced_stats_weights"
f8_binaries() { # f8_binaries <build dir>: the complete inventory of the eight F8 executables of one build
  local b="$1"
  echo "$b/tests/framework/test-framework-f8_tracers.xc $b/tests/framework/test-framework-particles_sort.xc" \
       "$b/tests/kernels/test-kernels-deposit.xc $b/tests/framework/test-framework-checkpoint-data.xc" \
       "$b/tests/framework/test-framework-checkpoint-extents.xc $b/tests/framework/test-framework-spectra.xc" \
       "$b/tests/kernels/test-kernels-injectors_global_weights.xc $b/tests/kernels/test-kernels-reduced_stats_weights.xc"
}

say() { printf '[%s +%ss] %s\n' "$(date -u +%H:%M:%S)" "$(( $(date +%s) - T_START ))" "$*"; }
remaining() { echo $(( BUDGET_S - ( $(date +%s) - T_START ) )); }
record() { # record <name> <kind> <budget> <seconds> <exit> <class>
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$LOG/steps.tsv"
  case "$6" in infrastructure_failure) ANY_INFRA=1 ;; scientific_check_failure) ANY_SCI=1 ;; skipped|blocked) ANY_SKIP=1 ;; esac
}
# run <step name> <kind: gate|infra|sci> <budget seconds> <command...>
#   skipped when the remaining time is below the budget (class skipped); blocked when the gate has not passed
#   (class blocked); otherwise run under timeout and classified: exit 0 -> ok; for kind sci, exit 1 with a line
#   starting with "FAIL: " in the log -> scientific_check_failure (an executed check failed); any other non-zero exit
#   (timeout 124, kill 137, exception 2, missing binary 126/127, the programs' own infrastructure exits 3 and 4) ->
#   infrastructure_failure, plus a <name>-scientific row when the log also carries "FAIL: " lines. Returns the exit code.
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
  elif [ "$kind" = "sci" ] && [ $rc -eq 1 ] && grep -q '^FAIL: ' "$LOG/$name.log"; then cls=scientific_check_failure
  else cls=infrastructure_failure; fi
  say "END $name: exit $rc ($cls, $(( $(date +%s) - t0 )) s)"
  record "$name" "$kind" "$budget" $(( $(date +%s) - t0 )) "$rc" "$cls"
  if [ "$cls" = "infrastructure_failure" ] && [ "$kind" = "sci" ] && grep -q '^FAIL: ' "$LOG/$name.log"; then
    say "$name: the log also carries executed check failures (FAIL lines): recorded as $name-scientific"
    record "$name-scientific" "$kind" 0 0 "$rc" scientific_check_failure
  fi
  return $rc
}
# require_files <step name> <file...>: an infrastructure failure (row <step>-inventory) when any listed file is missing,
# empty or not executable
require_files() {
  local name="$1"; shift
  local f missing=0
  for f in "$@"; do { [ -s "$f" ] && [ -x "$f" ]; } || { say "MISSING for $name: $f"; missing=1; }; done
  [ $missing -eq 0 ] || record "$name-inventory" infra 0 0 "-" infrastructure_failure
  return $missing
}
# build_complete <build step> <its exit code> <binary...>: 0 only when the build command exited 0 AND the precision's
# complete binary inventory exists (Codex 1's C1: a run is gated on its own build's successful completion, never on a
# single surviving executable)
build_complete() {
  local name="$1" brc="$2"; shift 2
  [ "$brc" = "0" ] || return 1
  require_files "$name" "$@"
}
write_status() { # sets OVERALL and OVERALL_RC and writes status.json at the run root
  if [ "$ANY_INFRA" -eq 1 ]; then OVERALL=infrastructure_failure; OVERALL_RC=20
  elif [ "$ANY_SKIP" -eq 1 ]; then OVERALL=incomplete_coverage; OVERALL_RC=30
  elif [ "$ANY_SCI" -eq 1 ]; then OVERALL=scientific_check_failure; OVERALL_RC=10
  else OVERALL=complete; OVERALL_RC=0; fi
  {
    printf '{\n  "job": "%s",\n  "commit": "%s",\n  "root": "%s",\n  "dry_run": %s,\n' "$JOB" "${ENTITY_COMMIT:-}" "$ROOT" "$DRY_RUN"
    printf '  "overall": "%s",\n  "any_infrastructure_failure": %s,\n  "any_scientific_check_failure": %s,\n  "any_skipped_or_blocked": %s,\n' \
      "$OVERALL" "$ANY_INFRA" "$ANY_SCI" "$ANY_SKIP"
    printf '  "elapsed_s": %s,\n  "steps": [\n' "$(( $(date +%s) - T_START ))"
    awk -F'\t' 'NR > 1 { printf ",\n" } { printf "    {\"name\": \"%s\", \"kind\": \"%s\", \"budget_s\": %s, \"seconds\": %s, \"exit\": \"%s\", \"class\": \"%s\"}", $1, $2, $3, $4, $5, $6 }' "$LOG/steps.tsv"
    printf '\n  ]\n}\n'
  } > "$ROOT/status.json"
}
finish() {
  write_status
  say "DONE: overall $OVERALL (exit $OVERALL_RC); steps:"; cat "$LOG/steps.tsv"
  if [ "$DRY_RUN" != "1" ]; then
    ( cd "$ROOT" && find out logs -type f | sort > "$LOG/files.txt" && sha256sum "$LOG/commit.txt" "$ROOT/status.json" "$OUT"/m1/*.csv > "$LOG/SHA256SUMS" 2>/dev/null ) || true
  fi
  exit $OVERALL_RC
}

# ---- PERF_SELFTEST=1: the negative controls of Codex 1's F2.2 and C1 on this script's own functions and on the F8 tool
#      (no checkout, build or measurement; the CTest controls are real CTest invocations on generated test files, the
#      launchers are mocks). Every control has an expected class; the self-test exits 0 only when all of them match.
if [ "${PERF_SELFTEST:-0}" = "1" ]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"; F8_TOOL="$HERE/f8_ctest.py"
  TMPBASE="${TMPDIR:-/tmp}"; TMPBASE="${TMPBASE%/}"   # no doubled slash: CTest normalises the paths it reports
  ROOT="${PERF_ROOT:-$(mktemp -d "$TMPBASE/p2perf-selftest.XXXXXX")/root}"; LOG="$ROOT/logs"; OUT="$ROOT/out"; MOCK="$ROOT/mock"
  mkdir -p "$LOG" "$OUT" "$OUT/m1" "$MOCK"; : > "$LOG/steps.tsv"; : > "$LOG/commit.txt"
  GATE_OK=1; N_CHECK=0; N_BAD=0
  expect_rc() { # expect_rc <what> <got> <expected>
    N_CHECK=$((N_CHECK + 1))
    if [ "$2" = "$3" ]; then echo "SELFTEST ok: $1 returned $2"; else echo "SELFTEST MISMATCH: $1 returned $2, expected $3"; N_BAD=$((N_BAD + 1)); fi
  }
  # mock launchers: mpirun drops "-n N" and runs the command once; mpiexec fails as Open MPI does without enough slots
  printf '#!/bin/bash\n[ "$1" = "-n" ] && shift 2\nexec "$@"\n' > "$MOCK/mpirun"
  printf '#!/bin/bash\necho "There are not enough slots available in the system to satisfy the 2"\necho "slots that were requested by the application:"\nexit 1\n' > "$MOCK/mpiexec"
  printf '#!/bin/bash\necho "fake test"\n' > "$MOCK/fake-test.xc"
  chmod +x "$MOCK/mpirun" "$MOCK/mpiexec" "$MOCK/fake-test.xc"
  # ctest_dir <dir> <name> <command...>: a CTest directory with one registered test (CMake bracket quoting)
  ctest_dir() {
    local d="$1" n="$2" a line; shift 2
    mkdir -p "$d"; line="add_test([=[$n]=]"
    for a in "$@"; do line="$line [=[$a]=]"; done
    printf '%s)\n' "$line" > "$d/CTestTestfile.cmake"
  }
  f8_probe() { # f8_probe <probe name> <entry name> <command...>: one F8 entry through the real tool and the real CTest
    local p="$1" n="$2"; shift 2
    ctest_dir "$ROOT/$p" "$n" "$@"
    run "$p" sci 60 python3 "$F8_TOOL" --build "$ROOT/$p" --dest "$ROOT/$p/plan" --launcher "$MOCK/mpirun -n 1" --timeout 2 "$n"
    echo "$p returned $?"
  }
  # the script's own classification (F2.2)
  run probe-ok sci 60 /usr/bin/true;                          echo "probe-ok returned $?"
  run probe-injected-failure sci 60 /usr/bin/false;           echo "probe-injected-failure returned $? (exit 1 without a FAIL line)"
  run probe-sci-fail sci 60 bash -c 'echo "FAIL: injected scientific check"; exit 1'; echo "probe-sci-fail returned $?"
  run probe-ctest-banner sci 60 bash -c 'echo "The following tests FAILED:"; exit 1'; echo "probe-ctest-banner returned $? (CTest's FAILED banner is not an executed check)"
  run probe-mixed sci 60 bash -c 'echo "FAIL: injected scientific check"; echo "INFRA: injected fault"; exit 4'; echo "probe-mixed returned $?"
  run probe-timeout infra 1 sleep 5;                          echo "probe-timeout returned $?"
  BUDGET_S=0; run probe-skipped sci 60 /usr/bin/true;         echo "probe-skipped returned $?"; BUDGET_S=3600
  GATE_OK=0; run probe-blocked sci 60 /usr/bin/true;          echo "probe-blocked returned $?"; GATE_OK=1
  require_files probe-inventory "$ROOT/does-not-exist";       expect_rc "require_files (missing file)" $? 1
  echo x > "$ROOT/not-executable"
  require_files probe-noexec "$ROOT/not-executable";          expect_rc "require_files (not executable)" $? 1
  # C1: each precision gated on its own build's success AND its complete inventory
  build_complete probe-build-ok 0 "$MOCK/mpirun" "$MOCK/fake-test.xc";      expect_rc "build_complete (exit 0, inventory complete)" $? 0
  build_complete probe-build-failed 1 "$MOCK/mpirun" "$MOCK/fake-test.xc";  expect_rc "build_complete (build exit 1, one target present)" $? 1
  build_complete probe-build-partial 0 "$MOCK/fake-test.xc" "$ROOT/missing-target.xc"; expect_rc "build_complete (exit 0, inventory incomplete)" $? 1
  build_complete probe-build-skipped 98 "$MOCK/fake-test.xc";               expect_rc "build_complete (build skipped)" $? 1
  # C1: the F8 entries classified from CTest's structured result (Codex 1's missing-executable control and its relatives)
  f8_probe probe-f8-pass FRAMEWORK::f8_tracers /bin/sh -c 'echo "PASS: S1 1D: npart per path"; echo "f8_tracers summary: 33 passed, 0 failed"; exit 0'
  f8_probe probe-f8-notrun FRAMEWORK::f8_tracers "$ROOT/does-not-exist/not-built.xc"
  f8_probe probe-f8-timeout FRAMEWORK::f8_tracers /bin/sh -c 'sleep 10'
  f8_probe probe-f8-segv FRAMEWORK::f8_tracers /bin/sh -c 'kill -SEGV $$'
  f8_probe probe-f8-exception FRAMEWORK::f8_tracers /bin/sh -c 'echo "EXCEPTION: f8_tracers: exception: out of memory"; exit 2'
  f8_probe probe-f8-no-assertion FRAMEWORK::f8_tracers /bin/sh -c 'echo "something went wrong"; exit 1'
  f8_probe probe-f8-assertion FRAMEWORK::f8_tracers /bin/sh -c 'echo "FAIL: S3 1D: alive id set exact"; echo "f8_tracers summary: 32 passed, 1 failed"; exit 1'
  f8_probe probe-f8-stock-fail KERNELS::deposit /bin/sh -c 'echo "Error: deposit mismatch" >&2; exit 1'
  f8_probe probe-f8-mpi-slots FRAMEWORK::particles_sort "$MOCK/mpiexec" -n 2 "$MOCK/fake-test.xc"
  ctest_dir "$ROOT/probe-f8-unregistered" KERNELS::something_else /bin/sh -c 'exit 0'
  run probe-f8-unregistered sci 60 python3 "$F8_TOOL" --build "$ROOT/probe-f8-unregistered" --dest "$ROOT/probe-f8-unregistered/plan" --launcher "$MOCK/mpirun -n 1" FRAMEWORK::f8_tracers
  echo "probe-f8-unregistered returned $?"
  # the launcher plan: the direct command prefixed by the launcher, the two-rank command left as it is
  N_CHECK=$((N_CHECK + 1))
  if grep -q "\[=\[$MOCK/mpirun\]=\] \[=\[-n\]=\] \[=\[1\]=\] \[=\[/bin/sh\]=\]" "$ROOT/probe-f8-pass/plan/CTestTestfile.cmake" \
     && grep -q "\[=\[$MOCK/mpiexec\]=\] \[=\[-n\]=\] \[=\[2\]=\]" "$ROOT/probe-f8-mpi-slots/plan/CTestTestfile.cmake" \
     && ! grep '^add_test' "$ROOT/probe-f8-mpi-slots/plan/CTestTestfile.cmake" | grep -q "mpirun"; then
    echo "SELFTEST ok: launcher plan (direct command wrapped in mpirun -n 1, two-rank command unchanged)"
  else echo "SELFTEST MISMATCH: launcher plan"; N_BAD=$((N_BAD + 1)); fi
  # the expected class of every recorded row
  while read -r name cls; do
    N_CHECK=$((N_CHECK + 1))
    got=$(awk -F'\t' -v n="$name" '$1 == n { c = $6 } END { print c }' "$LOG/steps.tsv")
    if [ "$got" = "$cls" ]; then echo "SELFTEST ok: $name -> $got"; else echo "SELFTEST MISMATCH: $name -> '$got', expected $cls"; N_BAD=$((N_BAD + 1)); fi
  done <<'EXPECT'
probe-ok ok
probe-injected-failure infrastructure_failure
probe-sci-fail scientific_check_failure
probe-ctest-banner infrastructure_failure
probe-mixed infrastructure_failure
probe-mixed-scientific scientific_check_failure
probe-timeout infrastructure_failure
probe-skipped skipped
probe-blocked blocked
probe-inventory-inventory infrastructure_failure
probe-noexec-inventory infrastructure_failure
probe-build-partial-inventory infrastructure_failure
probe-f8-pass ok
probe-f8-notrun infrastructure_failure
probe-f8-timeout infrastructure_failure
probe-f8-segv infrastructure_failure
probe-f8-exception infrastructure_failure
probe-f8-no-assertion infrastructure_failure
probe-f8-assertion scientific_check_failure
probe-f8-stock-fail scientific_check_failure
probe-f8-mpi-slots infrastructure_failure
probe-f8-unregistered infrastructure_failure
EXPECT
  DRY_RUN=1
  write_status
  N_CHECK=$((N_CHECK + 1))
  if [ "$OVERALL" = "infrastructure_failure" ] && [ "$OVERALL_RC" = "20" ] && [ "$ANY_SCI" = "1" ] && [ "$ANY_SKIP" = "1" ]; then
    echo "SELFTEST ok: status.json overall infrastructure_failure (20) with the scientific and skipped/blocked flags set"
  else echo "SELFTEST MISMATCH: status overall $OVERALL ($OVERALL_RC), flags infra $ANY_INFRA sci $ANY_SCI skip $ANY_SKIP"; N_BAD=$((N_BAD + 1)); fi
  echo "---- steps.tsv"; cat "$LOG/steps.tsv"
  echo "SELFTEST summary: $N_CHECK checks, $N_BAD mismatches (root $ROOT)"
  [ "$N_BAD" -eq 0 ] && exit 0
  exit 1
fi

ENTITY_COMMIT="${ENTITY_COMMIT:?set ENTITY_COMMIT to the merged head}"

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
#      build command's success AND the complete inventory of every binary the later steps need (C1: the eight F8
#      executables included)
ENTITY_CMAKE=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DTESTS=ON -Dpgen=bh_proto -Dmpi=ON -Doutput=ON -Dgpu_aware_mpi=OFF
              -DKokkos_DIR="${Kokkos_DIR:-/unset}" -Dadios2_DIR="${adios2_DIR:-/unset}")
SINGLE_TARGETS="entity.xc bh_check test-kernels-bh_perf.xc test-kernels-bh_reduction_timing.xc $F8_TARGETS"
# shellcheck disable=SC2207
SINGLE_BINARIES=("$BLD/src/entity.xc" "$BLD/tests/hadronic/bh_check" "$BLD/tests/kernels/test-kernels-bh_perf.xc"
                 "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc" $(f8_binaries "$BLD"))
DOUBLE_BINARIES=($(f8_binaries "$BLD2"))
run step1-build gate 780 bash -c "cmake -S '$SRC' -B '$BLD' ${ENTITY_CMAKE[*]} -Dprecision=single && cmake --build '$BLD' -j ${SLURM_CPUS_PER_TASK:-8} --target $SINGLE_TARGETS"
rc=$?
if [ "$DRY_RUN" != "1" ] && [ "$GATE_OK" -eq 1 ]; then
  if ! build_complete step1-build "$rc" "${SINGLE_BINARIES[@]}"; then
    say "build gate failed (exit $rc or an incomplete inventory of ${#SINGLE_BINARIES[@]} binaries): every measurement is BLOCKED"; GATE_OK=0
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

# ---- step 5: profiles for M3 (Codex 1 F6): one capture window PER CONFIGURATION (`bh_perf variants --mode M --kernel K`
#      runs that configuration's 20 + 100 launches alone), N fixed at PERF_N_M3 and asserted from the program's own
#      "BH_PERF_N =" line, every profile checked for content: nsys must contain the NVTX ranges of the launch, the finish
#      readback and the restore/reseed phases (nvtx_sum report) and ncu must have captured the requested kernels (CUDA
#      function names of the Kokkos launches contain the functor type: regex BetheHeitler on the demangled name;
#      --kernel-name-base demangled). The RNG-pool share is measured in isolation (`bh_perf rngpool`, the attribution
#      method of the README). The step is an infrastructure failure when a profile lacks the expected content.
profile_one() { # profile_one <mode> <kernel>
  local m="$1" k="$2" tag="m3-$1-$2" rc=0 n
  env BH_PERF_N="$PERF_N_M3" BH_QK_SOURCE=born OMP_NUM_THREADS=1 nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$OUT/nsys-$tag" \
    "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants --mode "$m" --kernel "$k" > "$OUT/nsys-$tag.log" 2>&1 || rc=$?
  grep -q "BH_PERF_N = $PERF_N_M3 " "$OUT/nsys-$tag.log" || { echo "profile $tag: BH_PERF_N is not $PERF_N_M3"; return 4; }
  nsys stats --report nvtx_sum --format csv -o "$OUT/nsys-$tag" "$OUT/nsys-$tag.nsys-rep" > "$OUT/nsys-$tag-stats.log" 2>&1 || rc=$?
  for r in bh_perf_restore_and_reseed bh_perf_finish_readback; do
    grep -q "$r" "$OUT/nsys-$tag"_nvtx_sum.csv 2>/dev/null || { echo "profile $tag: NVTX range $r missing from the nsys report"; return 4; }
  done
  grep -q "bh_perf_.*_$m\|bh_perf_production_$m\|bh_perf_variant_${k}_$m" "$OUT/nsys-$tag"_nvtx_sum.csv || { echo "profile $tag: the launch range of ($m, $k) is missing"; return 4; }
  env BH_PERF_N="$PERF_N_M3" BH_QK_SOURCE=born OMP_NUM_THREADS=1 ncu --kernel-name-base demangled --kernel-name regex:BetheHeitler \
    --launch-skip 20 --launch-count 12 --set basic --force-overwrite -o "$OUT/ncu-$tag" \
    "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants --mode "$m" --kernel "$k" > "$OUT/ncu-$tag.log" 2>&1 || rc=$?
  n=$(ncu --import "$OUT/ncu-$tag.ncu-rep" --csv --page raw 2>/dev/null | tail -n +2 | grep -c "BetheHeitler")
  [ "$n" -ge 12 ] || { echo "profile $tag: ncu captured $n BetheHeitler kernel launches, expected 12"; return 4; }
  echo "profile $tag: nsys ranges present, ncu captured $n launches"
  return $rc
}
export -f profile_one; export OUT BLD PERF_N_M3
run step5-profile-event-production infra 100 bash -c "profile_one event production"
run step5-profile-event-const infra 70 bash -c "profile_one event const"
run step5-profile-event-float infra 70 bash -c "profile_one event float"
run step5-profile-continuous-production infra 70 bash -c "profile_one continuous production"
run step5-rngpool sci 40 env BH_PERF_N="$PERF_N_M3" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" rngpool

# ---- step 6: Codex 2's F8 GPU tracers, single precision (Codex 1's C1): tests/hadronic/perf/rusty/f8_ctest.py runs the
#      eight named entries one by one through Rusty's validated launcher plan (the direct commands, f8_tracers among them,
#      whose GlobalInitialize calls MPI_Init in this MPI build, wrapped in `mpirun -n 1`; the stock two-rank commands kept)
#      and classifies each from CTest's JUnit result: Not Run / Timeout / Exception / an MPI launcher failure are
#      infrastructure, an executed assertion failure is scientific; a missing entry is the selection gate. The two-rank
#      entries run inside this one-task allocation with Open MPI's oversubscription allowed (OMPI_MCA_rmaps_base_oversubscribe:
#      a launcher setting for the 2-rank stock tests, not a resource change: still one node, one task's CPUs, one GPU)
run step6-f8-single sci 90 env OMPI_MCA_rmaps_base_oversubscribe=1 python3 "$F8_TOOL" --build "$BLD" --dest "$OUT/f8-single" \
  --launcher "mpirun -n 1" --timeout 80 $F8_NAMES

# ---- step 7: F8 in double precision: the eight test targets only (the libraries they need are the bulk of the build),
#      then the same entries, ONLY when this build exited 0 and produced the complete inventory of the eight executables
#      (C1); the step runs last so that the registered measurements come first, and it is SKIPPED (status
#      incomplete_coverage) if the hour does not hold it
run step7-f8-double-build infra 480 bash -c "cmake -S '$SRC' -B '$BLD2' ${ENTITY_CMAKE[*]} -Dprecision=double && cmake --build '$BLD2' -j ${SLURM_CPUS_PER_TASK:-8} --target $F8_TARGETS"
rc=$?
if [ "$DRY_RUN" = "1" ] || build_complete step7-f8-double-build "$rc" "${DOUBLE_BINARIES[@]}"; then
  run step7-f8-double sci 90 env OMPI_MCA_rmaps_base_oversubscribe=1 python3 "$F8_TOOL" --build "$BLD2" --dest "$OUT/f8-double" \
    --launcher "mpirun -n 1" --timeout 80 $F8_NAMES
elif [ "$rc" = "98" ]; then
  say "step7-f8-double SKIPPED: its build was skipped for time"; record step7-f8-double sci 90 0 "-" skipped
else
  say "step7-f8-double BLOCKED: the double build did not complete (exit $rc) or its inventory of ${#DOUBLE_BINARIES[@]} executables is incomplete"
  record step7-f8-double sci 90 0 "-" blocked
fi

finish
