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
# Parameters (environment, with sbatch --export=ALL,ENTITY_COMMIT=...,PERF_PART=...):
#   ENTITY_COMMIT  the merged head to measure (event mode + Born tables)          [required]
#   PERF_PART      the one-hour part of the campaign this job runs                [required]
#                  (round 4, from B7398318's measured costs: the registered campaign is ~5 A100-hours, so it is split into
#                  one-hour parts, never shrunk; `perf_budget.py --plan` lists the 15 parts and checks they cover the 143
#                  engine processes exactly once; report §7.9)
#                  kernel: M6 stride, M3 variants, M4, the four nsys profiles, RNG proxy, F8 single and double precision
#                  engine: one slice of M1/M2/M5 through perf_driver.sh, one step per N:
#                    PERF_N              the N of the slice (one or more of 262144 1048576 4194304 16777216)  [required]
#                    PERF_REPEAT_FIRST, PERF_REPEAT_LAST  [1 and the registered count: 3, or 2 at 16777216]
#                    PERF_M1 [1], PERF_MODES [none poisson budget event], PERF_VARIANTS [warm cold],
#                    PERF_M5_SETS [pic poisson budget] (empty: no M5)
#                  The job REFUSES to start (exit 7, before the run root exists) on a missing or unknown part, an N
#                  without a measured basis, or a slice whose ceilings plus the gate exceed 80 % of the hour
#   PERF_NCU       [0] 1 adds the four ncu captures to the kernel part (OPTIONAL: Rusty's driver refused the counters in
#                  B7398318, B3; off by default, recorded as not_requested, which sets no flag); the nsys steps always run
#   ENTITY_REPO    fork URL                               [git@github.com:mancininilay/entity.git]
#   PERF_ROOT      run root (must NOT exist)              [~/ceph/entity-validation/p2-perf-<jobid>]
#   PERF_N_M3      ion count for M3 (nsys/ncu), M4, M6    FIXED at 4194304 by this authorisation (Codex 1's F6/C4): any
#                  other value, explicit or inherited from the submitting shell through --export=ALL, makes the job
#                  refuse to start (exit 6, before the run root exists); the steps use the constant PERF_N_FIXED
#   PERF_WARMUP [50: the F3 cut]  PERF_MEASURED [500]  PERF_DIAG [1: every measured step is summed, F4]
#   PERF_QK_SOURCE [fit] the q_k source of the ENGINE timing runs. DECIDED by Nilay on 2026-10-31 ("fit for engine
#                  runs", decisions.md): the step-1 tables for the engine processes, because every bh_proto process
#                  would otherwise build the Born node table at startup (7.2 s per process on one A100-node thread,
#                  B7398318), while the code path and the table layout are the same. Label of these rows (Codex 1's
#                  C4): "engine runs on the step-1 (fit) tables: same code and layout as Born; the dynamic work (events,
#                  spawns, substeps) follows the table values, so equal stage cost is expected, not identical by
#                  construction". Since round 4 (B1) the driver refuses a template or input without the selector and a
#                  run whose startup line names another source. The kernel-level groups (M3, M6, the profiles) run on
#                  the Born tables (BH_QK_SOURCE=born, one table build per process). "born" remains selectable.
#   DRY_RUN=1      print every step instead of executing (the local dry run handed to the referee)
#   PERF_SELFTEST=1  run the script's own negative controls (no checkout, build or measurement) and exit 0 when every
#                  control gives its expected class (registered as HADRONIC::perf_job_selftest)
#
# Step budgets (ceilings, round 4: derived from the measured costs of B7398318 and A7397866 by rusty/perf_budget.py, whose
# table is embedded below and compared with the script's output by the self-test). Every part starts with the gate:
#   0  checkout                                                         measured 25 s     ceiling 120 s
#   1  build, single precision, the part's targets (a subset of B7398318's 12-target build: 654 s at -j 8)    780 s
# kernel part (sum of the ceilings 2650 s; 2850 s with PERF_NCU=1):
#   3  M6 stride (96 s) 150; M3 variants (62 s) 100;   4  M4 (3 s) 30
#   5  nsys per configuration x4 (each <= 32 s with the failed ncu attempt) 50 each; ncu x4 OPTIONAL 50 each; RNG (2 s) 30
#   6  F8 single (27 s; the injector aborted early, B5) 120
#   7  F8 double build (A7397866: the 27-target double build 1003 s at -j 12; B7398318: 49/69 steps at its 480 s
#      timeout) 1000, then F8 double 120, only when that build exited 0 AND produced its complete inventory
# engine part: one step per N, the ceiling being the sum over the slice's processes of 1.25 x the per-process model
#   (startup 3.0 s at 2^18 plus 1 us per further ion, the 550-step stepping measured at 2^18 plus a PLANNING slope of 2 ns
#   per particle-step, the end-of-run text dumps of bh_proto measured at 12-17 us per particle and linear in N). The
#   dumps dominate at large N: ~4.2 of the ~5.0 hours of the campaign (report §7.9).
# Priority order = execution order; a step that no longer fits is SKIPPED and the job ends incomplete_coverage.
#
# Outcome (Codex 1's F2.2, C1): every step is recorded in logs/steps.tsv (name, kind, budget, seconds, exit, class)
#   and the job ends with status.json at the run ROOT and the exit code of its overall class:
#     complete                  every step ran and passed                                  exit 0
#     scientific_check_failure  an EXECUTED check failed (ledger identity, the M6 event 4 s.e. cooling check, an M3
#                               identity, an F8 assertion): the program exited 1 and printed a line starting with
#                               "FAIL: "                                                                     exit 10
#     infrastructure_failure    checkout/build gate, incomplete binary inventory, timeout (124), kill (137), exception
#                               (2, signals), missing binary (126/127), a CTest entry Not Run / Timeout / Exception,
#                               an F8 entry that exited through a caught exception (native exit 2) or without a
#                               parsed check result (f8_ctest.py; Codex 1's reclosure R1),
#                               an MPI launcher failure, missing or malformed timer data, missing profile content,
#                               any non-zero exit other than (1 + a "FAIL: " line)                         exit 20
#     (refused before start: an existing run root, exit 5; PERF_N_M3 != 4194304, exit 6; a part or slice that is missing,
#      unknown or does not fit its hour, exit 7; no status.json then)
#     incomplete_coverage       a step was SKIPPED for lack of time or BLOCKED by a failed gate            exit 30
#   Precedence of the overall class: infrastructure > incomplete > scientific > complete; the three flags are
#   written separately so nothing is hidden: a step classed infrastructure_failure whose log ALSO carries "FAIL: "
#   lines (executed checks that failed before the infrastructure fault) gets a second row <step>-scientific.
#   Independent steps continue after a failure; nothing is retried.
set -uo pipefail
DRY_RUN="${DRY_RUN:-0}"
ENTITY_REPO="${ENTITY_REPO:-git@github.com:mancininilay/entity.git}"
PERF_PART="${PERF_PART:-}"
PERF_N="${PERF_N:-}"
PERF_NCU="${PERF_NCU:-0}"
PERF_M1="${PERF_M1:-1}"; PERF_MODES="${PERF_MODES:-none poisson budget event}"; PERF_VARIANTS="${PERF_VARIANTS:-warm cold}"
PERF_M5_SETS="${PERF_M5_SETS-pic poisson budget}"   # set but empty: no M5 in this slice
PERF_REPEAT_FIRST="${PERF_REPEAT_FIRST:-1}"
PERF_N_FIXED=4194304   # M3, M4, M6 and the profiles at the authorised N (Codex 1's F6/C4); PERF_N_M3 may only repeat it
readonly PERF_N_FIXED
PERF_N_M3="${PERF_N_M3:-$PERF_N_FIXED}"
export PERF_REPEATS="${PERF_REPEATS:-3}" PERF_REPEATS_TOP="${PERF_REPEATS_TOP:-2}" PERF_WARMUP="${PERF_WARMUP:-50}" PERF_MEASURED="${PERF_MEASURED:-500}" PERF_DIAG="${PERF_DIAG:-1}"
export PERF_QK_SOURCE="${PERF_QK_SOURCE:-fit}"
JOB="${SLURM_JOB_ID:-dryrun-$$}"
ROOT="${PERF_ROOT:-$HOME/ceph/entity-validation/p2-perf-$JOB}"
SRC="$ROOT/src"; BLD="$ROOT/build-single"; BLD2="$ROOT/build-double"; OUT="$ROOT/out"; LOG="$ROOT/logs"
F8_TOOL="$SRC/tests/hadronic/perf/rusty/f8_ctest.py"
PROFILE_TOOL="$SRC/tests/hadronic/perf/rusty/profile_check.py"
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

# ---- round 4: per-process ceilings [s] of the engine runs (N tag mode variant ceiling), the output of
#      `perf_budget.py --emit-table` (1.25 x the model of B7398318's measured costs, fit tables, planning slope 2 ns per
#      particle-step); the self-test fails when this copy and the script's output differ
CEILING_TABLE="$(cat <<'TABLE'
262144 m1 budget cold 16
262144 m1 budget warm 19
262144 m1 event cold 16
262144 m1 event warm 23
262144 m1 none cold 17
262144 m1 none warm 18
262144 m1 poisson cold 17
262144 m1 poisson warm 26
262144 m5budget budget warm 22
262144 m5budget_pic budget warm 22
262144 m5pic none warm 20
262144 m5poisson poisson warm 32
262144 m5poisson_pic poisson warm 32
1048576 m1 budget cold 32
1048576 m1 budget warm 43
1048576 m1 event cold 32
1048576 m1 event warm 56
1048576 m1 none cold 33
1048576 m1 none warm 39
1048576 m1 poisson cold 35
1048576 m1 poisson warm 66
1048576 m5budget budget warm 45
1048576 m5budget_pic budget warm 45
1048576 m5pic none warm 36
1048576 m5poisson poisson warm 73
1048576 m5poisson_pic poisson warm 73
4194304 m1 budget cold 95
4194304 m1 budget warm 136
4194304 m1 event cold 95
4194304 m1 event warm 188
4194304 m1 none cold 100
4194304 m1 none warm 123
4194304 m1 poisson cold 105
4194304 m1 poisson warm 227
4194304 m5budget budget warm 134
4194304 m5budget_pic budget warm 134
4194304 m5pic none warm 101
4194304 m5poisson poisson warm 235
4194304 m5poisson_pic poisson warm 239
16777216 m1 budget cold 345
16777216 m1 budget warm 511
16777216 m1 event cold 349
16777216 m1 event warm 716
16777216 m1 none cold 366
16777216 m1 none warm 458
16777216 m1 poisson cold 385
16777216 m1 poisson warm 869
16777216 m5budget budget warm 493
16777216 m5budget_pic budget warm 493
16777216 m5pic none warm 359
16777216 m5poisson poisson warm 885
16777216 m5poisson_pic poisson warm 900
TABLE
)"
GATE_S=900          # checkout 120 + single build 780
PART_CAP_S=2880     # 80 % of the hour: kill grace, environment, write_status
KERNEL_S=2650       # the kernel part's ceilings (perf_budget.py), 2850 with PERF_NCU=1
# registered_reps <N>: the registered repeat count (3; 2 at 2^24, the F3 cut)
registered_reps() { [ "$1" = "16777216" ] && echo 2 || echo 3; }
# slice_ceiling <N> <first> <last>: the sum of the per-process ceilings of the engine slice at N (PERF_M1, PERF_MODES,
# PERF_VARIANTS, PERF_M5_SETS), the driver's process list; prints nothing when N has no row in the table
slice_ceiling() {
  local n="$1" first="$2" last="$3" r mode var g t list=""
  echo "$CEILING_TABLE" | awk -v n="$n" '$1 == n { f = 1 } END { exit !f }' || return 1
  r="$first"
  while [ "$r" -le "$last" ]; do
    if [ "$PERF_M1" = "1" ]; then
      for mode in $PERF_MODES; do for var in $PERF_VARIANTS; do list="$list m1:$mode:$var"; done; done
    fi
    for g in $PERF_M5_SETS; do
      case "$g" in
        pic) list="$list m5pic:none:warm" ;;
        poisson) list="$list m5poisson:poisson:warm m5poisson_pic:poisson:warm" ;;
        budget) list="$list m5budget:budget:warm m5budget_pic:budget:warm" ;;
      esac
    done
    r=$((r + 1))
  done
  echo "$CEILING_TABLE" | awk -v n="$n" -v list="$list" '
    BEGIN { m = split(list, want, " "); for (i = 1; i <= m; i++) cnt[want[i]]++ }
    $1 == n { key = $2 ":" $3 ":" $4; if (key in cnt) { s += cnt[key] * $5; used[key] = 1 } }
    END { for (k in cnt) if (!(k in used)) { print "missing " k; exit 1 } printf "%d\n", s }'
}
# part_refusal: empty when PERF_PART and its slice are valid and fit PART_CAP_S, else the reason (checked before the root)
part_refusal() {
  local n tot=0 c last
  case "$PERF_PART" in
    kernel)
      tot=$KERNEL_S; [ "$PERF_NCU" = "1" ] && tot=$((KERNEL_S + 200))
      [ "$tot" -le "$PART_CAP_S" ] || echo "the kernel part's ceilings $tot s exceed $PART_CAP_S s" ;;
    engine)
      [ -n "$PERF_N" ] || { echo "PERF_PART=engine needs PERF_N (a slice from perf_budget.py --plan)"; return; }
      for g in $PERF_M5_SETS; do case "$g" in pic|poisson|budget) ;; *) echo "unknown PERF_M5_SETS group '$g'"; return ;; esac; done
      [ "$PERF_M1" = "1" ] || [ -n "$PERF_M5_SETS" ] || { echo "the slice is empty (PERF_M1=0 and no PERF_M5_SETS)"; return; }
      tot=$GATE_S
      for n in $PERF_N; do
        last="${PERF_REPEAT_LAST:-$(registered_reps "$n")}"
        [ "$PERF_REPEAT_FIRST" -ge 1 ] && [ "$last" -ge "$PERF_REPEAT_FIRST" ] && [ "$last" -le "$(registered_reps "$n")" ] \
          || { echo "repeats $PERF_REPEAT_FIRST-$last outside the registered 1-$(registered_reps "$n") at N = $n"; return; }
        c=$(slice_ceiling "$n" "$PERF_REPEAT_FIRST" "$last") || { echo "N = $n has no measured basis in the ceiling table"; return; }
        tot=$((tot + c))
      done
      [ "$tot" -le "$PART_CAP_S" ] || echo "the engine slice needs $tot s of ceilings (gate included), more than $PART_CAP_S s: split it (perf_budget.py --plan)" ;;
    "") echo "PERF_PART is not set (kernel or engine; perf_budget.py --plan lists the parts)" ;;
    *) echo "unknown PERF_PART '$PERF_PART' (kernel or engine)" ;;
  esac
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
    ( cd "$ROOT" && find out logs -type f | sort > "$LOG/files.txt" && sha256sum "$LOG/commit.txt" "$ROOT/status.json" "$OUT"/m1/*/*.csv > "$LOG/SHA256SUMS" 2>/dev/null ) || true
  fi
  exit $OVERALL_RC
}

# profile_nsys, profile_ncu <mode> <kernel>: nsys, ncu of ONE (mode, kernel) configuration (`bh_perf variants --mode M --kernel K` runs
# that configuration's 20 + 100 launches alone) at the FIXED N, every profile checked for content by profile_check.py
# (Codex 1's F6/C4): the program's own log of each profiled run must say BH_PERF_N = PERF_N_FIXED (the constant, never
# the overridable variable), the configuration filter (M, K) and one configuration of 120 launches; the nsys nvtx_sum
# report must hold the restore/reseed and finish-readback ranges and the launch range of (M, K) and no other
# configuration's; the ncu raw page must hold at least 12 UNIQUE launch IDs (not metric rows), every one of the
# configuration's functor type. Returns 4 (infrastructure) when any content check fails, else the profilers' exit code.
# Round 4: split into profile_nsys (always run) and profile_ncu (OPTIONAL, PERF_NCU=1: Rusty's driver refused the counters
# in B7398318, B3), the same commands and content checks as before.
profile_nsys() {
  local m="$1" k="$2" tag="m3-$1-$2" rc=0
  env BH_PERF_N="$PERF_N_FIXED" BH_QK_SOURCE=born OMP_NUM_THREADS=1 nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$OUT/nsys-$tag" \
    "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants --mode "$m" --kernel "$k" > "$OUT/nsys-$tag.log" 2>&1 || rc=$?
  python3 "$PROFILE_TOOL" log "$OUT/nsys-$tag.log" --n "$PERF_N_FIXED" --mode "$m" --kernel "$k" || return 4
  nsys stats --report nvtx_sum --format csv -o "$OUT/nsys-$tag" "$OUT/nsys-$tag.nsys-rep" > "$OUT/nsys-$tag-stats.log" 2>&1 || rc=$?
  python3 "$PROFILE_TOOL" nsys "$OUT/nsys-$tag"_nvtx_sum.csv --mode "$m" --kernel "$k" || return 4
  echo "profile nsys $tag: content verified (fixed N $PERF_N_FIXED, configuration ($m, $k), the three ranges)"
  return $rc
}
profile_ncu() {
  local m="$1" k="$2" tag="m3-$1-$2" rc=0
  env BH_PERF_N="$PERF_N_FIXED" BH_QK_SOURCE=born OMP_NUM_THREADS=1 ncu --kernel-name-base demangled --kernel-name regex:BetheHeitler \
    --launch-skip 20 --launch-count 12 --set basic --force-overwrite -o "$OUT/ncu-$tag" \
    "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants --mode "$m" --kernel "$k" > "$OUT/ncu-$tag.log" 2>&1 || rc=$?
  python3 "$PROFILE_TOOL" log "$OUT/ncu-$tag.log" --n "$PERF_N_FIXED" --mode "$m" --kernel "$k" || return 4
  ncu --import "$OUT/ncu-$tag.ncu-rep" --csv --page raw > "$OUT/ncu-$tag-raw.csv" 2> "$OUT/ncu-$tag-import.log" || rc=$?
  python3 "$PROFILE_TOOL" ncu "$OUT/ncu-$tag-raw.csv" --kernel "$k" --min 12 || return 4
  echo "profile ncu $tag: content verified (fixed N $PERF_N_FIXED, configuration ($m, $k), unique launches counted)"
  return $rc
}

# ---- PERF_SELFTEST=1: the negative controls of Codex 1's F2.2 and C1 on this script's own functions and on the F8 tool
#      (no checkout, build or measurement; the CTest controls are real CTest invocations on generated test files, the
#      launchers are mocks). Every control has an expected class; the self-test exits 0 only when all of them match.
if [ "${PERF_SELFTEST:-0}" = "1" ]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"; F8_TOOL="$HERE/f8_ctest.py"; PROFILE_TOOL="$HERE/profile_check.py"
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
  f8_probe probe-f8-stock-fail KERNELS::deposit /bin/sh -c 'echo "DepositCurrents_kernel::SumDivJ != 0" >&2; exit 1'
  f8_probe probe-f8-mpi-slots FRAMEWORK::particles_sort "$MOCK/mpiexec" -n 2 "$MOCK/fake-test.xc"
  # R1 (Codex 1's reclosure at 49cd6aa5): scientific needs positive evidence of an executed failed check, read with the
  # native exit value (CTest Test.xml); a caught exception (exit 2) or an exit without a parsed check result is
  # infrastructure. Codex 1's control: the injector-weights test with a referee-injected exception before any check
  # (its exact output and exit 2), and the same for reduced_stats_weights; the exit-without-results control (exit 1, the
  # program's no-check branch, summary 0 passed 0 failed); a stock exit 1 with unrecognised output; the positive controls
  # (a weights FAIL verdict, a raise::Error check of particles_sort at its own source line); an exception after a failed
  # verdict (infrastructure plus the -scientific row)
  f8_probe probe-f8-weights-exception KERNELS::injectors_global_weights /bin/sh -c 'echo "injectors_global_weights: real_t = float, execution space = OpenMP"; echo "injectors_global_weights: exception: REFEREE injected infrastructure exception" >&2; exit 2'
  f8_probe probe-f8-rsw-exception KERNELS::reduced_stats_weights /bin/sh -c 'echo "reduced_stats_weights: real_t = float, execution space = OpenMP"; echo "reduced_stats_weights: exception: REFEREE injected infrastructure exception" >&2; exit 2'
  f8_probe probe-f8-no-results KERNELS::reduced_stats_weights /bin/sh -c 'echo "reduced_stats_weights: real_t = float, execution space = OpenMP"; echo "reduced_stats_weights summary: 0 passed, 0 failed"; exit 1'
  f8_probe probe-f8-stock-unparsed KERNELS::deposit /bin/sh -c 'echo "Error: deposit mismatch" >&2; exit 1'
  f8_probe probe-f8-weights-assertion KERNELS::injectors_global_weights /bin/sh -c 'echo "PASS: 1D weighted: stored weight : 1 (expected 1)"; echo "FAIL: 1D weighted: the untouched tail still holds 99 : 0 (expected 1)"; echo "injectors_global_weights summary: 1 passed, 1 failed"; exit 1'
  f8_probe probe-f8-raise-check FRAMEWORK::particles_sort /bin/sh -c 'echo "Error: wrong number of alive particles after sort /src/tests/framework/particles_sort.cpp:167 @ main" >&2; exit 1'
  f8_probe probe-f8-fail-then-exception KERNELS::reduced_stats_weights /bin/sh -c 'echo "FAIL: case 1: mean energy : 1.0e-02 (tolerance 1.0e-06)"; echo "reduced_stats_weights: exception: REFEREE injected after a check" >&2; exit 2'
  # Codex 1's control through the job's run() and write_status() alone (his observed overall was
  # scientific_check_failure, class 10, infrastructure flag 0): a separate root and fresh flags
  (
    ROOT="$ROOT/r1-status-only"; LOG="$ROOT/logs"; mkdir -p "$LOG"; : > "$LOG/steps.tsv"; ANY_INFRA=0; ANY_SCI=0; ANY_SKIP=0
    f8_probe probe-f8-weights-exception KERNELS::injectors_global_weights /bin/sh -c 'echo "injectors_global_weights: exception: REFEREE injected infrastructure exception" >&2; exit 2' > /dev/null
    DRY_RUN=1; write_status
    echo "$OVERALL $OVERALL_RC $ANY_INFRA $ANY_SCI" > "$ROOT/overall.txt"
  )
  expect_rc "R1 weights-exception control alone: overall, class, infrastructure flag, scientific flag" "$(cat "$ROOT/r1-status-only/overall.txt")" "infrastructure_failure 20 1 0"
  ctest_dir "$ROOT/probe-f8-unregistered" KERNELS::something_else /bin/sh -c 'exit 0'
  run probe-f8-unregistered sci 60 python3 "$F8_TOOL" --build "$ROOT/probe-f8-unregistered" --dest "$ROOT/probe-f8-unregistered/plan" --launcher "$MOCK/mpirun -n 1" FRAMEWORK::f8_tracers
  echo "probe-f8-unregistered returned $?"
  # C4: the profile content checks on SYNTHETIC profiler output (mock nsys / ncu / bh_perf, NOT CUDA profiles): Codex 1's
  # one-launch / 24-metric-row control and wrong-N control, a wrong kernel type, another configuration's range, and a
  # correct synthetic profile; then the PERF_N_M3 refusal of the script itself, the override inherited from the environment
  mkdir -p "$MOCK/bld/tests/kernels" "$MOCK/bin"
  cat > "$MOCK/bld/tests/kernels/test-kernels-bh_perf.xc" <<'PROG'
#!/bin/bash
# mock bh_perf (self-test only): the lines the profile checks read; MOCK_PRINTED_N replaces the printed N
m=""; k=""; while [ $# -gt 0 ]; do case "$1" in --mode) m="$2"; shift ;; --kernel) k="$2"; shift ;; esac; shift; done
echo "bh_perf: BH_PERF_N = ${MOCK_PRINTED_N:-$BH_PERF_N} (synthetic)"
echo "bh_perf: configuration filter: mode '$m', kernel '$k'"
echo "(perf) M3 launches per configuration: 20 warm-up + 100 timed = 120 parallel_reduce launches of the kernel, 1 configurations selected (synthetic)"
PROG
  cat > "$MOCK/bin/nsys" <<'NSYS'
#!/bin/bash
# mock nsys (self-test only): "profile ... -o X prog args" runs prog and writes X.nsys-rep; "stats ... -o X rep" writes X_nvtx_sum.csv
cmd="$1"; shift; o=""
if [ "$cmd" = "profile" ]; then
  while [ $# -gt 0 ]; do case "$1" in -o) o="$2"; shift 2 ;; --*) shift ;; *) break ;; esac; done
  echo synthetic > "$o.nsys-rep"; exec "$@"
fi
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && o="$2"; shift; done
cp "$MOCK_NVTX_CSV" "${o}_nvtx_sum.csv"
NSYS
  cat > "$MOCK/bin/ncu" <<'NCU'
#!/bin/bash
# mock ncu (self-test only): "--import rep --csv --page raw" prints MOCK_NCU_CSV; otherwise "... -o X prog args" runs prog
case " $* " in *" --import "*) cat "$MOCK_NCU_CSV"; exit 0 ;; esac
o=""
while [ $# -gt 0 ]; do case "$1" in -o) o="$2"; shift 2 ;; --kernel-name-base|--kernel-name|--launch-skip|--launch-count|--set) shift 2 ;; --*) shift ;; *) break ;; esac; done
echo synthetic > "$o.ncu-rep"; exec "$@"
NCU
  chmod +x "$MOCK/bld/tests/kernels/test-kernels-bh_perf.xc" "$MOCK/bin/nsys" "$MOCK/bin/ncu"
  # synthetic reports: nvtx_sum with the production event ranges (and a variant of it with a const range), ncu raw pages
  printf '"Time (%%)","Total Time (ns)","Instances","Range"\n"60.0","1000","120",":bh_perf_production_event"\n"20.0","300","120",":bh_perf_restore_and_reseed"\n"20.0","300","120",":bh_perf_finish_readback"\n' > "$MOCK/nvtx-ok.csv"
  { cat "$MOCK/nvtx-ok.csv"; printf '"1.0","10","120",":bh_perf_variant_const_event"\n'; } > "$MOCK/nvtx-other.csv"
  KP='void Kokkos::Impl::cuda_parallel_launch_local_memory<Kokkos::Impl::ParallelReduce<Kokkos::Impl::CombinedFunctorReducer<kernel::hadronic::BetheHeitler_kernel<metric::Minkowski<(Dimension)1>>, Kokkos::Impl::FunctorAnalysis<>::Reducer, void>, Kokkos::RangePolicy<>, Kokkos::Cuda>>(T1)'
  KC='void Kokkos::Impl::cuda_parallel_launch_local_memory<Kokkos::Impl::ParallelReduce<Kokkos::Impl::CombinedFunctorReducer<kernel::hadronic_perf::BetheHeitler_kernel_variant<metric::Minkowski<(Dimension)1>, kernel::hadronic_perf::PerfTablesConst>, void>, Kokkos::Cuda>>(T1)'
  { echo '"ID","Process ID","Process Name","Host Name","Kernel Name","Metric Name","Metric Value"'; i=0
    while [ $i -lt 12 ]; do for mm in a b c; do echo "\"$i\",\"1\",\"bh_perf\",\"h\",\"$KP\",\"$mm\",\"1\""; done; i=$((i + 1)); done; } > "$MOCK/ncu-ok.csv"
  { echo '"ID","Process ID","Process Name","Host Name","Kernel Name","Metric Name","Metric Value"'; i=0
    while [ $i -lt 24 ]; do echo "\"0\",\"1\",\"bh_perf\",\"h\",\"$KP\",\"metric$i\",\"1\""; i=$((i + 1)); done; } > "$MOCK/ncu-one-launch.csv"
  { echo '"ID","Process ID","Process Name","Host Name","Kernel Name","Metric Name","Metric Value"'; i=0
    while [ $i -lt 12 ]; do echo "\"$i\",\"1\",\"bh_perf\",\"h\",\"$KC\",\"a\",\"1\""; i=$((i + 1)); done; } > "$MOCK/ncu-wrong-kernel.csv"
  prof_probe() { # prof_probe <probe> <nvtx csv> <ncu csv> [MOCK_PRINTED_N]: profile_nsys then profile_ncu (the job's functions) on mock profilers
    local p="$1" save_out="$OUT" save_path="$PATH" prc
    mkdir -p "$ROOT/$p"
    export OUT="$ROOT/$p" BLD="$MOCK/bld" MOCK_NVTX_CSV="$2" MOCK_NCU_CSV="$3" MOCK_PRINTED_N="${4:-}" PERF_N_FIXED PROFILE_TOOL
    export -f profile_nsys profile_ncu
    export PATH="$MOCK/bin:$save_path"
    run "$p" infra 60 bash -c "profile_nsys event production && profile_ncu event production"; prc=$?
    export PATH="$save_path"; OUT="$save_out"
    echo "$p returned $prc"
  }
  prof_probe probe-prof-ok "$MOCK/nvtx-ok.csv" "$MOCK/ncu-ok.csv"
  prof_probe probe-prof-one-launch-24-rows "$MOCK/nvtx-ok.csv" "$MOCK/ncu-one-launch.csv"
  prof_probe probe-prof-wrong-n "$MOCK/nvtx-ok.csv" "$MOCK/ncu-ok.csv" 65536
  prof_probe probe-prof-wrong-kernel "$MOCK/nvtx-ok.csv" "$MOCK/ncu-wrong-kernel.csv"
  prof_probe probe-prof-other-range "$MOCK/nvtx-other.csv" "$MOCK/ncu-ok.csv"
  # C4: the script refuses PERF_N_M3 != 4194304 before creating its run root, also when the value is only inherited from the
  # environment (what sbatch --export=ALL forwards); the fixed value itself is accepted (dry run)
  env -u PERF_SELFTEST PERF_N_M3=65536 DRY_RUN=1 ENTITY_COMMIT=selftest PERF_ROOT="$ROOT/guard-65536" bash "$0" > "$LOG/guard-65536.log" 2>&1
  expect_rc "PERF_N_M3=65536 inherited from the environment: refusal" $? 6
  expect_rc "PERF_N_M3=65536: no run root created" "$([ -e "$ROOT/guard-65536" ] && echo exists || echo absent)" absent
  env -u PERF_SELFTEST PERF_N_M3=4194304 PERF_PART=kernel DRY_RUN=1 ENTITY_COMMIT=selftest PERF_ROOT="$ROOT/guard-fixed" bash "$0" > "$LOG/guard-fixed.log" 2>&1
  expect_rc "PERF_N_M3=4194304 (the fixed value): dry run" $? 0
  # B1 (round 4, Codex 1's readback of B7398318: the template had no @QK_SOURCE@ and the fit request never reached the
  # engine): the real driver and the real template, input generation and the selector check only (PERF_CHECK_INPUTS_ONLY,
  # no engine run); negative control: the same inputs with the template's qk_source line removed must be refused (exit 3).
  # The runtime half of B1 (the engine's startup line against the request) is HADRONIC::perf_driver_replay's.
  INP_REAL="$HERE/../inputs"; mkdir -p "$ROOT/inputs-no-placeholder"; cp "$INP_REAL"/*.toml "$ROOT/inputs-no-placeholder/"
  grep -v '^ *qk_source *=' "$INP_REAL/perf_template.toml" > "$ROOT/inputs-no-placeholder/perf_template.toml"
  for q in fit born; do
    env -u PERF_SELFTEST PERF_CHECK_INPUTS_ONLY=1 PERF_QK_SOURCE=$q bash "$HERE/../perf_driver.sh" /nonexistent /nonexistent "$INP_REAL" "$ROOT/selector-$q" > "$LOG/selector-$q.log" 2>&1
    expect_rc "B1 selector probe, real template, PERF_QK_SOURCE=$q" $? 0
    N_CHECK=$((N_CHECK + 1))
    if grep -Eq '^ *qk_source *= *"'"$q"'" *$' "$ROOT/selector-$q/selector_probe/input.toml"; then echo "SELFTEST ok: B1 probe input holds qk_source = \"$q\""
    else echo "SELFTEST MISMATCH: B1 probe input lacks qk_source = \"$q\""; N_BAD=$((N_BAD + 1)); fi
  done
  env -u PERF_SELFTEST PERF_CHECK_INPUTS_ONLY=1 PERF_QK_SOURCE=fit bash "$HERE/../perf_driver.sh" /nonexistent /nonexistent "$ROOT/inputs-no-placeholder" "$ROOT/selector-none" > "$LOG/selector-none.log" 2>&1
  expect_rc "B1 negative control: template without the placeholder refused" $? 3
  # Round 4, budgets from B7398318: the embedded ceiling table is the budget script's output; the plan is valid (every part
  # fits, coverage of the 143 processes exact); the job refuses a missing, unknown, basis-less or oversized part before
  # creating its root (exit 7) and accepts EVERY part of the plan (dry runs through its own guard); ncu not requested by
  # default, run with PERF_NCU=1; an engine part's step budget is its slice's sum of the table
  BUDGET_TOOL="$HERE/perf_budget.py"
  N_CHECK=$((N_CHECK + 1))
  if [ "$(python3 "$BUDGET_TOOL" --emit-table)" = "$CEILING_TABLE" ]; then echo "SELFTEST ok: the embedded ceiling table equals perf_budget.py --emit-table"
  else echo "SELFTEST MISMATCH: the embedded ceiling table differs from perf_budget.py --emit-table"; N_BAD=$((N_BAD + 1)); fi
  python3 "$BUDGET_TOOL" --plan > "$LOG/budget-plan.log" 2>&1
  expect_rc "perf_budget.py --plan (every part fits, exact coverage)" $? 0
  guard() { # guard <tag> <expected exit> <env...>: a dry run of this script; a refusal must leave no run root
    local tag="$1" ex="$2" grc; shift 2
    env -u PERF_SELFTEST -u PERF_PART -u PERF_N -u PERF_NCU -u PERF_M1 -u PERF_MODES -u PERF_VARIANTS -u PERF_M5_SETS \
      -u PERF_REPEAT_FIRST -u PERF_REPEAT_LAST DRY_RUN=1 ENTITY_COMMIT=selftest PERF_ROOT="$ROOT/guard-$tag" "$@" bash "$0" > "$LOG/guard-$tag.log" 2>&1
    grc=$?
    expect_rc "part guard $tag" $grc "$ex"
    if [ "$ex" = "7" ]; then expect_rc "part guard $tag: no run root" "$([ -e "$ROOT/guard-$tag" ] && echo exists || echo absent)" absent; fi
  }
  guard no-part 7
  guard unknown-part 7 PERF_PART=everything
  guard engine-no-n 7 PERF_PART=engine
  guard engine-no-basis 7 PERF_PART=engine PERF_N=123
  guard engine-full-2to24 7 PERF_PART=engine PERF_N=16777216
  guard engine-2to22-two-repeats 7 PERF_PART=engine PERF_N=4194304 PERF_REPEAT_LAST=2
  guard engine-repeat-beyond 7 PERF_PART=engine PERF_N=16777216 PERF_REPEAT_FIRST=3 PERF_REPEAT_LAST=3 PERF_M1=0 PERF_M5_SETS=pic
  guard engine-empty-slice 7 PERF_PART=engine PERF_N=262144 PERF_M1=0 PERF_M5_SETS=
  guard kernel-ncu-off 0 PERF_PART=kernel
  guard kernel-ncu-on 0 PERF_PART=kernel PERF_NCU=1
  N_CHECK=$((N_CHECK + 1))
  if [ "$(awk -F'\t' '$1 ~ /^step5-ncu-/ && $6 == "not_requested"' "$ROOT/guard-kernel-ncu-off/logs/steps.tsv" | wc -l | tr -d ' ')" = "4" ] \
     && [ "$(awk -F'\t' '$1 ~ /^step5-nsys-/ && $6 == "dry"' "$ROOT/guard-kernel-ncu-off/logs/steps.tsv" | wc -l | tr -d ' ')" = "4" ] \
     && [ "$(awk -F'\t' '$1 ~ /^step5-ncu-/ && $6 == "dry"' "$ROOT/guard-kernel-ncu-on/logs/steps.tsv" | wc -l | tr -d ' ')" = "4" ] \
     && grep -q '"overall": "complete"' "$ROOT/guard-kernel-ncu-off/status.json"; then
    echo "SELFTEST ok: kernel part: four nsys steps; four ncu steps not_requested by default (overall unaffected), run with PERF_NCU=1"
  else echo "SELFTEST MISMATCH: kernel part nsys/ncu rows"; N_BAD=$((N_BAD + 1)); fi
  guard engine-e03 0 PERF_PART=engine PERF_N=4194304 PERF_REPEAT_FIRST=1 PERF_REPEAT_LAST=1
  N_CHECK=$((N_CHECK + 1))
  b=$(awk -F'\t' '$1 == "step2-engine-N4194304" { print $3 }' "$ROOT/guard-engine-e03/logs/steps.tsv")
  if [ "$b" = "1912" ]; then echo "SELFTEST ok: engine part E03: step2-engine-N4194304 budget 1912 s (the slice's 13 process ceilings)"
  else echo "SELFTEST MISMATCH: engine part E03 budget '$b', expected 1912"; N_BAD=$((N_BAD + 1)); fi
  n_parts=0
  while IFS='|' read -r pname ppart pn pfirst plast pm1 pmodes pvars pm5; do
    n_parts=$((n_parts + 1))
    if [ "$ppart" = "kernel" ]; then guard "plan-$pname" 0 PERF_PART=kernel
    else guard "plan-$pname" 0 PERF_PART=engine PERF_N="$pn" PERF_REPEAT_FIRST="$pfirst" PERF_REPEAT_LAST="$plast" PERF_M1="$pm1" \
           PERF_MODES="${pmodes:-none}" PERF_VARIANTS="${pvars:-warm}" PERF_M5_SETS="$pm5"; fi
  done <<PLANPARTS
$(python3 "$BUDGET_TOOL" --plan-env)
PLANPARTS
  expect_rc "the plan's part count" "$n_parts" 15
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
probe-f8-weights-exception infrastructure_failure
probe-f8-rsw-exception infrastructure_failure
probe-f8-no-results infrastructure_failure
probe-f8-stock-unparsed infrastructure_failure
probe-f8-weights-assertion scientific_check_failure
probe-f8-raise-check scientific_check_failure
probe-f8-fail-then-exception infrastructure_failure
probe-f8-fail-then-exception-scientific scientific_check_failure
probe-f8-unregistered infrastructure_failure
probe-prof-ok ok
probe-prof-one-launch-24-rows infrastructure_failure
probe-prof-wrong-n infrastructure_failure
probe-prof-wrong-kernel infrastructure_failure
probe-prof-other-range infrastructure_failure
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

# ---- the fixed N of this authorisation (Codex 1's F6/C4): refuse to start on any other PERF_N_M3, explicit or inherited
#      through --export=ALL, before anything is created or run
if [ "$PERF_N_M3" != "$PERF_N_FIXED" ]; then
  echo "configuration refused: PERF_N_M3 = '$PERF_N_M3', but this authorisation fixes $PERF_N_FIXED for M3, M4, M6 and the profiles (unset it or set it to $PERF_N_FIXED)"
  exit 6
fi
# ---- round 4: the part and its slice (kernel, or one engine slice) must be valid and fit the hour, before anything exists
REFUSAL="$(part_refusal)"
if [ -n "$REFUSAL" ]; then echo "configuration refused: $REFUSAL"; exit 7; fi

# ---- the run root: exclusive creation (Codex 1's F2.3: an existing root would let the driver overwrite CSVs)
if [ -e "$ROOT" ]; then echo "run root $ROOT exists: refusing to reuse it (fresh root per job)"; exit 5; fi
mkdir "$ROOT" || { echo "cannot create the run root $ROOT"; exit 5; }
mkdir -p "$OUT" "$LOG"
: > "$LOG/steps.tsv"
say "p2-perf-a100 job $JOB on $(hostname); commit $ENTITY_COMMIT; part $PERF_PART; root $ROOT; dry run $DRY_RUN; budget $BUDGET_S s"
[ "$PERF_PART" = "engine" ] && say "engine slice: PERF_N '$PERF_N', repeats $PERF_REPEAT_FIRST-${PERF_REPEAT_LAST:-registered}, PERF_M1 $PERF_M1, PERF_MODES '$PERF_MODES', PERF_VARIANTS '$PERF_VARIANTS', PERF_M5_SETS '$PERF_M5_SETS', PERF_QK_SOURCE $PERF_QK_SOURCE"
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
# round 4: each part builds only its own targets, a subset of B7398318's 12-target build (654 s), so 780 s stays a ceiling
if [ "$PERF_PART" = "engine" ]; then
  SINGLE_TARGETS="entity.xc bh_check"
  SINGLE_BINARIES=("$BLD/src/entity.xc" "$BLD/tests/hadronic/bh_check")
else
  SINGLE_TARGETS="test-kernels-bh_perf.xc test-kernels-bh_reduction_timing.xc $F8_TARGETS"
  # shellcheck disable=SC2207
  SINGLE_BINARIES=("$BLD/tests/kernels/test-kernels-bh_perf.xc" "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc" $(f8_binaries "$BLD"))
fi
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

# ---- step 2 (engine part): M1, M2, M5 through the engine (Entity's timers), one step per N of the slice, each with the
#      ceiling of its own processes (slice_ceiling: the embedded table), its own work directory, CSV and medians; a step
#      that times out loses only its unfinished runs (the CSV is appended per run) and the next N still runs if it fits
if [ "$PERF_PART" = "engine" ]; then
  for n in $PERF_N; do
    last="${PERF_REPEAT_LAST:-$(registered_reps "$n")}"
    m5=1; [ -n "$PERF_M5_SETS" ] || m5=0
    run "step2-engine-N$n" sci "$(slice_ceiling "$n" "$PERF_REPEAT_FIRST" "$last")" env PERF_N="$n" PERF_REPEATS="$last" \
      PERF_REPEATS_TOP="$last" PERF_REPEAT_FIRST="$PERF_REPEAT_FIRST" PERF_REPEAT_LAST="$last" PERF_M1="$PERF_M1" \
      PERF_MODES="$PERF_MODES" PERF_VARIANTS="$PERF_VARIANTS" PERF_M5="$m5" PERF_M5_SETS="${PERF_M5_SETS:-pic}" \
      PERF_LABEL="a100-$JOB-$ENTITY_COMMIT" OMP_NUM_THREADS=1 \
      bash "$SRC/tests/hadronic/perf/perf_driver.sh" "$ROOT/entity-1rank.sh" "$BLD/tests/hadronic/bh_check" "$SRC/tests/hadronic/perf/inputs" "$OUT/m1/N$n"
  done
  finish
fi

# ---- step 3: M6 stride and M3 variants at the kernel level
run step3-stride sci 150 env BH_PERF_N="$PERF_N_FIXED" BH_QK_SOURCE=born OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" stride
run step3-variants sci 100 env BH_PERF_N="$PERF_N_FIXED" BH_QK_SOURCE=born OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" variants

# ---- step 4: M4 reducer vs atomics vs per-team scratch at the contract's N
run step4-reduction sci 30 env BH_REDUCTION_N="$PERF_N_FIXED" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_reduction_timing.xc"

# ---- step 5: profiles for M3 (Codex 1 F6): one capture window PER CONFIGURATION (`bh_perf variants --mode M --kernel K`
#      runs that configuration's 20 + 100 launches alone), N fixed at PERF_N_FIXED and asserted from the program's own
#      "BH_PERF_N =" line, every profile checked for content by profile_nsys / profile_ncu / profile_check.py (above): the configuration,
#      the NVTX ranges, at least 12 UNIQUE ncu launch IDs of the configuration's functor type (CUDA function names of the
#      Kokkos launches contain the functor type: regex BetheHeitler on the demangled name). The RNG-pool cost is measured
#      by an isolated benchmark (`bh_perf rngpool`: an isolated proxy, not a bound on the in-kernel share; README). The
#      step is an infrastructure failure when a profile lacks the expected content. Round 4: nsys per configuration always;
#      ncu per configuration only with PERF_NCU=1 (optional: Rusty's driver refused the counters in B7398318, B3), recorded
#      as not_requested otherwise (no flag set: by design, not a skip)
export -f profile_nsys profile_ncu; export OUT BLD PERF_N_FIXED PROFILE_TOOL
for cfg in "event production" "event const" "event float" "continuous production"; do
  set -- $cfg
  run "step5-nsys-$1-$2" infra 50 bash -c "profile_nsys $1 $2"
done
for cfg in "event production" "event const" "event float" "continuous production"; do
  set -- $cfg
  if [ "$PERF_NCU" = "1" ]; then run "step5-ncu-$1-$2" infra 50 bash -c "profile_ncu $1 $2"
  else say "step5-ncu-$1-$2 not requested (PERF_NCU=0, the default)"; record "step5-ncu-$1-$2" infra 50 0 "-" not_requested; fi
done
run step5-rngpool sci 30 env BH_PERF_N="$PERF_N_FIXED" OMP_NUM_THREADS=1 mpirun -n 1 "$BLD/tests/kernels/test-kernels-bh_perf.xc" rngpool

# ---- step 6: Codex 2's F8 GPU tracers, single precision (Codex 1's C1): tests/hadronic/perf/rusty/f8_ctest.py runs the
#      eight named entries one by one through Rusty's validated launcher plan (the direct commands, f8_tracers among them,
#      whose GlobalInitialize calls MPI_Init in this MPI build, wrapped in `mpirun -n 1`; the stock two-rank commands kept)
#      and classifies each from CTest's JUnit result: Not Run / Timeout / Exception / an MPI launcher failure are
#      infrastructure, an executed assertion failure is scientific; a missing entry is the selection gate. The two-rank
#      entries run inside this one-task allocation with Open MPI's oversubscription allowed (OMPI_MCA_rmaps_base_oversubscribe:
#      a launcher setting for the 2-rank stock tests, not a resource change: still one node, one task's CPUs, one GPU)
run step6-f8-single sci 120 env OMPI_MCA_rmaps_base_oversubscribe=1 python3 "$F8_TOOL" --build "$BLD" --dest "$OUT/f8-single" \
  --launcher "mpirun -n 1" --timeout 80 $F8_NAMES

# ---- step 7: F8 in double precision: the eight test targets only (the libraries they need are the bulk of the build),
#      then the same entries, ONLY when this build exited 0 and produced the complete inventory of the eight executables
#      (C1); the step runs last so that the registered measurements come first, and it is SKIPPED (status
#      incomplete_coverage) if the hour does not hold it
run step7-f8-double-build infra 1000 bash -c "cmake -S '$SRC' -B '$BLD2' ${ENTITY_CMAKE[*]} -Dprecision=double && cmake --build '$BLD2' -j ${SLURM_CPUS_PER_TASK:-8} --target $F8_TARGETS"
rc=$?
if [ "$DRY_RUN" = "1" ] || build_complete step7-f8-double-build "$rc" "${DOUBLE_BINARIES[@]}"; then
  run step7-f8-double sci 120 env OMPI_MCA_rmaps_base_oversubscribe=1 python3 "$F8_TOOL" --build "$BLD2" --dest "$OUT/f8-double" \
    --launcher "mpirun -n 1" --timeout 80 $F8_NAMES
elif [ "$rc" = "98" ]; then
  say "step7-f8-double SKIPPED: its build was skipped for time"; record step7-f8-double sci 120 0 "-" skipped
else
  say "step7-f8-double BLOCKED: the double build did not complete (exit $rc) or its inventory of ${#DOUBLE_BINARIES[@]} executables is incomplete"
  record step7-f8-double sci 120 0 "-" blocked
fi

finish
