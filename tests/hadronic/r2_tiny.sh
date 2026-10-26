#!/usr/bin/env bash
# R2 (round 2), engine level: pair_weight = 1e-20 with the poisson source gives lambda ~ 1e20 > 1e9 on every
# ion-step, which the sampler counts as exceeding the cap without converting lambda to int: exactly
# max_events_per_step = 8 events per ion per step (16 ions x 200 steps x 8 = 25600 = n_events = n_spawned),
# n_capped_events = 16 x 200 = 3200, no RNG draw for the capped steps beyond the per-event spectra, and the
# ledger identity holds. Also the UBSan diagnostic input (run the same script with a UBSan entity.xc).
# Bash 3.2 compatible. Usage: r2_tiny.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
run_entity "$EXE" "$INP/r2_pairweight_tiny.toml" "$WD/tiny"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "R2 run (pair_weight = 1e-20, poisson)"
if [ $rc -eq 0 ]; then
  "$CHECK" ledger --file "$WD/tiny/hadronic_ledger_1.txt" --expect_events 25600 --expect_capped_events 3200 --expect_npart 25600 || report_fail=1
  grep -c "on_max_events" "$WD/tiny/run.log" | sed 's/^/(diag) on_max_events warnings in the log: /'
  if grep -q "runtime error" "$WD/tiny/run.log"; then echo "FAIL: the sanitizer reported a runtime error"; report_fail=1; grep -m3 "runtime error" "$WD/tiny/run.log"; else echo "ok: no sanitizer runtime error in the log"; fi
fi
# C-MPI-2 (round 3): the same input with on_max_events = error aborts at the first step through the common
# point at the end of the stage: the ledger row of step 0 is written exactly once (n_capped_events = 16), the
# gathered failure line of rank 0 and the abort message are in the log
mkdir -p "$WD/inputs" || exit 2
sed 's/^  on_max_events *= *"warn"/  on_max_events        = "error"/' "$INP/r2_pairweight_tiny.toml" > "$WD/inputs/tiny_error.toml"
if [ "$(grep -c 'on_max_events *= *"error"' "$WD/inputs/tiny_error.toml")" -ne 1 ]; then
  echo "FAIL: the derived on_max_events = error input was not produced"; report_fail=1
else
  run_entity "$EXE" "$WD/inputs/tiny_error.toml" "$WD/tiny_error"; rc=$?; n_run=$((n_run + 1))
  if [ $rc -eq 0 ]; then
    echo "FAIL: on_max_events = error: the run did not abort (exit 0)"; report_fail=1
  else
    echo "PASS: on_max_events = error: the run aborted (exit $rc)"
    "$CHECK" abort --log "$WD/tiny_error/run.log" --key "on_max_events:" --step_min 0 --step_max 0 || report_fail=1
    "$CHECK" ledger --file "$WD/tiny_error/hadronic_ledger_1.txt" --expect_capped_events 16 --expect_events 128 || report_fail=1
    grep -q "bh_proto: failure on rank 0: on_max_events" "$WD/tiny_error/run.log" \
      && echo "PASS: C-MPI-2: the gathered failure line of rank 0 is in the log" \
      || { echo "FAIL: C-MPI-2: no gathered failure line in the log"; report_fail=1; }
    n_rows=$(grep -v '^#' "$WD/tiny_error/hadronic_ledger_1.txt" | wc -l | tr -d ' ')
    if [ "$n_rows" -eq 1 ]; then
      echo "PASS: C-MPI-2: exactly one ledger row (the abort step) was written"
    else
      echo "FAIL: C-MPI-2: $n_rows ledger rows written for an abort at step 0"; report_fail=1
    fi
  fi
fi
if [ $n_run -eq 0 ]; then echo "FAIL: r2_tiny.sh executed no run"; report_fail=1; fi
echo "r2_tiny.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
