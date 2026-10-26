#!/usr/bin/env bash
# T12 (engine level): pair capacity exhausted at step ~1000 of 2000 (pair maxnpart = 60 for about
# 120 pairs due). on_capacity = error: the run aborts at that step with a message naming the step and
# the counts, the ledger row written before the abort has n_dropped > 0. on_capacity = warn: the run
# continues to the end with the identity exact, n_dropped > 0, both species full (npart = 60).
# Bash 3.2 compatible. Usage: t12.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
run_entity "$EXE" "$INP/t12_capacity_error.toml" "$WD/error"; rc=$?; n_run=$((n_run + 1))
if [ $rc -eq 0 ]; then
  echo "FAIL: T12 on_capacity = error: the run did not abort (exit 0)"; report_fail=1
else
  echo "PASS: T12 on_capacity = error: the run aborted (exit $rc)"
  "$CHECK" abort --log "$WD/error/run.log" --key "on_capacity: pair capacity exhausted" --step_min 600 --step_max 1400 || report_fail=1
  "$CHECK" ledger --file "$WD/error/hadronic_ledger_1.txt" --expect_dropped 1 --expect_npart 60 || report_fail=1
  # C-MPI-2 (round 3): the failure is acted upon at the common point at the end of the stage: the log carries
  # the gathered failure line of rank 0, the abort message names the policy, and the ledger row of the abort
  # step is written exactly once (no local write before the collective one)
  grep -q "bh_proto: failure on rank 0: on_capacity" "$WD/error/run.log" \
    && echo "PASS: C-MPI-2: the gathered failure line of rank 0 is in the log" \
    || { echo "FAIL: C-MPI-2: no gathered failure line in the log"; report_fail=1; }
  grep -q "every rank wrote its ledger row; the run aborts" "$WD/error/run.log" \
    && echo "PASS: C-MPI-2: the abort message states that every rank wrote its ledger row" \
    || { echo "FAIL: C-MPI-2: the abort message is missing"; report_fail=1; }
  last_step=$(grep -v '^#' "$WD/error/hadronic_ledger_1.txt" | tail -1 | awk '{print $1}')
  n_last=$(grep -v '^#' "$WD/error/hadronic_ledger_1.txt" | awk -v s="$last_step" '$1 == s' | wc -l | tr -d ' ')
  if [ "$n_last" -eq 1 ]; then
    echo "PASS: C-MPI-2: the ledger row of the abort step $last_step is written exactly once"
  else
    echo "FAIL: C-MPI-2: the ledger row of the abort step $last_step appears $n_last times"; report_fail=1
  fi
fi
run_entity "$EXE" "$INP/t12_capacity_warn.toml" "$WD/warn"; rc=$?; n_run=$((n_run + 1))
check_rc $rc "T12 on_capacity = warn: the run completes"
if [ $rc -eq 0 ]; then
  grep -h "on_capacity" "$WD/warn/run.log" | head -2 | cut -c1-300
  "$CHECK" ledger --file "$WD/warn/hadronic_ledger_1.txt" --expect_dropped 1 --expect_npart 60 || report_fail=1
  grep -h "bh_proto: array identity" "$WD/warn/run.log" | cut -c1-400
fi
if [ $n_run -eq 0 ]; then echo "FAIL: t12.sh executed no run"; report_fail=1; fi
echo "t12.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
