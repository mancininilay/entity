#!/usr/bin/env bash
# T-exp (engine level, single rank, absorbing x boundaries): ions drifting into the walls with the budget
# source export their debt: E_exported equals the host-side sum w r over the dead-tagged ions taken before
# each launch (before the sort) to 1e-12, pending_discarded their sum w delta, the ledger identity holds
# with the exported term; t_latency = 0 is refused in this box. Bash 3.2 compatible.
# Usage: texp.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
run_entity "$EXE" "$INP/texp_absorb.toml" "$WD/absorb"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "T-exp run (absorbing walls)"
if [ $rc -eq 0 ]; then
  "$CHECK" texp --log "$WD/absorb/run.log" || report_fail=1
  "$CHECK" ledger --file "$WD/absorb/hadronic_ledger_1.txt" || report_fail=1
fi
run_entity "$EXE" "$INP/reject_tlatency.toml" "$WD/tlatency"; rc=$?; n_run=$((n_run + 1))
if [ $rc -eq 0 ]; then
  echo "FAIL: t_latency = 0 with absorbing boundaries was accepted"; report_fail=1
elif grep -q "t_latency" "$WD/tlatency/run.log"; then
  echo "PASS: t_latency = 0 with absorbing boundaries refused (exit $rc), message names t_latency"
else
  echo "FAIL: refused but the message does not name t_latency"; report_fail=1
fi
if [ $n_run -eq 0 ]; then echo "FAIL: texp.sh executed no run"; report_fail=1; fi
echo "texp.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
