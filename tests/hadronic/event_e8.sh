#!/usr/bin/env bash
# E8 at the run level (P2 step 5, contract section 4): the host-side failure policies of the event mode.
#   (i)   on_bound: event_bound_error.toml aborts at the first event with pair_weight D > f_max K (ions
#         accelerated by E above the declared band; the flag names on_bound and the step); event_bound_warn.toml
#         completes with n_bounded > 0, the ledger identity exact and the array identity printed by the run.
#         Round 2 (F1, 2026-10-30): these two inputs are refused by the production binary at the a_max
#         refusal (checked here, exit non-zero and the message names a_max); the policy runs use the
#         test-only binary entity_bh_testing.xc (5th argument) with BH_TESTING_BYPASS_AMAX=1 in the
#         environment, and the log must carry the BH_TESTING banner and the SKIPPED line.
#   (ii)  on_capacity: event_capacity_error.toml aborts with the ledger row written (n_dropped > 0);
#         event_capacity_warn.toml completes with n_dropped > 0, both pair species full (npart = 60), identity exact.
#   (iii) on_max_events: event_maxevents_error.toml aborts at the first capped ion-step; event_maxevents_warn.toml
#         completes with n_capped_events > 0 and the identity exact.
# Also: event_smoke.toml completes (no failure path), identity exact, every payload slot zero (payload scan line).
# Bash 3.2 compatible; fails when no run was executed.
# Usage: event_e8.sh <entity.xc> <bh_check> <inputs> <workdir> <entity_bh_testing.xc>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"; TEST_EXE="$5"
n_run=0
if [ -z "$TEST_EXE" ] || [ ! -x "$TEST_EXE" ]; then
  echo "FAIL: event_e8.sh: the test-only binary entity_bh_testing.xc (5th argument) is missing: '$TEST_EXE'"
  exit 1
fi

expect_abort() {
  # expect_abort <name> <key in the abort message> <min_step> <max_step>
  local name="$1" key="$2" smin="$3" smax="$4"
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"; rc=$?; n_run=$((n_run + 1))
  if [ $rc -eq 0 ]; then
    echo "FAIL: $name: the run did not abort (exit 0)"; report_fail=1
  else
    echo "PASS: $name: the run aborted (exit $rc)"
    "$CHECK" abort --log "$WD/$name/run.log" --key "$key" --step_min "$smin" --step_max "$smax" || report_fail=1
    grep -q "bh_proto: failure on rank 0: $key" "$WD/$name/run.log" \
      && echo "PASS: $name: the gathered failure line of rank 0 names $key" \
      || { echo "FAIL: $name: no gathered failure line naming $key"; report_fail=1; }
    grep -q "every rank wrote its ledger row; the run aborts" "$WD/$name/run.log" \
      && echo "PASS: $name: the abort message states that every rank wrote its ledger row" \
      || { echo "FAIL: $name: the abort message is missing"; report_fail=1; }
  fi
}

expect_complete() {
  # expect_complete <name> <bh_check ledger options...>
  local name="$1"; shift
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"; rc=$?; n_run=$((n_run + 1))
  check_rc $rc "$name: the run completes"
  if [ $rc -eq 0 ]; then
    "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" "$@" || report_fail=1
    grep -h "bh_proto: array identity" "$WD/$name/run.log" | cut -c1-400
    grep -q "bh_proto: array identity species 1 .*PASS\|bh_proto: array identity species 1 : .* : ok" "$WD/$name/run.log" \
      || grep -h "bh_proto: array identity" "$WD/$name/run.log" | grep -q -v "FAIL" \
      && echo "PASS: $name: the array identity line of the run reports no failure" \
      || { echo "FAIL: $name: the array identity line reports a failure or is missing"; report_fail=1; }
    grep -h "payload scan" "$WD/$name/run.log" | cut -c1-200
  fi
}

# the smoke run: no failure path
expect_complete event_smoke --expect_dropped 0 --expect_bounded 0 --expect_capped_events 0 --min_spawned 100
# (i) on_bound. First the F1 check: the production binary refuses both bound inputs at the a_max refusal
for name in event_bound_error event_bound_warn; do
  run_entity "$EXE" "$INP/$name.toml" "$WD/production_$name"; rc=$?; n_run=$((n_run + 1))
  if [ $rc -eq 0 ]; then
    echo "FAIL: production entity.xc accepted $name.toml (exit 0): the a_max refusal is reachable from the input"; report_fail=1
  elif grep -q "exceeds a_max" "$WD/production_$name/run.log"; then
    echo "PASS: production entity.xc refuses $name.toml (exit $rc) at the a_max refusal"
  else
    echo "FAIL: production entity.xc refused $name.toml (exit $rc) but not at the a_max refusal"; report_fail=1
  fi
  if grep -q "BH_TESTING build" "$WD/production_$name/run.log"; then
    echo "FAIL: the production binary prints the BH_TESTING banner"; report_fail=1
  fi
done
# then the policy runs on the test-only binary with the bypass in the environment
PROD_EXE="$EXE"; EXE="$TEST_EXE"; export BH_TESTING_BYPASS_AMAX=1
expect_abort event_bound_error "on_bound" 1 200
expect_complete event_bound_warn --min_bounded 1 --min_spawned 100
unset BH_TESTING_BYPASS_AMAX; EXE="$PROD_EXE"
for name in event_bound_error event_bound_warn; do
  grep -q "bh_proto: BH_TESTING build" "$WD/$name/run.log" && grep -q "a_max is SKIPPED" "$WD/$name/run.log" \
    && echo "PASS: $name: the test-only binary announced itself and the skipped refusal" \
    || { echo "FAIL: $name: BH_TESTING banner or SKIPPED line missing"; report_fail=1; }
done
grep -h "on_bound" "$WD/event_bound_warn/run.log" | head -2 | cut -c1-300
# (ii) on_capacity
expect_abort event_capacity_error "on_capacity" 1 200
"$CHECK" ledger --file "$WD/event_capacity_error/hadronic_ledger_1.txt" --expect_dropped 1 || report_fail=1
expect_complete event_capacity_warn --expect_dropped 1 --expect_npart 60
# (iii) on_max_events
expect_abort event_maxevents_error "on_max_events" 1 200
expect_complete event_maxevents_warn --min_spawned 100
"$CHECK" ledger --file "$WD/event_maxevents_warn/hadronic_ledger_1.txt" --min_bounded 0 || report_fail=1
capped=$(grep -v '^#' "$WD/event_maxevents_warn/hadronic_ledger_1.txt" | tail -1 | awk '{print $32}')
if [ -n "$capped" ] && [ "$capped" -gt 0 ] 2>/dev/null; then
  echo "PASS: event_maxevents_warn: n_capped_events = $capped > 0"
else
  echo "FAIL: event_maxevents_warn: n_capped_events = '$capped' (expected > 0)"; report_fail=1
fi

if [ $n_run -eq 0 ]; then
  echo "FAIL: event_e8.sh executed no run"; report_fail=1
fi
echo "event_e8.sh summary: $n_run runs executed, report_fail = $report_fail"
[ $report_fail -eq 0 ] && [ $n_run -gt 0 ]
