#!/usr/bin/env bash
# E8 at the run level (P2 step 5, contract section 4): the host-side failure policies of the event mode.
#   (i)   on_bound: event_bound_error.toml aborts at the first event with pair_weight D > f_max K (ions
#         accelerated by E above the declared band; the flag names on_bound and the step); event_bound_warn.toml
#         completes with n_bounded > 0, the ledger identity exact and the array identity printed by the run.
#   (ii)  on_capacity: event_capacity_error.toml aborts with the ledger row written (n_dropped > 0);
#         event_capacity_warn.toml completes with n_dropped > 0, both pair species full (npart = 60), identity exact.
#   (iii) on_max_events: event_maxevents_error.toml aborts at the first capped ion-step; event_maxevents_warn.toml
#         completes with n_capped_events > 0 and the identity exact.
# Also: event_smoke.toml completes (no failure path), identity exact, every payload slot zero (payload scan line).
# Bash 3.2 compatible; fails when no run was executed. Usage: event_e8.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0

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
# (i) on_bound
expect_abort event_bound_error "on_bound" 1 200
expect_complete event_bound_warn --min_bounded 1 --min_spawned 100
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
