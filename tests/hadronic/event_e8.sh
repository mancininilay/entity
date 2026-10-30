#!/usr/bin/env bash
# E8 at the run level (P2 step 5, contract section 4): the host-side failure policies of the event mode.
#   (i)   on_bound: event_bound_error.toml aborts at the first event with pair_weight D > f_max K (ions
#         accelerated by E above the declared band; the flag names on_bound and the step); event_bound_warn.toml
#         completes with n_bounded > 0, the ledger identity exact and the array identity printed by the run.
#         Round 2 (F1, 2026-10-30): these two inputs are refused by the production binary at the a_max
#         refusal (checked here, exit non-zero and the message names a_max); the policy runs use the
#         test-only binary entity_bh_testing.xc (5th argument) with BH_TESTING_BYPASS_AMAX=1 in the
#         environment, and the log must carry the BH_TESTING banner and the SKIPPED line.
#   (v)   revision 1.1 (F2): event_accel_error.toml / event_accel_warn.toml (Codex 1's production-accelerated
#         probe, ions accelerated out of the declared band): error aborts at step 0 (the first) with the on_bound flag and
#         n_bound_flagged = 16 (ledger column added by F4), no event; warn completes sub-cycled (n_substeps >= 112), n_bounded = 0, identity exact.
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
expect_complete event_smoke --expect_dropped 0 --expect_bounded 0 --expect_capped_events 0 --min_spawned 100 --expect_identity 1 --expect_bound_flagged 0
# revision 1.1 (F4): every counter of the reducer in the summary line, the sampler counters as explicit zeros (delta)
for key in n_events n_spawned n_dropped n_substeps n_bounded n_capped_loss n_closing n_capped n_capped_events n_above_table \
           n_latency n_processed n_applied n_rng n_k0_trigger n_closing_at_rest n_resolution n_alive n_bound_steps \
           n_bound_flagged n_attempted n_unexecuted n_sampler_attempts n_sampler_overflow; do
  grep "bh_proto: ledger species 1 :" "$WD/event_smoke/run.log" | grep -q " $key [0-9]" \
    || { echo "FAIL: event_smoke: the summary line lacks $key"; report_fail=1; }
done
grep "bh_proto: ledger species 1 :" "$WD/event_smoke/run.log" | grep -q " n_sampler_attempts 0 n_sampler_overflow 0 " \
  && echo "PASS: event_smoke: the summary carries every reducer counter, the sampler counters as explicit zeros" \
  || { echo "FAIL: event_smoke: sampler counters not printed as zeros in the delta summary"; report_fail=1; }
grep -q "bh_proto: restframe sampler species 1 : n_sampler_attempts 0 n_sampler_overflow 0 .*sampler not used" "$WD/event_smoke/run.log" \
  && echo "PASS: event_smoke: the sampler line is printed in the delta mode with zeros" \
  || { echo "FAIL: event_smoke: no sampler line in the delta mode"; report_fail=1; }
cols=$(grep -v '^#' "$WD/event_smoke/hadronic_ledger_1.txt" | tail -1 | awk '{print NF}')
[ "$cols" = "51" ] && echo "PASS: event_smoke: the ledger row has 51 columns" || { echo "FAIL: event_smoke: ledger row has $cols columns (expected 51)"; report_fail=1; }
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
expect_complete event_bound_warn --min_bounded 1 --min_spawned 100 --expect_identity 1
unset BH_TESTING_BYPASS_AMAX; EXE="$PROD_EXE"
for name in event_bound_error event_bound_warn; do
  grep -q "bh_proto: BH_TESTING build" "$WD/$name/run.log" && grep -q "a_max is SKIPPED" "$WD/$name/run.log" \
    && echo "PASS: $name: the test-only binary announced itself and the skipped refusal" \
    || { echo "FAIL: $name: BH_TESTING banner or SKIPPED line missing"; report_fail=1; }
done
grep -h "on_bound" "$WD/event_bound_warn/run.log" | head -2 | cut -c1-300
# (v) revision 1.1 (F2): the current-state condition on an admitted input whose ions leave the declared band
#     (Codex 1's production-accelerated probe): error aborts with the flag at step 1, warn sub-cycles
expect_abort event_accel_error "on_bound" 0 0
"$CHECK" ledger --file "$WD/event_accel_error/hadronic_ledger_1.txt" --expect_bounded 0 --expect_events 0 --expect_bound_flagged 16 --min_bound_steps 16 --expect_alive 16 --expect_identity 1 || report_fail=1
grep -q "R <D> dt > f_max K" "$WD/event_accel_error/run.log" \
  && echo "PASS: event_accel_error: the failure message names the current-state condition" \
  || { echo "FAIL: event_accel_error: the failure message does not name the current-state condition"; report_fail=1; }
expect_complete event_accel_warn --expect_bounded 0 --min_spawned 500 --min_substeps 112 --expect_bound_flagged 0 --min_bound_steps 16 --expect_alive 16 --expect_identity 1
grep -q "hazard sub-cycled" "$WD/event_accel_warn/run.log" \
  && echo "PASS: event_accel_warn: the run reports the sub-cycled hazard: $(grep -m1 "hazard sub-cycled" "$WD/event_accel_warn/run.log" | cut -c1-220)" \
  || { echo "FAIL: event_accel_warn: no sub-cycling report"; report_fail=1; }
# (ii) on_capacity
expect_abort event_capacity_error "on_capacity" 1 200
"$CHECK" ledger --file "$WD/event_capacity_error/hadronic_ledger_1.txt" --expect_dropped 1 --expect_identity 1 || report_fail=1
expect_complete event_capacity_warn --expect_dropped 1 --expect_npart 60 --expect_identity 1
# (iii) on_max_events
expect_abort event_maxevents_error "on_max_events" 1 200
expect_complete event_maxevents_warn --min_spawned 100 --expect_identity 1
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
