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
if [ $n_run -eq 0 ]; then echo "FAIL: r2_tiny.sh executed no run"; report_fail=1; fi
echo "r2_tiny.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
