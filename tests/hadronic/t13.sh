#!/usr/bin/env bash
# T13 (engine level): discrete charge conservation. Pairs deposit (ions do not), periodic, no current
# filtering, E = B = 0 initially, a background pair plasma (32 ppc per species, T = 0.1) loaded through
# the stock injector, threshold spectrum with a longitudinal kick (unequal lepton velocities), 200 steps.
# The Gauss residual R_i = div_h E - (q0/(B0 V0)) rho_i (CIC rho from the arrays, double) is compared
# with its initial value R0 (a randomly loaded background does not satisfy Gauss's law; the deposit
# conserves R - R0 to float round-off): max|R - R0| / max|c rho| of the pair run within a factor 2 of
# the control run without pair creation, no growth trend, the first births within the control level.
# Then the same with one pass of the stock current filter (diagnostic). Bash 3.2 compatible.
# Usage: t13.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
run_entity "$EXE" "$INP/t13_gauss.toml" "$WD/gauss"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "T13 pair run"
run_entity "$EXE" "$INP/t13_control.toml" "$WD/control"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "T13 control run (source = none)"
[ $report_fail -ne 0 ] && { echo "t13.sh summary: a run failed"; exit 1; }
grep -h "background:" "$WD/gauss/run.log" | cut -c1-250
grep -h "bh_proto: ledger species" "$WD/gauss/run.log" | sed 's/P_drag.*//' | cut -c1-400
"$CHECK" gauss --test "$WD/gauss/hadronic_gauss.txt" --control "$WD/control/hadronic_gauss.txt" || report_fail=1
"$CHECK" ledger --file "$WD/gauss/hadronic_ledger_1.txt" --min_spawned 100 || report_fail=1
echo "(diag) the pair run's ledger identity holds with the kick energy in E_round (the kick is not rounding):"
grep -h "bh_proto: array identity" "$WD/gauss/run.log" | cut -c1-500
# diagnostic: the stock current filter on
run_entity "$EXE" "$INP/t13_filter.toml" "$WD/filter"; rc=$?
if [ $rc -eq 0 ]; then
  "$CHECK" gauss --test "$WD/filter/hadronic_gauss.txt" --control "$WD/control/hadronic_gauss.txt" | sed 's/^/(diag, filter on) /'
else
  echo "(diag) the filter run failed (exit $rc)"
fi
if [ $n_run -eq 0 ]; then echo "FAIL: t13.sh executed no run"; report_fail=1; fi
echo "t13.sh summary: $n_run registered runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
