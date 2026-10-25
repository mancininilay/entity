#!/usr/bin/env bash
# T12b (engine level): startup capacity estimate with a power-law envelope matching the initial ions
# (Gamma_p log-spaced on [3, 12], index 1), c_layer = 1, comm_headroom = 0. A probe run with a huge
# maxnpart prints the estimate; an input whose pair maxnpart is half the estimate is refused with the
# printed numbers; one with twice the estimate runs, and its measured spawn count lies within a factor
# 2 of the estimate. Bash 3.2 compatible. Usage: t12b.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
mkdir -p "$WD"
n_run=0
sed -e 's/MAXNPART/1000000.0/' "$INP/t12b_estimate_template.toml" > "$WD/probe.toml"
run_entity "$EXE" "$WD/probe.toml" "$WD/probe"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "T12b probe run"
[ $report_fail -ne 0 ] && { echo "t12b.sh summary: probe run failed"; exit 1; }
grep -h "capacity estimate" "$WD/probe/run.log" | cut -c1-400
out=$("$CHECK" t12b --log "$WD/probe/run.log" --measured 1); rc=$?; echo "$out"; [ $rc -ne 0 ] && report_fail=1
bare=$(echo "$out" | grep T12B_BARE | awk '{print $2}')
half=$(awk -v b="$bare" 'BEGIN{printf "%d", (b/2 > 1) ? b/2 : 1}')
twice=$(awk -v b="$bare" 'BEGIN{printf "%d", 2*b + 1}')
echo "estimate (bare) = $bare : half = $half, twice = $twice"
sed -e "s/MAXNPART/$half.0/" "$INP/t12b_estimate_template.toml" > "$WD/half.toml"
run_entity "$EXE" "$WD/half.toml" "$WD/half"; rc=$?; n_run=$((n_run + 1))
if [ $rc -eq 0 ]; then
  echo "FAIL: T12b: the input with maxnpart = $half (half the estimate) was accepted"; report_fail=1
elif grep -q "on_capacity_estimate" "$WD/half/run.log"; then
  echo "PASS: T12b: maxnpart = $half (half the estimate) refused (exit $rc): $(grep -m1 'on_capacity_estimate: the startup' "$WD/half/run.log" | cut -c1-250)"
else
  echo "FAIL: T12b: the half input was refused but not by on_capacity_estimate"; report_fail=1
fi
sed -e "s/MAXNPART/$twice.0/" "$INP/t12b_estimate_template.toml" > "$WD/twice.toml"
run_entity "$EXE" "$WD/twice.toml" "$WD/twice"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "T12b: maxnpart = $twice (twice the estimate) runs"
if [ $rc -eq 0 ]; then
  "$CHECK" t12b --log "$WD/twice/run.log" --measured 1 || report_fail=1
  "$CHECK" ledger --file "$WD/twice/hadronic_ledger_1.txt" --expect_dropped 0 || report_fail=1
fi
if [ $n_run -eq 0 ]; then echo "FAIL: t12b.sh executed no run"; report_fail=1; fi
echo "t12b.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
