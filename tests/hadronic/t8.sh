#!/usr/bin/env bash
# T8: bit identity on the Serial backend. Three runs of the same problem:
# hadronic.enabled = false; tau0 = 0; tau0 > 0 with every ion below the table's
# lower edge. Identical end-of-run checksums of all ion arrays and of the fields;
# in the last two ledgers every transfer and activity column exactly zero and the
# snapshot columns at their independently expected values (Codex 1's closure C7.1,
# 2026-10-31): n_alive = 0 for tau0 = 0 (the stage never launches the kernel) and
# = the ion count of the input (ions.per_K x the number of ions.K entries) for the
# below-threshold run (every ion alive, processed, below the edge);
# debt_outstanding = pending_outstanding = 0; npart_minus = npart_plus = 0.
# Usage: t8.sh <entity.xc (Serial build for the registered entry; an OpenMP build runs the same checks)> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then
  echo "SKIP: no Serial-backend build given (-D BH_SERIAL_ENTITY=...)"; exit 77
fi
for name in t8_disabled t8_tau0zero t8_below; do
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"; check_rc $? "t8 run $name"
done
[ $report_fail -ne 0 ] && exit 1
grep -h "execution space\|Kokkos::Serial\|Serial" "$WD/t8_below/run.log" | head -3
# the expected n_alive snapshots, computed from the inputs (not from any run output): the tau0 = 0 run never launches
# the kernel (0); the below-threshold run processes every ion of the input (ions.per_K x number of ions.K entries)
n_ions() { awk -F'=' '/^ *ions\.K *=/ { s = $2; gsub(/[^0-9.,eE+-]/, "", s); nk = split(s, a, ","); } /^ *ions\.per_K *=/ { p = $2 + 0 } END { print nk * p }' "$1"; }
ALIVE_B=0
ALIVE_C=$(n_ions "$INP/t8_below.toml")
echo "t8: expected n_alive snapshots: tau0 = 0 run $ALIVE_B, below-threshold run $ALIVE_C (from $INP/t8_below.toml)"
[ "$ALIVE_C" -gt 0 ] || { echo "FAIL: could not compute the ion count of t8_below.toml"; report_fail=1; }
"$CHECK" t8 --a "$WD/t8_disabled/run.log" --b "$WD/t8_tau0zero/run.log" --c "$WD/t8_below/run.log" \
  --ledger_b "$WD/t8_tau0zero/hadronic_ledger_1.txt" --ledger_c "$WD/t8_below/hadronic_ledger_1.txt" \
  --alive_b "$ALIVE_B" --alive_c "$ALIVE_C" || report_fail=1
# the kernel was launched in run c: n_processed stays 0 (every ion below the edge) but the stage is active
grep -h "stage active" "$WD/t8_below/run.log" | cut -c1-120 || { echo "FAIL: run c did not launch the stage"; report_fail=1; }
echo "t8.sh summary: $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
