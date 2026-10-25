#!/usr/bin/env bash
# T8: bit identity on the Serial backend. Three runs of the same problem:
# hadronic.enabled = false; tau0 = 0; tau0 > 0 with every ion below the table's
# lower edge. Identical end-of-run checksums of all ion arrays and of the fields;
# ledger identically zero in the last two.
# Usage: t8.sh <entity_serial.xc> <bh_check> <inputs> <workdir>
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
"$CHECK" t8 --a "$WD/t8_disabled/run.log" --b "$WD/t8_tau0zero/run.log" --c "$WD/t8_below/run.log" \
  --ledger_b "$WD/t8_tau0zero/hadronic_ledger_1.txt" --ledger_c "$WD/t8_below/hadronic_ledger_1.txt" || report_fail=1
# the kernel was launched in run c: n_processed stays 0 (every ion below the edge) but the stage is active
grep -h "stage active" "$WD/t8_below/run.log" | cut -c1-120 || { echo "FAIL: run c did not launch the stage"; report_fail=1; }
exit $report_fail
