#!/usr/bin/env bash
# T14b-lite: the production placement of the hadronic stage (srpic::HadronicInteractions, after the deposit
# and before the particle exchange; hadronic.placement = "production") against the post-step placement
# (CustomPostStep; default). On one rank, periodic, the budget run of pairs_smoke.toml gives ledger files
# identical to round-off (every real column to 1e-12 relative, every counter exactly, in every row) and the
# same pair count; the T-exp absorbed-ion harvest (texp_absorb.toml) works at the production placement and
# gives the same ledger. The production inputs are derived from the committed inputs by inserting the one
# key under [setup], so that nothing else differs. With a Serial-backend build the checksums of every
# particle array and of the fields are also bit-identical between the placements (budget and poisson runs).
# Bash 3.2 compatible; fails when it runs nothing.
# Usage: t14b_placement.sh <entity_openmp.xc> <entity_serial.xc or ""> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; EXE_SERIAL="$2"; CHECK="$3"; INP="$4"; WD="$5"
n_run=0
mkdir -p "$WD/inputs" || exit 2
derive() {
  # derive <name> : writes $WD/inputs/<name>_production.toml with hadronic.placement = "production"
  local name="$1" out="$WD/inputs/$1_production.toml"
  awk '{ print } /^  hadronic.enabled[ ]*=[ ]*true/ { print "  hadronic.placement  = \"production\"" }' \
    "$INP/$name.toml" > "$out" || return 1
  local n
  n=$(grep -c 'hadronic.placement' "$out")
  if [ "$n" -ne 1 ]; then echo "FAIL: derived input $out has $n placement keys (expected 1)"; return 1; fi
  if [ "$(diff "$INP/$name.toml" "$out" | grep -c '^[<>]')" -ne 1 ]; then
    echo "FAIL: derived input $out differs from $name.toml by more than the placement key"; return 1
  fi
  return 0
}
compare_pair() {
  # compare_pair <exe> <name> <tag> <extra bh_check ledger args...> : runs both placements, compares
  local exe="$1" name="$2" tag="$3"; shift 3
  derive "$name" || { report_fail=1; return; }
  run_entity "$exe" "$INP/$name.toml" "$WD/$tag/poststep"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "$tag: post-step placement run ($name)"
  run_entity "$exe" "$WD/inputs/${name}_production.toml" "$WD/$tag/production"; rc2=$?; n_run=$((n_run + 1)); check_rc $rc2 "$tag: production placement run"
  if [ $rc -ne 0 ] || [ $rc2 -ne 0 ]; then return; fi
  grep -q "production placement" "$WD/$tag/production/run.log" \
    && echo "ok: $tag: the production run reports the production placement" \
    || { echo "FAIL: $tag: the production run log does not report the production placement"; report_fail=1; }
  grep -q "production placement" "$WD/$tag/poststep/run.log" \
    && { echo "FAIL: $tag: the post-step run log reports the production placement"; report_fail=1; }
  "$CHECK" cmp --a "$WD/$tag/poststep/hadronic_ledger_1.txt" --b "$WD/$tag/production/hadronic_ledger_1.txt" --tol 1e-12 || report_fail=1
  "$CHECK" ledger --file "$WD/$tag/production/hadronic_ledger_1.txt" "$@" || report_fail=1
}
# 1. periodic budget run (T10/T11 quantities: the ledger) on the build under test
compare_pair "$EXE" pairs_smoke budget --min_spawned 1
# the checksum lines on the OpenMP build are a diagnostic (order-independent checksums; equal values expected)
"$CHECK" det --a "$WD/budget/poststep/run.log" --b "$WD/budget/production/run.log" --expect same | sed 's/^/(diag OpenMP) /'
# 2. T-exp: absorbing walls, the dead-ion harvest at the production placement
compare_pair "$EXE" texp_absorb texp
if [ -f "$WD/texp/production/run.log" ]; then
  "$CHECK" texp --log "$WD/texp/production/run.log" || report_fail=1
fi
# 3. Serial backend: bit identity of every checksum between the placements (budget and poisson)
if [ -n "$EXE_SERIAL" ] && [ -x "$EXE_SERIAL" ]; then
  compare_pair "$EXE_SERIAL" pairs_smoke serial_budget --min_spawned 1
  "$CHECK" det --a "$WD/serial_budget/poststep/run.log" --b "$WD/serial_budget/production/run.log" --expect same || report_fail=1
  compare_pair "$EXE_SERIAL" t10_poisson_det_a serial_poisson --min_spawned 1
  "$CHECK" det --a "$WD/serial_poisson/poststep/run.log" --b "$WD/serial_poisson/production/run.log" --expect same || report_fail=1
  cmp -s "$WD/serial_poisson/poststep/hadronic_pairs.txt" "$WD/serial_poisson/production/hadronic_pairs.txt" \
    && echo "PASS: Serial poisson: identical pair files between the placements" \
    || { echo "FAIL: Serial poisson: the pair files differ between the placements"; report_fail=1; }
else
  echo "(no Serial-backend build given: the bit-identity part is not run)"
fi
if [ $n_run -eq 0 ]; then echo "FAIL: t14b_placement.sh executed no run"; report_fail=1; fi
echo "t14b_placement.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
