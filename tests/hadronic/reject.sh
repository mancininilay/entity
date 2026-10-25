#!/usr/bin/env bash
# rejection tests, end to end: entity.xc refuses each bad input with exit code 1
# and a message naming the key. Usage: reject.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
mkdir -p "$WD"
declare -A KEYS=( [reject_tau0]=tau0 [reject_fmax]=f_max [reject_bool]=hadronic.enabled
                  [reject_unsafe]=allow_unsafe_application [reject_type]=photon_field.type
                  [reject_ionsK]=ions.K )
for name in reject_tau0 reject_fmax reject_bool reject_unsafe reject_type reject_ionsK; do
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "FAIL: $name: entity.xc accepted the input (exit 0)"; report_fail=1
  else
    echo "ok: $name refused (exit $rc)"
    "$CHECK" reject --log "$WD/$name/run.log" --key "${KEYS[$name]}" || report_fail=1
  fi
done
# species-level and particles-level refusals (messages name the input key)
for pair in "reject_payloads:n_payloads_real" "reject_weights:use_weights"; do
  name="${pair%%:*}"; key="${pair##*:}"
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "FAIL: $name: entity.xc accepted the input (exit 0)"; report_fail=1
  elif grep -q "$key" "$WD/$name/run.log"; then
    echo "PASS: $name refused (exit $rc), message names $key: $(grep -m1 "$key" "$WD/$name/run.log" | cut -c1-200)"
  else
    echo "FAIL: $name refused but the message does not name $key"; report_fail=1
  fi
done
exit $report_fail
