#!/usr/bin/env bash
# rejection tests, end to end: entity.xc refuses each bad input with exit code 1
# and a message naming the key. Usage: reject.sh <entity.xc> <bh_check> <inputs> <workdir>
# Bash 3.2 compatible (no associative arrays); fails when no input was executed.
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
mkdir -p "$WD"
n_run=0; n_pass=0; n_fail=0
# <input name>:<setup key named by the message>
for pair in reject_tau0:tau0 reject_fmax:f_max reject_bool:hadronic.enabled \
            reject_unsafe:allow_unsafe_application reject_type:photon_field.type \
            reject_ionsK:ions.K \
            reject_pair_missing:pair_species reject_pair_zero:pair_species reject_pair_ion:pair_species \
            reject_pair_same:pair_species reject_epsmax:photon_field.eps_max \
            reject_gamma_typical:gamma_p_typical reject_fr:f_r reject_rth:r_th_factor \
            reject_pairweight:pair_weight reject_maxevents:max_events_per_step \
            reject_oncapacity:on_capacity reject_envelope:envelope.gamma_max \
            reject_placement:hadronic.placement; do
  name="${pair%%:*}"; key="${pair#*:}"
  if [ ! -f "$INP/$name.toml" ]; then
    echo "FAIL: $name: input $INP/$name.toml not found"; n_fail=$((n_fail + 1)); continue
  fi
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"
  rc=$?
  n_run=$((n_run + 1))
  if [ $rc -eq 0 ]; then
    echo "FAIL: $name: entity.xc accepted the input (exit 0)"; n_fail=$((n_fail + 1))
  else
    echo "ok: $name refused (exit $rc)"
    if "$CHECK" reject --log "$WD/$name/run.log" --key "$key"; then
      n_pass=$((n_pass + 1))
    else
      n_fail=$((n_fail + 1))
    fi
  fi
done
# species-level and particles-level refusals (messages name the input key)
for pair in "reject_payloads:n_payloads_real" "reject_weights:use_weights" "reject_gca:GCA" \
            "reject_poisson_weights:use_weights" "reject_ions_weight_flag:use_weights" \
            "reject_pair_charge:charge" "reject_pair_mass:mass 1" "reject_pair_slots:n_payloads_real" \
            "reject_tlatency:t_latency" "reject_two_species:exactly one hadronic ion species"; do
  name="${pair%%:*}"; key="${pair##*:}"
  if [ ! -f "$INP/$name.toml" ]; then
    echo "FAIL: $name: input $INP/$name.toml not found"; n_fail=$((n_fail + 1)); continue
  fi
  run_entity "$EXE" "$INP/$name.toml" "$WD/$name"
  rc=$?
  n_run=$((n_run + 1))
  if [ $rc -eq 0 ]; then
    echo "FAIL: $name: entity.xc accepted the input (exit 0)"; n_fail=$((n_fail + 1))
  elif grep -q "$key" "$WD/$name/run.log"; then
    echo "PASS: $name refused (exit $rc), message names $key: $(grep -m1 "$key" "$WD/$name/run.log" | cut -c1-200)"
    n_pass=$((n_pass + 1))
  else
    echo "FAIL: $name refused but the message does not name $key"; n_fail=$((n_fail + 1))
  fi
done
if [ $n_run -eq 0 ]; then
  echo "FAIL: reject.sh executed no input"; n_fail=$((n_fail + 1))
fi
echo "reject.sh summary: $n_run inputs executed, $n_pass passed, $n_fail failed"
[ $n_fail -eq 0 ] && [ $n_run -gt 0 ]
