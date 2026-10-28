#!/usr/bin/env bash
# Step 4 round 2 (finding 5), non-MPI code path: the restframe sampler's attempt and overflow counters
# reach the ledger file. A budget run with the weighted chain's rejection loop bounded at one trial per
# packet (test.max_sampler_attempts = 1) at Gamma_p = 6 in mono05, where the acceptance is ~15%, so that
# most packets exhaust the loop and fall back to the mean-energy law: the last ledger row must carry
# n_sampler_overflow > 0 (and n_sampler_attempts >= n_sampler_overflow), equal to the counters of the
# run summary, with the identity residual at 1e-12. The rank-asymmetric check of the MPI reduction of
# the two counters belongs to the next multi-rank (Rusty) job. Bash 3.2 compatible.
# Usage: sampler_overflow.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
run_entity "$EXE" "$INP/sampler_overflow.toml" "$WD/overflow"; rc=$?; n_run=$((n_run + 1))
check_rc $rc "sampler overflow run"
[ $report_fail -ne 0 ] && { echo "sampler_overflow.sh summary: the run failed"; tail -20 "$WD/overflow/run.log"; exit 1; }
LEDGER="$WD/overflow/hadronic_ledger_1.txt"
"$CHECK" ledger --file "$LEDGER" --min_spawned 100 --min_sampler_overflow 1 || report_fail=1
# the ledger's counters equal the run summary's (both from the reduced ledger)
SUM=$(grep -h "bh_proto: restframe sampler species" "$WD/overflow/run.log" | tail -1)
echo "(diag) $SUM"
S_ATT=$(echo "$SUM" | sed -n 's/.*n_sampler_attempts \([0-9]*\).*/\1/p')
S_OVF=$(echo "$SUM" | sed -n 's/.*n_sampler_overflow \([0-9]*\).*/\1/p')
L_ATT=$(grep -v '^#' "$LEDGER" | tail -1 | awk '{print $44}')
L_OVF=$(grep -v '^#' "$LEDGER" | tail -1 | awk '{print $45}')
if [ -n "$S_ATT" ] && [ "$S_ATT" = "$L_ATT" ] && [ -n "$S_OVF" ] && [ "$S_OVF" = "$L_OVF" ]; then
  echo "PASS: ledger counters = run summary counters (attempts $L_ATT, overflow $L_OVF)"
else
  echo "FAIL: ledger counters (attempts '$L_ATT', overflow '$L_OVF') differ from the run summary ('$S_ATT', '$S_OVF')"
  report_fail=1
fi
if [ $n_run -eq 0 ]; then echo "FAIL: sampler_overflow.sh executed no run"; report_fail=1; fi
echo "sampler_overflow.sh summary: $n_run registered runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
