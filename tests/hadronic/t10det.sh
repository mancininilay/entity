#!/usr/bin/env bash
# T10 poisson determinism on the Serial backend: the same simulation.seed gives identical checksums of
# every particle array (ions and both pair species) and of the fields; a different seed differs. The
# OpenMP build is run as a diagnostic (its pool assigns generator states by scheduling).
# Bash 3.2 compatible. Usage: t10det.sh <entity_serial.xc> <entity_openmp.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; EXE_OMP="$2"; CHECK="$3"; INP="$4"; WD="$5"
if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then
  echo "SKIP: no Serial-backend build given (-D BH_SERIAL_ENTITY=...)"; exit 77
fi
n_run=0
for tag in a1 a2 b; do
  case "$tag" in a1|a2) inp=t10_poisson_det_a ;; *) inp=t10_poisson_det_b ;; esac
  run_entity "$EXE" "$INP/$inp.toml" "$WD/$tag"; rc=$?; n_run=$((n_run + 1)); check_rc $rc "poisson run $tag ($inp) on Serial"
done
[ $report_fail -ne 0 ] && { echo "t10det.sh summary: a run failed"; exit 1; }
grep -h "execution space\|Serial" "$WD/a1/run.log" | head -2
grep -h "RNG pool seeded" "$WD/a1/run.log" "$WD/b/run.log" | cut -c1-200
"$CHECK" det --a "$WD/a1/run.log" --b "$WD/a2/run.log" --expect same || report_fail=1
"$CHECK" det --a "$WD/a1/run.log" --b "$WD/b/run.log" --expect differ || report_fail=1
cmp -s "$WD/a1/hadronic_pairs.txt" "$WD/a2/hadronic_pairs.txt" && echo "PASS: identical pair files (positions, momenta, payloads) for the same seed" || { echo "FAIL: pair files differ for the same seed"; report_fail=1; }
"$CHECK" ledger --file "$WD/a1/hadronic_ledger_1.txt" || report_fail=1
if [ -n "$EXE_OMP" ] && [ -x "$EXE_OMP" ]; then
  run_entity "$EXE_OMP" "$INP/t10_poisson_det_a.toml" "$WD/omp1" && run_entity "$EXE_OMP" "$INP/t10_poisson_det_a.toml" "$WD/omp2" && \
    "$CHECK" det --a "$WD/omp1/run.log" --b "$WD/omp2/run.log" --expect same | sed 's/^/(diag OpenMP) /'
fi
if [ $n_run -eq 0 ]; then echo "FAIL: t10det.sh executed no run"; report_fail=1; fi
echo "t10det.sh summary: $n_run runs executed, $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
