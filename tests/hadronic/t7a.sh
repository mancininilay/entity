#!/usr/bin/env bash
# T7a: uniform B perpendicular to u; K(t) equals the B = 0 run to 1e-4 at every
# output time and the pitch cosine is constant to 1e-5.
# Usage: t7a.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
run_entity "$EXE" "$INP/t7a_B.toml"  "$WD/B";  check_rc $? "t7a run with B = 100 z-hat"
run_entity "$EXE" "$INP/t7a_B0.toml" "$WD/B0"; check_rc $? "t7a run with B = 0"
[ $report_fail -ne 0 ] && exit 1
grep -h "bh_proto: closure\|bh_proto: ledger" "$WD/B/run.log" "$WD/B0/run.log" | cut -c1-250
REAL=$(grep -o "real_t = [a-z]*" "$WD/B/run.log" | head -1 | awk '{print $3}')
"$CHECK" t7a --run "$WD/B/hadronic_ions_1.txt" --ref "$WD/B0/hadronic_ions_1.txt" --tol 1e-4 --pitch_tol 1e-5 \
  --eta_u 1e-4 --floor_ulp 100 --mu_d 1.0 --real "${REAL:-float}" || report_fail=1
echo "t7a.sh summary: $([ $report_fail -eq 0 ] && echo "all checks passed" || echo "at least one check FAILED")"
exit $report_fail
