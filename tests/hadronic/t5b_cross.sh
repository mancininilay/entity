#!/usr/bin/env bash
# T5b cross-build: single vs double Entity builds, mu_d in {1, 1836.15}, Gamma_p = 3,
# dt = 0.5, tau0 such that dK/dK_ulp in {0.3, 1, 3, 10, 100} (single-precision ulp),
# run until K has dropped by 1% in the double build. Design on: pass if the single
# build agrees with the double build to <= 1e-4 relative at that time. Design off:
# expected to fail; the ratio K_single/K_double - 1 is tabulated.
# Usage: t5b_cross.sh <entity_single.xc> <entity_double.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE_S="$1"; EXE_D="$2"; CHECK="$3"; INP="$4"; WD="$5"
if [ -z "$EXE_D" ] || [ ! -x "$EXE_D" ]; then
  echo "SKIP: no double-precision build given (-D BH_DOUBLE_ENTITY=...)"; exit 77
fi
mkdir -p "$WD"
DT=0.5
TABLE="$WD/design_off_table.txt"
echo "# mu_d ratio tau0 steps K_single/K_double-1 (design off) K_eff ratio-1 realised/expected" > "$TABLE"
for mu in 1.0 1836.15; do
  for ratio in 0.3 1 3 10 100; do
    line=$("$CHECK" tau0 --eps0 0.5 --mu_r 1836.15 --mu_d $mu --gamma_p 3.0 --dt $DT --skindepth0 1.0 --ratio $ratio) || { echo "FAIL: tau0 helper"; exit 1; }
    tau0=$(echo "$line" | awk '{print $2}'); steps=$(echo "$line" | awk '{print $4}')
    runtime=$(awk -v s=$steps -v d=$DT 'BEGIN{printf "%.6f", s*d}')
    echo "== mu_d = $mu, ratio = $ratio: $line, runtime = $runtime"
    for design in on off; do
      if [ $design = on ]; then eta=1e-4; floor=100; allow=false; else eta=0.0; floor=0.0; allow=true; fi
      tag="mu${mu}_r${ratio}_${design}"
      sed -e "s/MU_D/$mu/; s/TAU0/$tau0/; s/RUNTIME/$runtime/; s/ETA_U/$eta/; s/FLOOR/$floor/; s/ALLOW/$allow/" \
          "$INP/t5b_template.toml" > "$WD/$tag.toml"
      run_entity "$EXE_S" "$WD/$tag.toml" "$WD/${tag}_single"; check_rc $? "single $tag"
      run_entity "$EXE_D" "$WD/$tag.toml" "$WD/${tag}_double"; check_rc $? "double $tag"
      [ $report_fail -ne 0 ] && exit 1
      grep -h "dK/dK_ulp" "$WD/${tag}_single/run.log" | sed 's/.*dK\/K = /[DIAG] single: dK\/K = /' | cut -c1-200
      if [ $design = on ]; then
        "$CHECK" t5b --single "$WD/${tag}_single/hadronic_ions_1.txt" --double "$WD/${tag}_double/hadronic_ions_1.txt" --tol 1e-4 || report_fail=1
      else
        out=$("$CHECK" t5b-off --single "$WD/${tag}_single/hadronic_ions_1.txt" --double "$WD/${tag}_double/hadronic_ions_1.txt")
        echo "$out"
        row=$(echo "$out" | grep T5B_ROW | awk '{print $2, $3, $4}')
        echo "$mu $ratio $tau0 $steps $row" >> "$TABLE"
        echo "EXPECTED-FAIL (design off, reported only): mu_d $mu ratio $ratio: K_single/K_double - 1 = $(echo $row | awk '{print $1}')"
      fi
    done
  done
done
echo "---- design off table ($TABLE) ----"; cat "$TABLE"
exit $report_fail
