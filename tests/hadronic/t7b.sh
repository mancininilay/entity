#!/usr/bin/env bash
# T7b: uniform E parallel to u, B = 0; K(t) vs RK4 of (field work - loss) at
# three dt (CFL 0.5, 0.25, 0.125): <= 1e-3 at the finest, order 1.0 +- 0.3.
# Field work per unit time: dK/dt = omegaB0 E beta_d (q = +1; the pusher applies
# du = (q/m) omegaB0 E dt per step, sr.hpp normalized_dt_half). E = 11/32 makes
# the half-kicks exact multiples of ulp(u) in single precision at all three dt;
# the diagnostic at the end runs E = 0.34 without loss to show the pusher's own
# float rounding bias (21.76 ulps per half-kick rounded to 22 at CFL 0.125).
# Usage: t7b.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
declare -a ENDS ENDSH ENDET
for tag in 1 2 4; do
  run_entity "$EXE" "$INP/t7b_cfl$tag.toml" "$WD/cfl$tag"; check_rc $? "t7b run CFL tag $tag"
  [ $report_fail -ne 0 ] && exit 1
  grep -h "bh_proto: closure" "$WD/cfl$tag/run.log" | cut -c1-250
  if [ "$tag" = "4" ]; then tol="--tol 1e-3"; else tol=""; fi
  out=$("$CHECK" t7b --ions "$WD/cfl$tag/hadronic_ions_1.txt" --eps0 0.5 --mu_r 1836.15 --mu_d 1.0 \
        --K0 9180.75 --E 0.34375 --omegaB0 1.0 --tau0 0.208288 --skindepth0 1.0 $tol)
  rc=$?
  echo "$out"
  [ $rc -ne 0 ] && report_fail=1
  ENDS[$tag]=$(echo "$out" | grep T7B_END | awk '{print $2}')
  ENDSH[$tag]=$(echo "$out" | grep T7B_END | awk '{print $3}')
  ENDET[$tag]=$(echo "$out" | grep T7B_END | awk '{print $4}')
done
echo "end-time errors (t = (n+1) dt convention): ${ENDS[1]} ${ENDS[2]} ${ENDS[4]}"
"$CHECK" order --e1 "${ENDS[1]}" --e2 "${ENDS[2]}" --e4 "${ENDS[4]}" || report_fail=1
echo "[DIAG] the same with the half-step convention t = (n+1/2) dt: ${ENDSH[1]} ${ENDSH[2]} ${ENDSH[4]}"
"$CHECK" order --e1 "${ENDSH[1]}" --e2 "${ENDSH[2]}" --e4 "${ENDSH[4]}" | sed 's/^/[DIAG half-step] /'
echo "[DIAG] K_eff vs interpolated-table RK4 end errors: ${ENDET[1]} ${ENDET[2]} ${ENDET[4]}"
"$CHECK" order --e1 "${ENDET[1]}" --e2 "${ENDET[2]}" --e4 "${ENDET[4]}" | sed 's/^/[DIAG K_eff, table reference] /'
# diagnostic: pusher rounding bias at E = 0.34 (no loss), finest and coarsest dt
for tag in 1 4; do
  sed -e 's/tau0                = 0.208288/tau0                = 0.0/' -e 's/fields.E            = \[0.34375, 0.0, 0.0\]/fields.E            = [0.34, 0.0, 0.0]/' \
      "$INP/t7b_cfl$tag.toml" > "$WD/Eonly_cfl$tag.toml"
  run_entity "$EXE" "$WD/Eonly_cfl$tag.toml" "$WD/Eonly_cfl$tag" || { echo "[DIAG] E-only run failed"; continue; }
  "$CHECK" kick --ions "$WD/Eonly_cfl$tag/hadronic_ions_1.txt" --E 0.34 --omegaB0 1.0 --mu_d 1.0 | sed "s/^/[DIAG E = 0.34, no loss, cfl tag $tag] /"
  sed -e 's/tau0                = 0.208288/tau0                = 0.0/' "$INP/t7b_cfl$tag.toml" > "$WD/Eonly_exact_cfl$tag.toml"
  run_entity "$EXE" "$WD/Eonly_exact_cfl$tag.toml" "$WD/Eonly_exact_cfl$tag" || { echo "[DIAG] E-only run failed"; continue; }
  "$CHECK" kick --ions "$WD/Eonly_exact_cfl$tag/hadronic_ions_1.txt" --E 0.34375 --omegaB0 1.0 --mu_d 1.0 | sed "s/^/[DIAG E = 0.34375, no loss, cfl tag $tag] /"
done
exit $report_fail
