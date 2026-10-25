#!/usr/bin/env bash
# T7b: uniform E parallel to u, B = 0; K(t) vs RK4 of (field work - loss) at
# three dt (CFL 0.5, 0.25, 0.125): stored K <= 1e-3 at the finest (exact-rate
# RK4); the order (re-registered 2026-10-25, decision (a)) is measured on
# K_eff = K - delta against the interpolated-table RK4 with eta_u scaled as dt
# (1e-4, 5e-5, 2.5e-5 in t7b_cfl{1,2,4}.toml), pass 1.0 +- 0.3, with eta_u K
# above the pending floor at every dt; the round-1 fixed-eta_u stored-K orders
# are rerun with t7b_cfl{1,2,4}_fixed.toml and printed as (diag).
# Field work per unit time: dK/dt = omegaB0 E beta_d (q = +1; the pusher applies
# du = (q/m) omegaB0 E dt per step, sr.hpp normalized_dt_half). E = 11/32 makes
# the half-kicks exact multiples of ulp(u) in single precision at all three dt;
# the diagnostic at the end runs E = 0.34 without loss to show the pusher's own
# float rounding bias (21.76 ulps per half-kick rounded to 22 at CFL 0.125).
# Usage: t7b.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
declare -a ENDS ENDSH ENDET FLR
declare -A ETA=( [1]=1e-4 [2]=5e-5 [4]=2.5e-5 )
for tag in 1 2 4; do
  run_entity "$EXE" "$INP/t7b_cfl$tag.toml" "$WD/cfl$tag"; check_rc $? "t7b run CFL tag $tag"
  [ $report_fail -ne 0 ] && exit 1
  grep -h "bh_proto: closure" "$WD/cfl$tag/run.log" | cut -c1-250
  REAL=$(grep -o "real_t = [a-z]*" "$WD/cfl$tag/run.log" | head -1 | awk '{print $3}')
  if [ "$tag" = "4" ]; then tol="--tol 1e-3"; else tol=""; fi
  out=$("$CHECK" t7b --ions "$WD/cfl$tag/hadronic_ions_1.txt" --eps0 0.5 --mu_r 1836.15 --mu_d 1.0 \
        --K0 9180.75 --E 0.34375 --omegaB0 1.0 --tau0 0.208288 --skindepth0 1.0 $tol \
        --eta_u "${ETA[$tag]}" --floor_ulp 100 --real "${REAL:-float}")
  rc=$?
  echo "$out"
  [ $rc -ne 0 ] && report_fail=1
  ENDS[$tag]=$(echo "$out" | grep T7B_END | awk '{print $2}')
  ENDSH[$tag]=$(echo "$out" | grep T7B_END | awk '{print $3}')
  ENDET[$tag]=$(echo "$out" | grep T7B_END | awk '{print $4}')
  FLR[$tag]=$(echo "$out" | grep T7B_FLOOR_RATIO | awk '{print $2}')
done
echo "eta_u K / pending_floor at the end of each run: ${FLR[1]} ${FLR[2]} ${FLR[4]} (must exceed 1)"
for tag in 1 2 4; do
  if awk -v r="${FLR[$tag]}" 'BEGIN{exit !(r > 1.0)}'; then echo "PASS: T7b (re-registered): eta_u K above the pending floor, CFL tag $tag : ${FLR[$tag]} (tolerance 1)";
  else echo "FAIL: T7b (re-registered): eta_u K above the pending floor, CFL tag $tag : ${FLR[$tag]} (tolerance 1)"; report_fail=1; fi
done
echo "T7b (re-registered) end-time errors of K_eff vs the interpolated-table RK4 (eta_u scaled as dt): ${ENDET[1]} ${ENDET[2]} ${ENDET[4]}"
"$CHECK" order --e1 "${ENDET[1]}" --e2 "${ENDET[2]}" --e4 "${ENDET[4]}" | sed 's/^\(PASS\|FAIL\): /\1: T7b (re-registered) /' || report_fail=1
echo "(diag) stored K vs exact RK4 end-time errors in the eta_u-scaled runs (t = (n+1) dt): ${ENDS[1]} ${ENDS[2]} ${ENDS[4]}"
"$CHECK" order --e1 "${ENDS[1]}" --e2 "${ENDS[2]}" --e4 "${ENDS[4]}" | sed 's/^/(diag stored K, eta_u scaled) /'
echo "(diag) the same with the half-step convention t = (n+1/2) dt: ${ENDSH[1]} ${ENDSH[2]} ${ENDSH[4]}"
# round-1 registration kept as history: eta_u fixed at 1e-4, stored K vs exact RK4
declare -a ENDF
for tag in 1 2 4; do
  run_entity "$EXE" "$INP/t7b_cfl${tag}_fixed.toml" "$WD/cfl${tag}_fixed" || { echo "(diag) fixed-eta_u run failed"; continue; }
  out=$("$CHECK" t7b --ions "$WD/cfl${tag}_fixed/hadronic_ions_1.txt" --eps0 0.5 --mu_r 1836.15 --mu_d 1.0 \
        --K0 9180.75 --E 0.34375 --omegaB0 1.0 --tau0 0.208288 --skindepth0 1.0)
  echo "$out" | grep "t7b:" | sed 's/^/(diag fixed eta_u) /'
  ENDF[$tag]=$(echo "$out" | grep T7B_END | awk '{print $2}')
done
echo "(diag) round-1 registration (eta_u fixed, stored K vs exact RK4) end-time errors: ${ENDF[1]} ${ENDF[2]} ${ENDF[4]}"
"$CHECK" order --e1 "${ENDF[1]}" --e2 "${ENDF[2]}" --e4 "${ENDF[4]}" | sed 's/^/(diag round-1 registration, kept as history) /'
# diagnostic: pusher rounding bias at E = 0.34 (no loss), finest and coarsest dt
for tag in 1 4; do
  sed -e 's/tau0                = 0.208288/tau0                = 0.0/' -e 's/fields.E            = \[0.34375, 0.0, 0.0\]/fields.E            = [0.34, 0.0, 0.0]/' \
      "$INP/t7b_cfl$tag.toml" > "$WD/Eonly_cfl$tag.toml"
  run_entity "$EXE" "$WD/Eonly_cfl$tag.toml" "$WD/Eonly_cfl$tag" || { echo "(diag) E-only run failed"; continue; }
  "$CHECK" kick --ions "$WD/Eonly_cfl$tag/hadronic_ions_1.txt" --E 0.34 --omegaB0 1.0 --mu_d 1.0 | sed "s/^/(diag E = 0.34, no loss, cfl tag $tag) /"
  sed -e 's/tau0                = 0.208288/tau0                = 0.0/' "$INP/t7b_cfl$tag.toml" > "$WD/Eonly_exact_cfl$tag.toml"
  run_entity "$EXE" "$WD/Eonly_exact_cfl$tag.toml" "$WD/Eonly_exact_cfl$tag" || { echo "(diag) E-only run failed"; continue; }
  "$CHECK" kick --ions "$WD/Eonly_exact_cfl$tag/hadronic_ions_1.txt" --E 0.34375 --omegaB0 1.0 --mu_d 1.0 | sed "s/^/(diag E = 0.34375, no loss, cfl tag $tag) /"
done
exit $report_fail
