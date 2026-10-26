#!/bin/bash
# Produce the step-4 sampler fixtures of contract section 4 (items 1 to 3) with
# bh_sampler_dump: qk_table.txt, moments_<bath>.txt for the five step-1 baths,
# hist_<bath>_G<Gamma>.txt for mono05 and powerlaw at Gamma_p in
# {1.05 Gamma_th, 3, 10, 100}. Bash 3.2 compatible; fails when nothing runs.
#
# Usage: bh_sampler_fixtures.sh <bh_sampler_dump> <outdir> [--n N] [--n-weighted-attempts N] [--seed S]
set -u
if [ $# -lt 2 ]; then
  echo "usage: $0 <bh_sampler_dump> <outdir> [--n N] [--n-weighted-attempts N] [--seed S]" >&2
  exit 2
fi
DUMP="$1"; OUT="$2"; shift 2
EXTRA="$*"
if [ ! -x "$DUMP" ]; then
  echo "bh_sampler_fixtures.sh: $DUMP is not executable" >&2
  exit 2
fi
mkdir -p "$OUT" || exit 1
n_run=0; n_fail=0
run() {
  echo "+ $*"
  if "$@"; then n_run=$((n_run + 1)); else n_fail=$((n_fail + 1)); echo "FAILED: $*" >&2; fi
}
run "$DUMP" qk_table --out "$OUT/qk_table.txt"
for bath in mono05 mono03 powerlaw blackbody mono05_mu100; do
  # shellcheck disable=SC2086
  run "$DUMP" moments --bath "$bath" --out "$OUT/moments_$bath.txt" $EXTRA
done
# Gamma_p = 1.05 Gamma_th: mono05 Gamma_th = 2.125 -> 2.23125; powerlaw Gamma_th = 25/24 -> 1.09375
for g in 2.23125 3 10 100; do
  # shellcheck disable=SC2086
  run "$DUMP" hist --bath mono05 --gamma "$g" --out "$OUT/hist_mono05_G$g.txt" $EXTRA
done
for g in 1.09375 3 10 100; do
  # shellcheck disable=SC2086
  run "$DUMP" hist --bath powerlaw --gamma "$g" --out "$OUT/hist_powerlaw_G$g.txt" $EXTRA
done
echo "bh_sampler_fixtures.sh: $n_run commands succeeded, $n_fail failed"
if [ "$n_run" -eq 0 ] || [ "$n_fail" -ne 0 ]; then
  exit 1
fi
exit 0
