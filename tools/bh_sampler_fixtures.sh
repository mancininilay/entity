#!/bin/bash
# Produce the step-4 sampler fixtures of contract section 4 (items 1 to 3) with
# bh_sampler_dump: qk_table.txt, moments_<bath>.txt for the five step-1 baths,
# hist_<bath>_G<Gamma>.txt for mono05 and powerlaw at Gamma_p in
# {1.05 Gamma_th, 3, 10, 100}, exact_chain_cdf_mono05_G<Gamma>.txt at 100 and 1000
# (the exact-chain self-check of the lab spectrum; round 1 called it ka2008_*).
# Bash 3.2 compatible; fails when nothing runs.
#
# Round 2: the weighted chains run to a fixed number of accepted packets (1e5
# per node for the moments, 1e6 per histogram; the attempts are reported), and
# the histograms take their bin edges from the reference's file of the same
# bath and Gamma_p when it is present (tests/data/hadronic/bhref-0.2.0).
#
# Usage: bh_sampler_fixtures.sh <bh_sampler_dump> <outdir> [--n N] [--n-weighted N] [--seed S] [--threads N]
set -u
if [ $# -lt 2 ]; then
  echo "usage: $0 <bh_sampler_dump> <outdir> [--n N] [--n-weighted N] [--seed S] [--threads N]" >&2
  exit 2
fi
DUMP="$1"; OUT="$2"; shift 2
EXTRA="$*"
if [ ! -x "$DUMP" ]; then
  echo "bh_sampler_fixtures.sh: $DUMP is not executable" >&2
  exit 2
fi
mkdir -p "$OUT" || exit 1
# absolute paths, then run from the repository root so that the reference path
# recorded in the histogram headers is repository-relative
DUMP="$(cd "$(dirname "$DUMP")" && pwd)/$(basename "$DUMP")"
OUT="$(cd "$OUT" && pwd)"
cd "$(dirname "$0")/.." || exit 1
REF="tests/data/hadronic/bhref-0.2.0"
bins_from() {
  if [ -f "$REF/hist_$1_G$2.txt" ]; then echo "--bins-from $REF/hist_$1_G$2.txt"; fi
}
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
  run "$DUMP" hist --bath mono05 --gamma "$g" --out "$OUT/hist_mono05_G$g.txt" $(bins_from mono05 "$g") $EXTRA
done
for g in 1.09375 3 10 100; do
  # shellcheck disable=SC2086
  run "$DUMP" hist --bath powerlaw --gamma "$g" --out "$OUT/hist_powerlaw_G$g.txt" $(bins_from powerlaw "$g") $EXTRA
done
for g in 100 1000; do
  run "$DUMP" exact_chain_cdf --bath mono05 --gamma "$g" --out "$OUT/exact_chain_cdf_mono05_G$g.txt"
done
echo "bh_sampler_fixtures.sh: $n_run commands succeeded, $n_fail failed"
if [ "$n_run" -eq 0 ] || [ "$n_fail" -ne 0 ]; then
  exit 1
fi
exit 0
