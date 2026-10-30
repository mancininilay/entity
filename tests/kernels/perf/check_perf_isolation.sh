#!/usr/bin/env bash
# P2 step 7, M3: the production build does not compile the perf variants in, and the generated kernel copy is
# current. Usage: check_perf_isolation.sh <source root> <build dir>   (Bash 3.2 compatible)
#   1. no production source (src/, pgens/) mentions the perf harness (BH_PERF, bh_perf, hadronic_perf);
#   2. in the build's ninja file BH_PERF_VARIANT appears only in the compile rules of the perf target;
#   3. regenerating tests/kernels/perf/bh_perf_kernel_variant.hpp from the production header reproduces the
#      committed file byte for byte.
SRC="$1"; BLD="$2"; fail=0
[ -d "$SRC" ] && [ -d "$BLD" ] || { echo "usage: check_perf_isolation.sh <source root> <build dir>"; exit 2; }
hits=$(grep -rl "BH_PERF\|bh_perf\|hadronic_perf" "$SRC/src" "$SRC/pgens" 2>/dev/null | wc -l | tr -d ' ')
if [ "$hits" = "0" ]; then echo "PASS: no production source under src/ or pgens/ mentions the perf harness"; else
  echo "FAIL: production sources mention the perf harness:"; grep -rl "BH_PERF\|bh_perf\|hadronic_perf" "$SRC/src" "$SRC/pgens"; fail=1; fi
if [ -f "$BLD/build.ninja" ]; then
  # a DEFINES line belongs to the preceding "build <target>" line: the define may appear only under bh_perf objects
  other=$(awk '/^build / { tgt = $0 } /BH_PERF_VARIANT/ { if (tgt !~ /bh_perf/) n++ } END { print n + 0 }' "$BLD/build.ninja")
  mine=$(grep -c "BH_PERF_VARIANT" "$BLD/build.ninja")
  if [ "$other" = "0" ] && [ "$mine" -gt 0 ]; then echo "PASS: BH_PERF_VARIANT appears in $mine ninja lines, all of the perf target"; else
    echo "FAIL: BH_PERF_VARIANT in $other non-perf ninja lines ($mine in all)"; fail=1; fi
else
  echo "FAIL: $BLD/build.ninja not found"; fail=1
fi
tmp="${TMPDIR:-/tmp}/bh_perf_kernel_variant.$$.hpp"
bash "$SRC/tests/kernels/perf/make_variant_kernel.sh" "$SRC/src/kernels/hadronic/bethe_heitler.hpp" "$tmp" > /dev/null || fail=1
if cmp -s "$tmp" "$SRC/tests/kernels/perf/bh_perf_kernel_variant.hpp"; then echo "PASS: the committed variant kernel copy equals a fresh generation from the production header"; else
  echo "FAIL: tests/kernels/perf/bh_perf_kernel_variant.hpp is stale: regenerate with make_variant_kernel.sh"; fail=1; fi
rm -f "$tmp"
echo "check_perf_isolation.sh summary: $([ $fail -eq 0 ] && echo all checks passed || echo FAILED)"
exit $fail
