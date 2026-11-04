#!/usr/bin/env bash
# Codex 1's F1 (2026-11-03): the pair dump under bounds-checked views. Usage: dump_bounds.sh <harness> <old-method harness> <workdir>
# Cases are <capacity e-> <npart e-> <capacity e+> <npart e+>. The method of pgen.hpp must pass every case: unequal counts and
# capacities with each species the shorter, an allocation shorter than the other species' count, an empty allocation.
# Negative control, the frozen method of 22449c32: it must pass while the capacities hold the larger count and abort with
# Kokkos' out-of-bounds message when they do not (Codex 1's control: counts 3/1, capacities 3/3 exit 0, 3/1 SIGABRT).
# Exit: 0 every case as expected, 1 otherwise. Bash 3.2 compatible.
source "$(dirname "$0")/common.sh"
NEW="$1"; OLD="$2"; WD="$3"
rm -rf "$WD"; mkdir -p "$WD"
n_bad=0; n_check=0

expect() { # expect <harness> <label> <pass|abort> <case...>
  local exe="$1" label="$2" want="$3" rc oob log; shift 3
  log="$WD/$label-$(echo "$*" | tr ' ' '_').log"
  OMP_NUM_THREADS=1 "$exe" "$@" "$log.dump" > "$log" 2>&1; rc=$?
  oob=$(grep -c "out of bounds access" "$log")
  n_check=$((n_check + 1))
  if { [ "$want" = "pass" ] && [ $rc -eq 0 ] && [ "$oob" = "0" ]; } || { [ "$want" = "abort" ] && [ $rc -ge 128 ] && [ "$oob" != "0" ]; }; then
    echo "PASS: $label, capacity/npart e- $1/$2, e+ $3/$4: $want (exit $rc)$(grep -h 'summary' "$log" | sed 's/^/, /')"
  else
    echo "FAIL: $label, capacity/npart e- $1/$2, e+ $3/$4: expected $want, exit $rc, out-of-bounds lines $oob (log $log)"; n_bad=$((n_bad + 1))
  fi
}

for c in "3 3 1 1" "1 1 3 3" "5 4 2 2" "2 2 5 4" "3 3 0 0" "0 0 3 3" "4 3 4 1" "4 1 4 3" "5 2 3 2" "2 2 5 2"; do
  # shellcheck disable=SC2086
  expect "$NEW" current pass $c
done
for c in "3 3 3 1" "3 1 3 3" "5 2 3 2"; do
  # shellcheck disable=SC2086
  expect "$OLD" old-22449c32 pass $c
done
for c in "3 3 1 1" "1 1 3 3" "5 4 2 2" "2 2 5 4"; do
  # shellcheck disable=SC2086
  expect "$OLD" old-22449c32 abort $c
done
echo "dump_bounds summary: $n_check checks, $n_bad mismatches"
[ "$n_bad" -eq 0 ]
