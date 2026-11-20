#!/usr/bin/env bash
# Incremental builds of the unit test from the pull-request worktree (dev/bh-emission) in the four
# trees of the first correction round (Mac, Apple clang 21, Kokkos 5.0.1 OpenMP): single and double
# precision, plain and with -DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK. The four builds run side by side.
#   build_pr_tests.sh <master folder>/Entity Local Experiments
set -uo pipefail
E="$1"; BR="$E/worktrees/builds/corr"
pids=()
for name in pr-tests-single pr-tests-double pr-tests-bounds-single pr-tests-bounds-double; do
  (
    cmake --build "$BR/$name" -j 2 --target test-archetypes-bethe_heitler.xc > "$BR/$name.build.log" 2>&1; rc=$?
    echo "$(date +%H:%M:%S) build $name rc=$rc errors=$(grep -c 'error:' "$BR/$name.build.log") warnings=$(grep -c 'warning:' "$BR/$name.build.log")"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
grep -h "CMAKE_CXX_FLAGS:\|precision:\|^precision" "$BR"/pr-tests-*/CMakeCache.txt | sort | uniq -c
