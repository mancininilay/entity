#!/usr/bin/env bash
# Tier A (the unit test) of the four trees at 1, 2 and 8 OpenMP threads. The four trees run side by
# side, the three thread counts of a tree one after the other. Logs: tierA/<tree>-t<threads>.log
#   run_tierA.sh <master folder>/Entity Local Experiments <output directory>
set -uo pipefail
E="$1"; OUT="$2"; BR="$E/worktrees/builds/corr"
pids=()
for name in pr-tests-single pr-tests-double pr-tests-bounds-single pr-tests-bounds-double; do
  (
    for t in 8 2 1; do
      log="$OUT/$name-t$t.log"
      s=$SECONDS
      ( cd "$BR/$name" && OMP_NUM_THREADS=$t OMP_PROC_BIND=false ./tests/archetypes/test-archetypes-bethe_heitler.xc ) > "$log" 2>&1
      rc=$?
      echo "rc=$rc" >> "$log"
      echo "$name threads=$t rc=$rc seconds=$((SECONDS - s))"
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
