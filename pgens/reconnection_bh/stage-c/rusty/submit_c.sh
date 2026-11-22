#!/usr/bin/env bash
# Stage C: the five submissions, from a login node, in the stage-C directory. For each run:
# `sbatch --test-only` first (it submits nothing), then the submission. Everything is appended
# to submitted.txt. One A100-80GB and one rank each, gpupreempt, no requeue.
#   submit_c.sh [name ...]        (default: the five runs)
set -uo pipefail
C="$HOME/ceph/bh-reconnection-claude1-20261121/stage-c"
cd "$C" || exit 1
mkdir -p slurm runs
( sha256sum -c INPUTS.sha256 && sha256sum -c SCRIPTS.sha256 ) > hash-check.txt 2>&1 || { echo "hash check failed"; cat hash-check.txt; exit 1; }
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(r-mid r-low-b r-low-c r-off-b r-off-c)
{
  date
  for n in "${names[@]}"; do
    if [ "$n" = r-mid ]; then limit=02:35:00; cap=9000; else limit=02:05:00; cap=7200; fi
    echo "== test-only $n (limit $limit, cap $cap s)"
    sbatch --test-only --job-name="bhrec-$n" --time="$limit" jobs/run_c.sbatch "$n" "$cap" 2>&1
    echo "== submit $n"
    sbatch --job-name="bhrec-$n" --time="$limit" jobs/run_c.sbatch "$n" "$cap" 2>&1
  done
  squeue -u "$USER" -h -o "%i %P %q %j %T %l %M %R" | grep bhrec-
} >> submitted.txt 2>&1
tail -n 30 submitted.txt
