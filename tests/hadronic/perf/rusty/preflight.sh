#!/bin/bash
# Preflight for p2-perf-a100.sbatch (run on Rusty by Codex 1 before submission; submits NOTHING):
#   1. creates the pre-existing Slurm log parent the #SBATCH --output/--error lines need (Codex 1's F2.1);
#   2. checks that the run root of the prospective job does not exist (the job creates it exclusively);
#   3. (round 4) dry-runs the job with the part configuration exported in the calling shell (PERF_PART and, for an engine
#      part, its slice: perf_budget.py --plan-env), in a temporary root: the job's own guard must accept it (exit 0);
#   4. runs `sbatch --test-only` with the given commit, which validates the resource request only.
# Usage: PERF_PART=... [PERF_N=... PERF_REPEAT_FIRST=... ...] preflight.sh <ENTITY_COMMIT>
set -u
COMMIT="${1:?usage: preflight.sh <ENTITY_COMMIT>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HOME/ceph/entity-validation/p2-perf-logs"
mkdir -p "$LOGDIR" && echo "log parent: $LOGDIR ($(ls "$LOGDIR" | wc -l) files)"
echo "run roots present: $(ls -d "$HOME"/ceph/entity-validation/p2-perf-* 2>/dev/null | grep -v p2-perf-logs | wc -l) (each job creates p2-perf-<jobid> exclusively)"
DRY="$(mktemp -d "${TMPDIR:-/tmp}/p2perf-preflight.XXXXXX")"
if env -u PERF_SELFTEST DRY_RUN=1 ENTITY_COMMIT="$COMMIT" PERF_ROOT="$DRY/root" bash "$HERE/p2-perf-a100.sbatch" > "$DRY/dry-run.log" 2>&1; then
  echo "part configuration accepted (dry run $DRY/dry-run.log):"; awk -F'\t' '{ printf "  %-40s %6s s  %s\n", $1, $3, $6 }' "$DRY/root/logs/steps.tsv"
else
  echo "part configuration REFUSED (dry run $DRY/dry-run.log):"; tail -3 "$DRY/dry-run.log"; exit 7
fi
echo "sbatch --test-only:"
sbatch --test-only --export=ALL,ENTITY_COMMIT="$COMMIT" "$HERE/p2-perf-a100.sbatch"
