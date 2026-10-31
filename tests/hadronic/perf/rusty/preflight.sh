#!/bin/bash
# Preflight for p2-perf-a100.sbatch (run on Rusty by Codex 1 before submission; submits NOTHING):
#   1. creates the pre-existing Slurm log parent the #SBATCH --output/--error lines need (Codex 1's F2.1);
#   2. checks that the run root of the prospective job does not exist (the job creates it exclusively);
#   3. runs `sbatch --test-only` with the given commit, which validates the resource request only.
# Usage: preflight.sh <ENTITY_COMMIT>
set -u
COMMIT="${1:?usage: preflight.sh <ENTITY_COMMIT>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HOME/ceph/entity-validation/p2-perf-logs"
mkdir -p "$LOGDIR" && echo "log parent: $LOGDIR ($(ls "$LOGDIR" | wc -l) files)"
echo "run roots present: $(ls -d "$HOME"/ceph/entity-validation/p2-perf-* 2>/dev/null | grep -v p2-perf-logs | wc -l) (each job creates p2-perf-<jobid> exclusively)"
echo "sbatch --test-only:"
sbatch --test-only --export=ALL,ENTITY_COMMIT="$COMMIT" "$HERE/p2-perf-a100.sbatch"
