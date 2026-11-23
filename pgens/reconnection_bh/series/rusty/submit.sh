#!/usr/bin/env bash
# Submission of one job of the resolution pair (Claude 1, 2026-11-23), from a login node:
#   submit.sh build [--test-only]
#   submit.sh <name> <wall limit, hh:mm:ss> <gpu|gpupreempt> [--test-only]
# The cap of the engine is the wall limit less 10 minutes (less 5 below an hour). The budget
# rule of the brief is applied before the submission: over the jobs listed in jobs.txt, the
# A100-hours charged by those that have ended (elapsed time times GPUs) plus the GPUs times the
# wall limit of those still pending or running, plus the GPUs times the wall limit of the new
# job, must not exceed BUDGET (75); otherwise nothing is submitted. Every submission (not a test)
# is appended to jobs.txt.
set -uo pipefail
R="$HOME/ceph/bh-resolution-pair-claude1-20261123"
S="$R/src/pgens/reconnection_bh/series/rusty"
BUDGET=75
cd "$R" || exit 1
mkdir -p slurm
secs() { IFS=: read -r h m s <<< "$1"; echo $((10#$h * 3600 + 10#$m * 60 + 10#$s)); }
used() { # A100-hours: ended jobs at their elapsed time, pending or running ones at their wall limit
  local ids; ids=$(awk '{print $2}' jobs.txt 2>/dev/null | paste -sd, -)
  [[ -z "$ids" ]] && { echo 0; return; }
  sacct -n -X -j "$ids" -o State%20,ElapsedRaw,TimelimitRaw,ReqTRES%80 | awk '
    { g = 0; if (match($4, /gres\/gpu=[0-9]+/)) g = substr($4, RSTART + 9, RLENGTH - 9)
      if ($1 ~ /PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING/) s += 60 * $3 * g; else s += $2 * g }
    END { printf "%.3f", s / 3600 }'
}
if [[ "${1:-}" == build ]]; then
  test="${2:-}"; ngpu=1; limit=00:55:00
  args=(--job-name=bhpair-build "$S/build.sbatch")
else
  name="${1:?name}"; limit="${2:?wall limit hh:mm:ss}"; part="${3:?gpu or gpupreempt}"; test="${4:-}"
  [[ "$limit" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || { echo "wall limit as hh:mm:ss"; exit 2; }
  l=$(secs "$limit"); m=600; (( l <= 3600 )) && m=300
  cap=$((l - m))
  if [[ "$name" == r-low-1card ]]; then shape=(--ntasks=1 --gpus-per-node=1 --mem=96G); ngpu=1; else shape=(); ngpu=4; fi
  case "$part" in
    gpu) q=() ;;
    gpupreempt) q=(--qos=gpupreempt) ;;
    *) echo "partition gpu or gpupreempt"; exit 2 ;;
  esac
  (( l > 3600 )) && [[ "$part" == gpupreempt ]] && { echo "a wall limit above one hour goes to the regular gpu partition (brief)"; exit 2; }
  args=(--job-name="bhpair-$name" --time="$limit" --partition="$part" "${q[@]}" "${shape[@]}" "$S/run.sbatch" "$name" "$cap")
fi
u=$(used); ask=$(awk -v g="$ngpu" -v l="$(secs "$limit")" 'BEGIN { printf "%.3f", g * l / 3600 }')
echo "A100-hours charged or reserved so far: $u; this job at most $ask; budget $BUDGET"
awk -v u="$u" -v a="$ask" -v b="$BUDGET" 'BEGIN { exit !(u + a <= b) }' || { echo "over the budget: not submitted"; exit 3; }
if [[ "$test" == --test-only ]]; then
  sbatch --test-only "${args[@]}"
else
  out=$(sbatch --parsable "${args[@]}") || { echo "sbatch failed: $out"; exit 1; }
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${out%%;*} ${args[*]}" >> jobs.txt
  echo "submitted ${out%%;*}"
fi
