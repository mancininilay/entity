#!/usr/bin/env bash
# Submits one run of the decomposition question, or asks Slurm when it would start.
#   submit_d.sh test|submit <b22|c22|a41|b41>
# Wall limits (the cap of 12 A100 GPU-hours holds by construction: 2 x 40 + 2 x 49 minutes on
# four cards are 11.87 GPU-hours): [2, 2] 40 minutes (the run of M4 took 33), [4, 1] 49 minutes
# (its fullest rank holds more particles). The engine is stopped 120 s before the limit.
set -uo pipefail
mode="${1:?test or submit}"; tag="${2:?b22, c22, a41 or b41}"
case "$tag" in
  b22|c22) limit=00:40:00; cap=2280 ;;
  a41|b41) limit=00:49:00; cap=2820 ;;
  *) echo "unknown run $tag"; exit 1 ;;
esac
R="$HOME/ceph/bh-mpi-gpu-claude1-20261122"
cd "$R" || exit 1
s="src-rec-d/pgens/reconnection_bh/mpi/decomposition/rusty/run_d.sbatch"
case "$mode" in
  test)   sbatch --test-only --job-name="bhd-$tag" --time="$limit" "$s" "r-low-mpi-$tag" "$cap" ;;
  submit) sbatch --parsable --job-name="bhd-$tag" --time="$limit" "$s" "r-low-mpi-$tag" "$cap" ;;
  *) echo "test or submit"; exit 1 ;;
esac
