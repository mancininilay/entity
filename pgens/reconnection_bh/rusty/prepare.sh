#!/usr/bin/env bash
# Stage B, before the build job (login node, no Slurm): a clone of the fork
# at the commit to be run, in the root of the shakedown (a fresh clone, or
# the existing clean clone moved to that commit). The jobs are then
# submitted from that root with the sbatch files of the clone, which start
# the drivers of the clone: what runs is what the commit holds.
#   prepare.sh <commit>
set -uo pipefail
want="$1"
R="${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}"
mkdir -p "$R/slurm" "$R/runs" || exit 1
if [[ -e "$R/src" ]]; then
  # a clone of an earlier commit: it must be clean; it is moved to the new commit
  [[ -z $(git -C "$R/src" status --porcelain) ]] || { echo "$R/src is not clean"; exit 1; }
  git -C "$R/src" fetch -q origin dev/bh-reconnection || exit 1
else
  git clone -q --branch dev/bh-reconnection --single-branch https://github.com/mancininilay/entity.git "$R/src" || exit 1
fi
git -C "$R/src" checkout -q --detach "$want" || exit 1
[[ $(git -C "$R/src" rev-parse HEAD) == "$want" && -z $(git -C "$R/src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
echo "$want" > "$R/COMMIT"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $want" >> "$R/COMMIT.history"
echo "source: $R/src at $want"
