#!/usr/bin/env bash
# Stage B, before the build job (login node, no Slurm): a fresh clone of the
# fork at the commit to be run, in the root of the shakedown. The jobs are then
# submitted from that root with the sbatch files of the clone, which start
# the drivers of the clone: what runs is what the commit holds.
#   prepare.sh <commit>
set -uo pipefail
want="$1"
R="${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}"
mkdir -p "$R/slurm" "$R/runs" || exit 1
[[ ! -e "$R/src" ]] || { echo "$R/src exists already"; exit 1; }
git clone -q --branch dev/bh-reconnection --single-branch https://github.com/mancininilay/entity.git "$R/src" || exit 1
git -C "$R/src" checkout -q --detach "$want" || exit 1
[[ $(git -C "$R/src" rev-parse HEAD) == "$want" && -z $(git -C "$R/src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
echo "$want" > "$R/COMMIT"
echo "source: $R/src at $want"
