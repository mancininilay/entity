#!/usr/bin/env bash
# Before the runs (Rusty login node, no Slurm): a fresh clone of the fork at the commit of the
# runs, beside the clone the executable was built from (src-rec, which is not touched). The
# clone starts from the pushed branch dev/bh-reconnection and takes the commits of
# dev/bh-reconnection-mpi from a bundle, or from the fork once the branch is pushed.
#   prepare_d.sh <commit of dev/bh-reconnection-mpi> [<bundle>]
set -uo pipefail
want="$1"; bundle="${2:-}"
R="$HOME/ceph/bh-mpi-gpu-claude1-20261122"
d="$R/src-rec-d"
if [[ ! -e "$d" ]]; then
  git clone -q --branch dev/bh-reconnection --single-branch https://github.com/mancininilay/entity.git "$d" || exit 1
fi
[[ -z $(git -C "$d" status --porcelain) ]] || { echo "src-rec-d is not clean"; exit 1; }
if [[ -n "$bundle" ]]; then
  git -C "$d" bundle verify "$bundle" > /dev/null 2>&1 || { echo "$bundle does not verify"; exit 1; }
  git -C "$d" fetch -q "$bundle" dev/bh-reconnection-mpi || exit 1
else
  git -C "$d" fetch -q origin dev/bh-reconnection-mpi || exit 1
fi
git -C "$d" checkout -q --detach "$want" || exit 1
[[ $(git -C "$d" rev-parse HEAD) == "$want" && -z $(git -C "$d" status --porcelain) ]] || { echo "src-rec-d is not a clean $want"; exit 1; }
built=$(cat "$R/COMMIT-rec") || exit 1
other=$(git -C "$d" diff --name-only "$built" "$want" | grep -v '^pgens/reconnection_bh/mpi/decomposition/' || true)
[[ -z "$other" ]] || { echo "files outside mpi/decomposition differ from the commit of the build:"; echo "$other"; exit 1; }
echo "$want" > "$R/COMMIT-rec-d"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rec-d $want" >> "$R/COMMIT.history"
echo "source: $d at $want (the executable was built from $built; only mpi/decomposition differs)"
