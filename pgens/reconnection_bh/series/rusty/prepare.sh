#!/usr/bin/env bash
# Before a build or a run (Rusty login node, no Slurm): the clean clone `src` of the fork's branch
# dev/bh-resolution-pair at a given commit, under the root of the pair.
#   prepare.sh <commit>
# The code the runs execute must be that of 27ae11a0 (dev/bh-series-prep, refereed by Codex 2):
# only files under pgens/reconnection_bh/series/, pgens/reconnection_bh/mpi/decomposition/ and the
# analysis scripts pgens/reconnection_bh/scripts/ may differ between it and the commit; otherwise
# nothing is prepared.
set -uo pipefail
want="${1:?commit}"
CODE=27ae11a0bcf8630920a1ed2219971de55d5c27d8
R="$HOME/ceph/bh-resolution-pair-claude1-20261123"
d="$R/src"
mkdir -p "$R" "$R/runs" "$R/slurm"
if [[ ! -e "$d" ]]; then
  git clone -q --branch dev/bh-resolution-pair --single-branch https://github.com/mancininilay/entity.git "$d" || exit 1
fi
[[ -z $(git -C "$d" status --porcelain) ]] || { echo "src is not clean"; exit 1; }
git -C "$d" fetch -q origin dev/bh-resolution-pair || exit 1
git -C "$d" fetch -q origin dev/bh-series-prep || exit 1
git -C "$d" checkout -q --detach "$want" || exit 1
[[ $(git -C "$d" rev-parse HEAD) == "$want" && -z $(git -C "$d" status --porcelain) ]] || { echo "src is not a clean $want"; exit 1; }
other=$(git -C "$d" diff --name-only "$CODE" "$want" | grep -v -E '^pgens/reconnection_bh/(series|mpi/decomposition|scripts)/' || true)
[[ -z "$other" ]] || { echo "files outside series/, mpi/decomposition/ and scripts/ differ from $CODE:"; echo "$other"; exit 1; }
echo "$want" > "$R/COMMIT"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $want" >> "$R/COMMIT.history"
echo "source: $d at $want; the code is that of $CODE"
