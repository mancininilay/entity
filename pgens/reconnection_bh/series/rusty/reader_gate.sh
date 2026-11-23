#!/usr/bin/env bash
# The reader gate, on the login node rustyamd2 after the build job (no Slurm): reader_gate.py on
# the smoke run of the build marked usable, then check 1 and the per-rank check on it.
#   reader_gate.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$HOME/ceph/bh-resolution-pair-claude1-20261123"
BP="$HOME/ceph/bh-reconnection-claude1-20261121/stage-c/bp2raw-login"
bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
d="$bld/smoke/toy-s1"
module load python/3.11.11 >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
CACHE=$(mktemp -d "$R/scratch-cache.XXXXXX") || exit 1
out="$bld/reader-gate.txt"
{ hostname; date -u; echo "build $bld, commit $(cat "$bld/COMMIT")"
  python3 -B "$HERE/../reader_gate.py" "$BP" "$CACHE" "$d"; echo "reader_gate exit $?"
  PYTHONPATH="$HERE/../../scripts" python3 -B "$HERE/../../scripts/check1_counters.py" "$BP" "$CACHE" "$d"; echo "check1 exit $?"
  python3 -B "$HERE/../ranks_check.py" "$d"; echo "ranks exit $?"
  python3 -B "$HERE/../gauss_level.py" "$d"; echo "gauss exit $?"; } > "$out" 2>&1
rm -rf -- "$CACHE"
cat "$out"
