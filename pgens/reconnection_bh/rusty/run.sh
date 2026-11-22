#!/usr/bin/env bash
# One run of the shakedown (r-off or r-low) on one A100: the executable of the
# build marked usable, the input of the commit, a fresh run directory, a time
# cap below the wall limit of the job. Nothing is retried; the exit status of
# the engine is kept.
#   run.sh <r-off|r-low>
# Started by r-off.sbatch or r-low.sbatch from the clone of the commit.
# BHR_ROOT and BHR_CAP exist only for the dry run on the login node.
set -uo pipefail
name="$1"
[[ "$name" == r-off || "$name" == r-low ]] || { echo "run.sh: r-off or r-low"; exit 2; }
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}"
cap="${BHR_CAP:-21000}"
src="$R/src"
want=$(cat "$R/COMMIT") || exit 1
bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
x="$bld/tree/src/entity.xc"
run="$R/runs/$name-$SLURM_JOB_ID"
mkdir "$run" || exit 1
exec > "$run/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$run/exit-status"' EXIT
T0=$SECONDS
hostname; date -u
module list 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
grep Cpus_allowed_list /proc/self/status
[[ $(git -C "$src" rev-parse HEAD) == "$want" && -z $(git -C "$src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
[[ -f "$bld/BUILD_OK" && -x "$x" && $(sha256sum "$x" | cut -d' ' -f1) == $(cat "$bld/BUILD_OK") ]] || { echo "the executable is not the one marked usable"; exit 1; }
echo "source: $src at $want"; echo "build: $bld"; echo "executable sha256: $(cat "$bld/BUILD_OK")"
cp "$src/pgens/reconnection_bh/$name.toml" "$run/$name.toml" || exit 1
sha256sum "$run/$name.toml"
df -h "$R" | tail -1
echo "==== run $name (cap $cap s)"
rc=0
( cd "$run" && timeout --kill-after=60 "$cap" "$x" -input "$name.toml" > run.log 2>&1 ) || rc=$?
echo "RUN $name rc=$rc seconds=$((SECONDS - T0)) cap=$cap"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
grep -h 'Bethe-Heitler emission' "$run/run.log" | head -1
tail -25 "$run/run.log"
echo "==== what was written"
for k in fields particles spectra; do
  echo "$k: $(ls "$run/$name/$k" 2>/dev/null | wc -l) outputs, $(du -sh "$run/$name/$k" 2>/dev/null | cut -f1)"
done
du -sh "$run" | cut -f1
wc -l "$run/bh_counters.txt" "$run/bh_gauss.txt" 2>/dev/null
tail -1 "$run/bh_counters.txt" 2>/dev/null | cut -c1-400
tail -1 "$run/bh_gauss.txt" 2>/dev/null
exit $rc
