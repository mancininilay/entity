#!/usr/bin/env bash
# One run of the decomposition question: an input of this folder on four A100s of one node,
# four ranks, one GPU per rank, with the executable of the build marked usable (the build of
# the run of M4). A fresh run directory, a time cap below the wall limit of the job. Nothing
# is retried; the exit status of the engine is kept.
#   run_d.sh <name of the input without .toml> <cap in seconds>
# BHM_ROOT and BHM_SHARE_GPU exist only for the dry run on the login node.
set -uo pipefail
R="${BHM_ROOT:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}"
V="$R/src-val/pgens/bh_validation/mpi-gpu/rusty"
source "$V/env.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
name="${1:?name of the run}"; cap="${2:?cap in seconds}"
case "$name" in
  r-low-mpi-b22|r-low-mpi-c22|r-low-mpi-a41|r-low-mpi-b41) ;;
  *) echo "unknown run $name"; exit 1 ;;
esac
src="$R/src-rec-d"
D="$src/pgens/reconnection_bh/mpi/decomposition"
want=$(cat "$R/COMMIT-rec-d") || exit 1
built=$(cat "$R/COMMIT-rec") || exit 1
bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
x="$bld/rec-single/src/entity.xc"
run="$R/runs/$name-$SLURM_JOB_ID"
mkdir -p "$R/runs" && mkdir "$run" || exit 1
exec > "$run/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$run/exit-status"' EXIT
T0=$SECONDS
hostname; date -u
module list 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
grep Cpus_allowed_list /proc/self/status
[[ $(git -C "$src" rev-parse HEAD) == "$want" && -z $(git -C "$src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
[[ -f "$bld/rec-single/BUILD_OK" && -x "$x" && $(sha256sum "$x" | cut -d' ' -f1) == $(cat "$bld/rec-single/BUILD_OK") ]] || { echo "the executable is not the one marked usable"; exit 1; }
# the executable was built from $built: between it and the commit of the runs only files of this folder may differ
other=$(git -C "$src" diff --name-only "$built" "$want" | grep -v '^pgens/reconnection_bh/mpi/decomposition/' || true)
[[ -z "$other" ]] || { echo "files outside mpi/decomposition differ from the commit of the build:"; echo "$other"; exit 1; }
echo "source: $src at $want; the executable was built from $built (only mpi/decomposition differs)"
echo "build: $bld"; echo "executable sha256: $(cat "$bld/rec-single/BUILD_OK")"
(cd "$D" && grep " $name.toml\$" INPUTS.sha256 | sha256sum -c -) || { echo "the input is not the registered one"; exit 1; }
IFS=, read -ra devs <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ ${#devs[@]} == 4 || -n "${BHM_SHARE_GPU:-}" ]] || { echo "${#devs[@]} GPUs visible, 4 needed"; exit 1; }
cp "$D/$name.toml" "$run/$name.toml" || exit 1
sha256sum "$run/$name.toml"
df -h "$R" | tail -1
echo "==== run $name on 4 ranks (cap $cap s)"
rc=0
( cd "$run" && BHM_GPU_LOG="$run/gpu" timeout --kill-after=60 "$cap" mpirun --bind-to none --map-by ppr:4:node -n 4 \
    "$V/rank-gpu.sh" "$x" -input "$name.toml" > run.log 2>&1 ) || rc=$?
echo "RUN $name rc=$rc seconds=$((SECONDS - T0)) cap=$cap"
cat "$run"/gpu.* 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
grep -h 'Bethe-Heitler emission' "$run/run.log" | head -4
tail -25 "$run/run.log"
echo "==== what was written"
for k in fields particles spectra; do
  echo "$k: $(ls "$run/$name/$k" 2>/dev/null | wc -l) outputs, $(du -sh "$run/$name/$k" 2>/dev/null | cut -f1)"
done
ls -d "$run"/*.ckpt 2>/dev/null && echo "a checkpoint directory exists"
du -sh "$run" | cut -f1
wc -l "$run/bh_counters.txt" "$run/bh_ranks.txt" 2>/dev/null
tail -1 "$run/bh_counters.txt" 2>/dev/null | cut -c1-400
tail -4 "$run/bh_ranks.txt" 2>/dev/null
exit $rc
