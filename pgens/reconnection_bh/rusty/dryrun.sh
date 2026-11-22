#!/usr/bin/env bash
# Dry run of the three drivers on the login node, end to end: a temporary root
# whose source is a one-commit git repository holding this directory and the
# three inputs of the real tree; BHR_STUB_RC makes build.sh write a stub
# executable instead of compiling. The drivers are started through the three
# sbatch files (run as plain scripts: their last line is what the job
# executes). No Slurm, no GPU, no compiler.
#   dryrun.sh <real source tree> <exit status of the stub>
set -uo pipefail
real="$1"; stub_rc="${2:-0}"
T=$(mktemp -d)
mkdir -p "$T/src/pgens/reconnection_bh" "$T/runs" "$T/slurm"
cp "$real/pgens/reconnection_bh/"{r-off,r-low,toy-low}.toml "$T/src/pgens/reconnection_bh/"
cp -r "$real/pgens/reconnection_bh/rusty" "$T/src/pgens/reconnection_bh/"
( cd "$T/src" && git init -q . && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m stub )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
J="$T/src/pgens/reconnection_bh/rusty"
show() { grep -v -E '^ *[0-9]+\)|^$|Where:|Sticky|Currently Loaded|NVIDIA-SMI has failed' "$1"; }
echo "==== build.sbatch"
( cd "$T" && SLURM_JOB_ID=dry1 SLURM_CPUS_PER_TASK=16 BHR_ROOT="$T" BHR_STUB_RC="$stub_rc" timeout 120 bash "$J/build.sbatch" ); echo "build.sbatch rc=$? (124: did not return)"
show "$T/build-dry1/driver.log"
for n in r-off r-low; do
  echo "==== $n.sbatch"
  ( cd "$T" && SLURM_JOB_ID=dry2 BHR_ROOT="$T" BHR_CAP=60 timeout 120 bash "$J/$n.sbatch" ); echo "$n.sbatch rc=$? (124: did not return)"
  [[ -f "$T/runs/$n-dry2/driver.log" ]] && show "$T/runs/$n-dry2/driver.log"
done
echo "==== files"
( cd "$T" && find . -path ./src/.git -prune -o -type f -print | sort )
rm -rf "$T"
