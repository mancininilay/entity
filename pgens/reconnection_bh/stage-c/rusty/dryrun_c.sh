#!/usr/bin/env bash
# Dry run of the stage-C driver on the login node, end to end, through run_c.sbatch (run as a
# plain script): a temporary root with a one-commit source, a stub executable marked usable,
# the five registered inputs. The stub writes lines of bh_counters.txt as the engine does.
# Four cases: a healthy run; a run whose counters show particle-steps above the tables (the
# flag must stop it); a run with one unreadable line among healthy ones (the flag must not
# stop it); a run whose pairs are hot (the flag must stop it). No Slurm, no GPU.
#   dryrun_c.sh <directory with run_c.sh, run_c.sbatch> <directory with the five inputs>
set -uo pipefail
J="$1"; I="$2"
T=$(mktemp -d)
mkdir -p "$T/src" "$T/stage-c/jobs" "$T/stage-c/inputs" "$T/stage-c/slurm" "$T/build-dry/tree/src"
echo stub > "$T/src/file"
( cd "$T/src" && git init -q . && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m stub )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
echo "$T/build-dry" > "$T/BUILD"
cp "$J/run_c.sh" "$J/run_c.sbatch" "$T/stage-c/jobs/"
cp "$I"/r-*.toml "$T/stage-c/inputs/"
( cd "$T/stage-c" && sha256sum inputs/r-*.toml > INPUTS.sha256 )
good="100 8.63 50 0 0 0 0 90.0 37579570 37579570 193802117.4 264.6 37579575 37579575 193846309.3 264.6 150000 150000 450000.5 300.1 150000 150000 460000.25 310.2"
stub() { # $1 case
  cat > "$T/build-dry/tree/src/entity.xc" <<STUB
#!/bin/bash
echo "stub engine: \$@"
echo "# steps time pairs ..." > bh_counters.txt
case "$1" in
  healthy)   for i in 1 2 3 4 5 6; do echo "$good" >> bh_counters.txt; sleep 1; done ;;
  tables)    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$good" | awk '{\$4 = 3; print}' >> bh_counters.txt; sleep 1; done ;;
  garbled)   echo "$good" >> bh_counters.txt; sleep 2; echo "$good" | awk '{\$24 = "3.1e"; print}' >> bh_counters.txt; sleep 1.2; for i in 1 2 3 4; do echo "$good" >> bh_counters.txt; sleep 1; done ;;
  hot)       for i in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$good" | awk '{\$19 = 3.0e7; \$23 = 3.0e7; print}' >> bh_counters.txt; sleep 1; done ;;
  nan)       for i in 1 2 3 4 5 6 7 8 9 10 11 12; do echo "$good" | awk '{\$11 = "nan"; print}' >> bh_counters.txt; sleep 1; done ;;
esac
exit 0
STUB
  chmod +x "$T/build-dry/tree/src/entity.xc"
  sha256sum "$T/build-dry/tree/src/entity.xc" | cut -d' ' -f1 > "$T/build-dry/BUILD_OK"
}
show() { grep -v -E '^ *[0-9]+\)|^$|Where:|Sticky|Currently Loaded|NVIDIA-SMI has failed|nvidia-smi' "$1"; }
k=0
for c in healthy:r-mid tables:r-low-b garbled:r-off-b hot:r-mid nan:r-off-c; do
  k=$((k + 1)); case_="${c%%:*}"; name="${c##*:}"
  stub "$case_"
  echo "==== case $case_ ($name)"
  ( cd "$T/stage-c" && SLURM_JOB_ID=dry$k BHR_ROOT="$T" BHR_FLAG_PERIOD=1 timeout 120 bash jobs/run_c.sbatch "$name" 60 ); echo "run_c.sbatch rc=$? (124: did not return)"
  show "$T/stage-c/runs/$name-dry$k/driver.log" | grep -E "RUN |Exit status|stop flag|stopped by|input|sha256|executable|source" | cut -c1-200
  ls "$T/stage-c/runs/$name-dry$k"
done
echo "==== a changed input must be refused"
echo "# changed" >> "$T/stage-c/inputs/r-mid.toml"
stub healthy
( cd "$T/stage-c" && SLURM_JOB_ID=dry9 BHR_ROOT="$T" BHR_FLAG_PERIOD=1 timeout 120 bash jobs/run_c.sbatch r-mid 60 ); echo "rc=$?"
grep -E "registered|Exit status" "$T/stage-c/runs/r-mid-dry9/driver.log"
echo "==== an unknown name must be refused"
( cd "$T/stage-c" && SLURM_JOB_ID=dry10 BHR_ROOT="$T" timeout 120 bash jobs/run_c.sbatch r-high 60 ); echo "rc=$?"
rm -rf "${T:?}"
