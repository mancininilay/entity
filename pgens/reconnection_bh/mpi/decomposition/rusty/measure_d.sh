#!/usr/bin/env bash
# After a run, or for an existing run (Rusty login node, no Slurm, niced): the scalars and the
# interface measure of one run as JSON files; with "checks" also checks 1 and 2 of the
# shakedown, the fill per rank and the timers, as for the run of M4.
#   measure_d.sh <output directory> <run directory> [checks]
# The raw extractions go to a temporary directory of this node, removed at the end.
set -uo pipefail
out="$1"; RUN="$2"; checks="${3:-}"
R="$HOME/ceph/bh-mpi-gpu-claude1-20261122"
D="$R/src-rec-d/pgens/reconnection_bh/mpi/decomposition"
label=$(basename "$(dirname "$(ls -d "$RUN"/*/fields | head -1)")") || exit 1
mkdir -p "$out/json" "$out/$label" || exit 1
source "$R/jobs/login_env.sh" "$out/$label" || exit 1
export PYTHONPATH="$D:$S"
CACHE=$(mktemp -d "${BHM_CACHE_ROOT:-/tmp}/bhd-cache.XXXXXX") || exit 1
trap 'rm -rf "$CACHE"' EXIT
log="$out/$label/measure.log"
{
  hostname; date -u
  python3 -c "import sys, numpy; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
  echo "commit $(cat "$R/COMMIT-rec-d"); build $bld"; echo "run: $RUN"
  (cd "$D" && sha256sum -c SCRIPTS.sha256)
} > "$log" 2>&1
(cd "$D" && sha256sum -c --quiet SCRIPTS.sha256) || { echo "$label: a script is not the registered one" | tee -a "$log"; exit 1; }
nice -n 10 python3 -B "$D/scalars.py" "$WRAP" "$CACHE" "$RUN" "$out/json/$label.scalars.json" > "$out/$label/scalars.txt" 2>&1
echo "scalars exit $?" >> "$log"
nice -n 10 python3 -B "$D/interface.py" "$WRAP" "$CACHE" "$RUN" "$out/json/$label.interface.json" > "$out/$label/interface.txt" 2>&1
echo "interface exit $?" >> "$log"
if [[ "$checks" == checks ]]; then
  nice -n 10 python3 -B "$S/check1_counters.py" "$WRAP" "$CACHE" "$RUN" > "$out/$label/check1.txt" 2>&1; echo "check1 exit $?" >> "$log"
  nice -n 10 python3 -B "$S/check2_paircount.py" "$WRAP" "$CACHE" "$RUN" "$J" > "$out/$label/check2.txt" 2>&1; echo "check2 exit $?" >> "$log"
  python3 -B "$R/jobs/m4_ranks.py" "$RUN" > "$out/$label/ranks.txt" 2>&1; echo "ranks exit $?" >> "$log"
  # in the logs of the runs with MPI on GPUs the step lines follow the engine's terminal control codes
  sed 's/^\(\x1b\[[0-9;]*[A-Za-z]\)*//' "$RUN/run.log" > "$CACHE/run-plain.log"
  awk -f "$R/jobs/timers.awk" "$CACHE/run-plain.log" > "$out/$label/timers.txt"
  secs=$(grep -o "RUN $label rc=[0-9]* seconds=[0-9]*" "$RUN/driver.log" | grep -o 'seconds=[0-9]*' | cut -d= -f2)
  python3 -B "$R/jobs/m4_cost.py" "$RUN/bh_counters.txt" "$out/$label/timers.txt" "${secs:-0}" 4 > "$out/$label/cost.txt" 2>&1; echo "cost exit $?" >> "$log"
fi
date -u >> "$log"
echo "MEASURE DONE" >> "$log"
tail -n +6 "$log"
