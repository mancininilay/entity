#!/usr/bin/env bash
# After a run of the resolution pair (Rusty login node rustyamd2, no Slurm, niced), run by run:
#   analyze.sh <run directory> [<run directory> ...]
# Into <run directory>/analysis/:
#   check1.txt   check1_counters.py (the run against R-off of stage B for the counters of a run
#                without the policy)
#   check2.txt   check2_paircount.py with the bhref table of the 1.8 bath (runs with the emitter
#                first only)
#   check4.txt   check4_gauss.py against R-off of stage B (7520718)
#   upstream.txt upstream.py (the delivered upstream: density, B^2, proxy, against time)
#   scalars.json, interface.json   the late scalars of check 3 and Codex 1's interface measure, by
#                the scripts of the decomposition question (runs with field output)
#   series.*     check6_series.py (the core series for check 6 by position)
#   summary.txt  the exit codes
# For lf-ctl and lf-neg only the Gauss files are compared (lf.txt in the first directory given).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$HOME/ceph/bh-resolution-pair-claude1-20261123"
src="$R/src/pgens/reconnection_bh"
SC="$src/scripts"; DC="$src/mpi/decomposition"
J="$src/data/j_table_mono18.txt"
OFF="$HOME/ceph/bh-reconnection-claude1-20261121/runs/r-off-7520718"
BP="$HOME/ceph/bh-reconnection-claude1-20261121/stage-c/bp2raw-login"
module load python/3.11.11 >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$BP" ]] || { echo "no bp2raw wrapper"; exit 1; }
for run in "$@"; do
  name=$(basename "$run"); name=${name%-*}
  out="$run/analysis"; mkdir -p "$out" || exit 1
  CACHE=$(mktemp -d "$R/scratch-cache.XXXXXX") || exit 1
  log="$out/analysis.log"
  { hostname; date -u; echo "run $run (exit status $(cat "$run/exit-status" 2>/dev/null))"; echo "commit $(git -C "$R/src" rev-parse HEAD)"
    (cd "$SC" && sha256sum check1_counters.py check2_paircount.py check4_gauss.py check6_series.py bhio.py)
    (cd "$DC" && sha256sum scalars.py interface.py check3_ensemble.py); sha256sum "$J" "$HERE/../upstream.py"; } > "$log" 2>&1
  : > "$out/summary.txt"
  case "$name" in
    lf-ctl|lf-neg) ;;
    *)
      PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check1_counters.py" "$BP" "$CACHE" "$run" "$OFF" > "$out/check1.txt" 2>> "$log"; echo "check1 exit $?" >> "$out/summary.txt"
      [[ "$name" == r-low-1card ]] || { PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check2_paircount.py" "$BP" "$CACHE" "$run" "$J" > "$out/check2.txt" 2>> "$log"; echo "check2 exit $?" >> "$out/summary.txt"; }
      PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check4_gauss.py" "$BP" "$CACHE" "$OFF" "$run" > "$out/check4.txt" 2>> "$log"; echo "check4 exit $?" >> "$out/summary.txt"
      [[ -s "$run/bh_profile.txt" ]] && { python3 -B "$HERE/../upstream.py" "$run" > "$out/upstream.txt" 2>> "$log"; echo "upstream exit $?" >> "$out/summary.txt"; }
      PYTHONPATH="$SC:$DC" nice -n 10 python3 -B "$DC/scalars.py" "$BP" "$CACHE" "$run" "$out/scalars.json" > "$out/scalars.log" 2>&1; echo "scalars exit $?" >> "$out/summary.txt"
      PYTHONPATH="$SC:$DC" nice -n 10 python3 -B "$DC/interface.py" "$BP" "$CACHE" "$run" "$out/interface.json" > "$out/interface.log" 2>&1; echo "interface exit $?" >> "$out/summary.txt"
      PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check6_series.py" "$BP" "$CACHE" "$run" "$out/series" > "$out/series.log" 2>&1; echo "check6_series exit $?" >> "$out/summary.txt"
      ;;
  esac
  rm -rf -- "$CACHE"
  date -u >> "$log"; echo "ANALYSIS DONE $name" >> "$log"
  cat "$out/summary.txt"
done
