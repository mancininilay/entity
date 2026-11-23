#!/usr/bin/env bash
# After a run of the resolution pair (Rusty login node rustyamd2, no Slurm, niced), run by run:
#   analyze.sh <run directory> [<run directory> ...]
# Into <run directory>/analysis/:
#   check1.txt   check1_counters.py (the run against R-off of stage B for the counters of a run
#                without the policy)
#   check2.txt   check2_counts.py with the bhref table of the 1.8 bath, the ions scaled by the
#                counters (registered for the pair; runs with the emitter first only);
#                check2-spectrum.txt the registered check of stages B and C, reported
#   check4.txt   check4_ref.py against the three policy-off runs of stages B and C (registered for
#                the pair; present grid and one card only, the fine grid has another kappa);
#                check4-r-off.txt check4_gauss.py against R-off alone, reported
#   gauss.txt    gauss_level.py: the level of the residual in units of one filtered charge
#   upstream.txt upstream.py (the delivered upstream: density, B^2, proxy, against time; with
#                --c5, check 5 of the pre-registration, for the runs s1-*)
#   scalars.json, interface.json   the late scalars of check 3 and Codex 1's interface measure, by
#                the scripts of the decomposition question (runs with field output)
#   ranks.txt    ranks_check.py (check 1 per rank: the sums of the ranks, no species above 90
#                percent of its capacity on a rank)
#   core.*       core_series.py (the core series: one centre, physical footprints, the frame of
#                the pairs), read by compare_pair.py
#   summary.txt  the exit codes
# For lf-ctl and lf-neg only the level of the Gauss residual (gauss.txt); their registered check is
# lf_check.py, run on both at once.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$HOME/ceph/bh-resolution-pair-claude1-20261123"
src="$R/src/pgens/reconnection_bh"
SC="$src/scripts"; DC="$src/mpi/decomposition"
J="$src/data/j_table_mono18.txt"
OFF="$HOME/ceph/bh-reconnection-claude1-20261121/runs/r-off-7520718"
OFFS=("$OFF" "$HOME/ceph/bh-reconnection-claude1-20261121/stage-c/runs/r-off-b-7523492" "$HOME/ceph/bh-reconnection-claude1-20261121/stage-c/runs/r-off-c-7523494")
BP="$HOME/ceph/bh-reconnection-claude1-20261121/stage-c/bp2raw-login"
module load python/3.11.11 >/dev/null 2>&1
# the analysis scripts must be those fixed by the pre-registration (series/SCRIPTS.sha256)
(cd "$src" && sha256sum --quiet -c series/SCRIPTS.sha256) || { echo "the analysis scripts are not the registered ones"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$BP" ]] || { echo "no bp2raw wrapper"; exit 1; }
for run in "$@"; do
  name=$(basename "$run"); name=${name%-*}
  out="$run/analysis"; mkdir -p "$out" || exit 1
  CACHE=$(mktemp -d "$R/scratch-cache.XXXXXX") || exit 1
  log="$out/analysis.log"
  { hostname; date -u; echo "run $run (exit status $(cat "$run/exit-status" 2>/dev/null))"; echo "commit $(git -C "$R/src" rev-parse HEAD)"
    (cd "$SC" && sha256sum check1_counters.py check2_paircount.py check4_gauss.py check6_position.py bhio.py)
    (cd "$HERE/.." && sha256sum ranks_check.py core_series.py compare_pair.py gauss_level.py lf_check.py check2_counts.py check4_ref.py)
    (cd "$DC" && sha256sum scalars.py interface.py check3_ensemble.py); sha256sum "$J" "$HERE/../upstream.py"; } > "$log" 2>&1
  : > "$out/summary.txt"
  case "$name" in
    lf-ctl|lf-neg) python3 -B "$HERE/../gauss_level.py" "$run" > "$out/gauss.txt" 2>> "$log"; echo "gauss exit $?" >> "$out/summary.txt" ;;
    *)
      PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check1_counters.py" "$BP" "$CACHE" "$run" "$OFF" > "$out/check1.txt" 2>> "$log"; echo "check1 exit $?" >> "$out/summary.txt"
      [[ -s "$run/bh_ranks.txt" ]] && { python3 -B "$HERE/../ranks_check.py" "$run" > "$out/ranks.txt" 2>> "$log"; echo "ranks exit $?" >> "$out/summary.txt"; }
      if [[ "$name" != r-low-1card ]]; then
        nice -n 10 python3 -B "$HERE/../check2_counts.py" "$BP" "$CACHE" "$run" "$J" > "$out/check2.txt" 2>> "$log"; echo "check2 exit $?" >> "$out/summary.txt"
        PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check2_paircount.py" "$BP" "$CACHE" "$run" "$J" > "$out/check2-spectrum.txt" 2>> "$log"; echo "check2-spectrum exit $?" >> "$out/summary.txt"
      fi
      case "$name" in
        s1-f-*) ;;
        *) python3 -B "$HERE/../check4_ref.py" "$run" "${OFFS[@]}" > "$out/check4.txt" 2>> "$log"; echo "check4 exit $?" >> "$out/summary.txt"
           PYTHONPATH="$SC" nice -n 10 python3 -B "$SC/check4_gauss.py" "$BP" "$CACHE" "$OFF" "$run" > "$out/check4-r-off.txt" 2>> "$log"; echo "check4-r-off exit $?" >> "$out/summary.txt" ;;
      esac
      [[ -s "$run/bh_gauss.txt" ]] && { python3 -B "$HERE/../gauss_level.py" "$OFF" "$run" > "$out/gauss.txt" 2>> "$log"; echo "gauss exit $?" >> "$out/summary.txt"; }
      case "$name" in
        s1-*) c5=--c5 ;;
        *) c5= ;;
      esac
      [[ -s "$run/bh_profile.txt" ]] && { python3 -B "$HERE/../upstream.py" "$run" $c5 > "$out/upstream.txt" 2>> "$log"; echo "upstream${c5:+ (check 5)} exit $?" >> "$out/summary.txt"; }
      PYTHONPATH="$SC:$DC" nice -n 10 python3 -B "$DC/scalars.py" "$BP" "$CACHE" "$run" "$out/scalars.json" > "$out/scalars.log" 2>&1; echo "scalars exit $?" >> "$out/summary.txt"
      PYTHONPATH="$SC:$DC" nice -n 10 python3 -B "$DC/interface.py" "$BP" "$CACHE" "$run" "$out/interface.json" > "$out/interface.log" 2>&1; echo "interface exit $?" >> "$out/summary.txt"
      [[ "$name" == r-low-1card ]] || { nice -n 10 python3 -B "$HERE/../core_series.py" "$BP" "$CACHE" "$run" "$out/core" > "$out/core.log" 2>&1; echo "core_series exit $?" >> "$out/summary.txt"; }
      ;;
  esac
  rm -rf -- "$CACHE"
  date -u >> "$log"; echo "ANALYSIS DONE $name" >> "$log"
  cat "$out/summary.txt"
done
