#!/usr/bin/env bash
# One run of stage C (r-mid, r-low-b, r-low-c, r-off-b, r-off-c) on one A100: the executable of
# the build marked usable (unchanged since stage B), the registered input of stage C, a fresh
# run directory, a time cap below the wall limit of the job. Nothing is retried; the exit
# status of the engine is kept.
#   run_c.sh <name> <cap in seconds>
# Started by run_c.sbatch. BHR_ROOT and BHR_FLAG_PERIOD exist only for the dry run.
#
# Stop flag (the driver, not the generator): every 60 s the last line of bh_counters.txt is
# read; the run is stopped (TERM) when on two readings in a row it shows a value that is not a
# number, particle-steps above the tables, pairs dropped for capacity, or a mean K of all
# leptons of the pairs above FLAG_KPAIR (ten times the largest value of R-low, 17.5) with at
# least 1e5 of them present. The reason is written to STOPPED-BY-FLAG.txt.
set -uo pipefail
name="$1"; cap="$2"
case "$name" in r-mid|r-low-b|r-low-c|r-off-b|r-off-c) ;; *) echo "run_c.sh: unknown run $name"; exit 2 ;; esac
[[ "$cap" =~ ^[0-9]+$ ]] || { echo "run_c.sh: cap in seconds"; exit 2; }
FLAG_KPAIR=175
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}"
C="$R/stage-c"
src="$R/src"
want=$(cat "$R/COMMIT") || exit 1
bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
x="$bld/tree/src/entity.xc"
run="$C/runs/$name-$SLURM_JOB_ID"
mkdir -p "$C/runs" || exit 1
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
reg=$(awk -v f="inputs/$name.toml" '$2 == f { print $1 }' "$C/INPUTS.sha256")
[[ -n "$reg" && $(sha256sum "$C/inputs/$name.toml" | cut -d' ' -f1) == "$reg" ]] || { echo "the input is not the registered one"; exit 1; }
cp "$C/inputs/$name.toml" "$run/$name.toml" || exit 1
sha256sum "$run/$name.toml"
df -h "$R" | tail -1

flag() {
  local pid="$1" hits=0 reason
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${BHR_FLAG_PERIOD:-60}"
    [[ -s "$run/bh_counters.txt" ]] || continue
    reason=$(tail -n 1 "$run/bh_counters.txt" | awk -v kmax="$FLAG_KPAIR" '
      NF != 24 { exit }
      { for (i = 1; i <= NF; i++) if ($i !~ /^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$/) { print "column " i " is not a number: " $i; exit } }
      $4 > 0 { print "particle-steps above the tables: " $4; exit }
      $7 > 0 { print "pairs dropped for capacity: " $7; exit }
      ($17 + $21) >= 1e5 && ($19 + $23) / ($17 + $21) > kmax { print "mean K of the leptons of the pairs " ($19 + $23) / ($17 + $21) " above " kmax; exit }')
    if [[ -n "$reason" ]]; then hits=$((hits + 1)); else hits=0; fi
    if (( hits >= 2 )); then
      { date -u; echo "stop flag: $reason (two readings in a row)"; tail -n 1 "$run/bh_counters.txt"; } > "$run/STOPPED-BY-FLAG.txt"
      kill -TERM "$pid" 2>/dev/null
      return 0
    fi
  done
}

echo "==== run $name (cap $cap s, stop flag at a mean K of the pairs of $FLAG_KPAIR)"
( cd "$run" && exec timeout --kill-after=60 "$cap" "$x" -input "$name.toml" > run.log 2>&1 ) &
pid=$!
flag "$pid" &
fpid=$!
rc=0
wait "$pid" || rc=$?
kill "$fpid" 2>/dev/null
wait "$fpid" 2>/dev/null
echo "RUN $name rc=$rc seconds=$((SECONDS - T0)) cap=$cap"
[[ -f "$run/STOPPED-BY-FLAG.txt" ]] && { echo "==== stopped by the flag"; cat "$run/STOPPED-BY-FLAG.txt" | cut -c1-400; }
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
grep -h 'Bethe-Heitler emission' "$run/run.log" | head -1
# the engine returns 0 after a fatal error of its own: say so here
echo "lines with FATAL or ERROR in run.log: $(grep -c -E 'FATAL|ERROR' "$run/run.log")"
grep -m 3 -h -E 'FATAL|ERROR' "$run/run.log" | cut -c1-300
tail -25 "$run/run.log"
echo "==== what was written"
for k in fields particles spectra; do
  echo "$k: $(ls "$run/$name/$k" 2>/dev/null | wc -l) outputs, $(du -sh "$run/$name/$k" 2>/dev/null | cut -f1)"
done
ls -d "$run"/*.ckpt 2>/dev/null && echo "a checkpoint directory exists (not expected)"
du -sh "$run" | cut -f1
wc -l "$run/bh_counters.txt" "$run/bh_gauss.txt" 2>/dev/null
tail -1 "$run/bh_counters.txt" 2>/dev/null | cut -c1-400
tail -1 "$run/bh_gauss.txt" 2>/dev/null
exit $rc
