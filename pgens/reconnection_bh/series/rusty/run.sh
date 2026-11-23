#!/usr/bin/env bash
# One run of the resolution pair (Claude 1, 2026-11-23): a registered input of `series/`, a fresh
# run directory, a time cap below the wall limit of the job. Nothing is retried; the exit status
# of the engine is kept. Started by run.sbatch (submit.sh).
#   run.sh <name> <cap in seconds>
#
# Executables:
#   r-low-1card            the executable of the decomposition runs (second local session, build
#                          at b996538f, `~/ceph/bh-mpi-gpu-claude1-20261122/BUILD`), checked
#                          against its BUILD_OK and its known hash; one rank on one card.
#   every other input      the executable of this root (`BUILD`), built from a commit whose code
#                          is that of the run commit: only files under `series/` may differ.
#                          Four ranks, one card each.
#
# Stop flag (the driver, not the generator): every 60 s the last line of bh_counters.txt and the
# lines of the last step of bh_ranks.txt are read; the run is stopped (TERM) when on two
# readings in a row they show a value that is not a number, particle-steps above the tables,
# pairs dropped for capacity, a mean K of all leptons of the pairs above FLAG_KPAIR with at least
# 1e5 of them present, or a species above FLAG_FILL of its capacity on any rank. The reason is
# written to STOPPED-BY-FLAG.txt.
# BHP_ROOT, BHP_FLAG_PERIOD, BHP_SECOND and BHP_SECOND_SHA exist only for the dry run.
set -uo pipefail
name="${1:?name of the run}"; cap="${2:?cap in seconds}"
[[ "$cap" =~ ^[0-9]+$ ]] || { echo "run.sh: cap in seconds"; exit 2; }
case "$name" in
  s1-n-a|s1-n-b|s1-n-c|s1-n-long|s1-f-a|s1-f-b|s1-f-c) ranks=4; exe=own; leptons="3 4" ;;
  lf-ctl|lf-neg) ranks=4; exe=own; leptons="1 2" ;;
  r-low-1card) ranks=1; exe=second; leptons="3 4" ;;
  *) echo "run.sh: unknown run $name"; exit 2 ;;
esac
FLAG_KPAIR=175
FLAG_FILL=0.9
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHP_ROOT:-$HOME/ceph/bh-resolution-pair-claude1-20261123}"
src="$R/src"
S="$src/pgens/reconnection_bh/series"
want=$(cat "$R/COMMIT") || exit 1
run="$R/runs/$name-$SLURM_JOB_ID"
mkdir -p "$R/runs" || exit 1
mkdir "$run" || exit 1
exec > "$run/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$run/exit-status"' EXIT
T0=$SECONDS
hostname; date -u
module list 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
grep Cpus_allowed_list /proc/self/status
[[ $(git -C "$src" rev-parse HEAD) == "$want" && -z $(git -C "$src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
if [[ $exe == own ]]; then
  bld=$(cat "$R/BUILD") || { echo "no usable build"; exit 1; }
  x="$bld/rec-single/src/entity.xc"
  [[ -f "$bld/rec-single/BUILD_OK" && -x "$x" && $(sha256sum "$x" | cut -d' ' -f1) == $(cat "$bld/rec-single/BUILD_OK") ]] || { echo "the executable is not the one marked usable"; exit 1; }
  built=$(cat "$bld/COMMIT") || exit 1
  other=$(git -C "$src" diff --name-only "$built" "$want" | grep -v '^pgens/reconnection_bh/series/' || true)
  [[ -z "$other" ]] || { echo "files outside series/ differ between the commit of the build and that of the run:"; echo "$other"; exit 1; }
  echo "source: $src at $want; the executable was built from $built"
else
  B2="${BHP_SECOND:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}"
  bld=$(cat "$B2/BUILD") || { echo "no build of the decomposition runs"; exit 1; }
  x="$bld/rec-single/src/entity.xc"
  known="${BHP_SECOND_SHA:-2f6d102603f70f8857fbabe89fa86dfad20414868c4a3f30515d98e120859aa5}"
  [[ -x "$x" && $(sha256sum "$x" | cut -d' ' -f1) == $(cat "$bld/rec-single/BUILD_OK") && $(cat "$bld/rec-single/BUILD_OK") == "$known" ]] || { echo "the executable of the decomposition runs is not the known one"; exit 1; }
  echo "source of the input: $src at $want; executable of the decomposition runs ($(cat "$B2/COMMIT-rec"))"
fi
echo "build: $bld"; echo "executable sha256: $(sha256sum "$x" | cut -d' ' -f1)"
(cd "$S" && grep " $name.toml\$" INPUTS.sha256 | sha256sum -c -) || { echo "the input is not the registered one"; exit 1; }
IFS=, read -ra devs <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ ${#devs[@]} == "$ranks" || -n "${BHP_SHARE_GPU:-}" ]] || { echo "${#devs[@]} GPUs visible, $ranks needed"; exit 1; }
cp "$S/$name.toml" "$run/$name.toml" || exit 1
sha256sum "$run/$name.toml"
df -h "$R" | tail -1

flag() {
  local pid="$1" hits=0 reason
  set -- $leptons
  local e="$1" p="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${BHP_FLAG_PERIOD:-60}"
    [[ -s "$run/bh_counters.txt" ]] || continue
    reason=$(tail -n 1 "$run/bh_counters.txt" | awk -v kmax="$FLAG_KPAIR" -v e="$e" -v p="$p" '
      NF < 16 || (NF - 8) % 4 != 0 { print "a line of " NF " columns"; exit }
      { for (i = 1; i <= NF; i++) if ($i !~ /^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$/) { print "column " i " is not a number: " $i; exit } }
      $4 > 0 { print "particle-steps above the tables: " $4; exit }
      $7 > 0 { print "pairs dropped for capacity: " $7; exit }
      { ne = 9 + 4 * (e - 1); np = 9 + 4 * (p - 1) }
      ($ne + $np) >= 1e5 && ($(ne + 2) + $(np + 2)) / ($ne + $np) > kmax { print "mean K of the leptons of the pairs " ($(ne + 2) + $(np + 2)) / ($ne + $np) " above " kmax; exit }')
    if [[ -z "$reason" && -s "$run/bh_ranks.txt" ]]; then
      reason=$(awk -v fill="$FLAG_FILL" '
        /^#/ { next }
        { if ($1 != last) { delete row; n = 0; last = $1 } row[++n] = $0 }
        END { for (k = 1; k <= n; k++) { split(row[k], f, " "); m = length(f)
                for (j = 4; j + 1 <= m - 2; j += 2) if (f[j + 1] > 0 && f[j] / f[j + 1] > fill) {
                  printf "rank %s, species %d: %s of %s slots\n", f[3], (j - 2) / 2, f[j], f[j + 1]; exit } } }' "$run/bh_ranks.txt")
    fi
    if [[ -n "$reason" ]]; then hits=$((hits + 1)); else hits=0; fi
    if (( hits >= 2 )); then
      { date -u; echo "stop flag: $reason (two readings in a row)"; tail -n 1 "$run/bh_counters.txt"; } > "$run/STOPPED-BY-FLAG.txt"
      kill -TERM "$pid" 2>/dev/null
      return 0
    fi
  done
}

echo "==== run $name on $ranks rank(s) (cap $cap s; stop flag: mean K of the pairs $FLAG_KPAIR, fill $FLAG_FILL)"
( cd "$run" && BHM_GPU_LOG="$run/gpu" exec timeout --kill-after=60 "$cap" mpirun --bind-to none --map-by "ppr:$ranks:node" -n "$ranks" \
    "$HERE/rank-gpu.sh" "$x" -input "$name.toml" > run.log 2>&1 ) &
pid=$!
flag "$pid" &
fpid=$!
rc=0
wait "$pid" || rc=$?
kill "$fpid" 2>/dev/null
wait "$fpid" 2>/dev/null
echo "RUN $name rc=$rc seconds=$((SECONDS - T0)) cap=$cap"
[[ -f "$run/STOPPED-BY-FLAG.txt" ]] && { echo "==== stopped by the flag"; cut -c1-400 "$run/STOPPED-BY-FLAG.txt"; }
cat "$run"/gpu.* 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
# the engine can return 0 after a fatal error of its own; the logs carry control codes
sed 's/^\(\x1b\[[0-9;]*[A-Za-z]\)*//' "$run/run.log" | grep -h 'Bethe-Heitler emission' | head -2
echo "lines with FATAL or ERROR in run.log: $(grep -c -E 'FATAL|ERROR' "$run/run.log")"
grep -m 3 -h -E 'FATAL|ERROR' "$run/run.log" | cut -c1-300
tail -25 "$run/run.log"
echo "==== what was written"
for k in fields particles spectra; do
  echo "$k: $(ls "$run/$name/$k" 2>/dev/null | wc -l) outputs, $(du -sh "$run/$name/$k" 2>/dev/null | cut -f1)"
done
ls -d "$run"/*.ckpt 2>/dev/null && echo "a checkpoint directory exists (not expected)"
du -sh "$run" | cut -f1
wc -l "$run"/bh_*.txt 2>/dev/null
tail -1 "$run/bh_counters.txt" 2>/dev/null | cut -c1-400
tail -1 "$run/bh_gauss.txt" 2>/dev/null
tail -"$ranks" "$run/bh_ranks.txt" 2>/dev/null
exit $rc
