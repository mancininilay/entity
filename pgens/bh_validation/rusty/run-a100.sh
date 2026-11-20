#!/usr/bin/env bash
# Job 2 of the Rusty proposal (report of 2026-11-19, section 8): one A100, one rank.
#   run-a100.sh <job id of the CUDA build>
# In this order, each run under a time cap, every run attempted:
#   tier A (A1, A2, A3)                                          single, double
#   V1 on one rank: both species orders, the controls, the negative control,
#                   and the same with stale slots                single, double
#   V4 at Gamma_p = 3 and 10, ions first                         single, double
#   V2 and V3 at Gamma_p0 = 6, 1e5 emitters, 40000 steps         single, double
# then the analysis scripts of the validation branch on the CPU cores of the
# allocation. Inputs, commands and criteria are those of the runs on the Mac
# (scripts/run_v1.sh, scripts/run_validation.sh, scripts/analyze_final.sh).
set -uo pipefail
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
build_job="${1:?job id of the CUDA build}"
B="$HOME/ceph/entity-builds/bh-emission-claude1-20261119"
O="$HOME/ceph/bh-emission-claude1-20261119"
cuda="$B/cuda-build-$build_job"
V="$B/src/pgens/bh_validation"
S="$V/scripts"
DATA="$V/data"
TOOL="$B/tools/bhv_spectra"
run="$O/run-$SLURM_JOB_ID"
mkdir -p "$O" || exit 1
mkdir "$run" || exit 1
exec > >(tee "$run/driver.log") 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$run/exit-status"' EXIT
T0=$SECONDS
# seconds for the runs; the analysis and a reserve take the rest of the hour
BUDGET=3000
hostname
date -u
module list 2>&1
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "builds: $cuda"
ls "$cuda"/*/BUILD_OK 2>&1
[[ $(git -C "$B/src" rev-parse HEAD) == a8a6f39946657819e6f021343203c0422644df1d && -z $(git -C "$B/src" status --porcelain) ]] || { echo "src is not a clean a8a6f399"; exit 1; }
( cd "$DATA" && sha256sum -c SHA256SUMS ) || { echo "reference data do not match"; exit 1; }
status=0
launch() { # directory cap command...
  local dir="$1" cap="$2"; shift 2
  local rem=$(( BUDGET - (SECONDS - T0) ))
  if (( rem < 60 )); then
    echo "SKIPPED ${dir#$run/}: no time left"
    status=1
    return 0
  fi
  (( cap > rem )) && cap=$rem
  mkdir -p "$dir"
  local t=$SECONDS rc=0
  ( cd "$dir" && timeout --kill-after=20 "$cap" "$@" > run.log 2>&1 ) || rc=$?
  echo "RUN ${dir#$run/} rc=$rc seconds=$((SECONDS - t)) cap=$cap"
  [[ $rc == 0 ]] || status=1
}
entity() { # directory precision cap key=value...
  local dir="$run/$1" prec="$2" cap="$3"; shift 3
  if [[ ! -f "$cuda/val-$prec/BUILD_OK" ]]; then
    echo "NOT BUILT val-$prec: $1 not run"
    status=1
    return 0
  fi
  mkdir -p "$dir"
  if ! python3 "$V/make_input.py" "$dir/input.toml" "$@" > "$dir/make_input.log" 2>&1; then
    echo "INPUT FAILED $1"
    status=1
    return 0
  fi
  launch "$dir" "$cap" "$cuda/val-$prec/src/entity.xc" -input input.toml
}

echo "==== tier A"
for prec in single double; do
  if [[ -f "$cuda/pr-$prec/BUILD_OK" ]]; then
    launch "$run/tierA/$prec" 300 "$cuda/pr-$prec/tests/archetypes/test-archetypes-bethe_heitler.xc"
    cat "$run/tierA/$prec/run.log"
  else
    echo "NOT BUILT pr-$prec: tier A not run"
    status=1
  fi
done

echo "==== V1"
for prec in single double; do
  entity v1/$prec/r1/A_on        $prec 180 name=A_on        order=leptons_first policy=on
  entity v1/$prec/r1/A_off       $prec 180 name=A_off       order=leptons_first policy=off
  entity v1/$prec/r1/B_on        $prec 180 name=B_on        order=ions_first    policy=on
  entity v1/$prec/r1/B_off       $prec 180 name=B_off       order=ions_first    policy=off
  entity v1/$prec/r1/A_neg       $prec 180 name=A_neg       order=leptons_first policy=negative
  entity v1/$prec/r1/A_on_stale  $prec 180 name=A_on_stale  order=leptons_first policy=on       stale_prev=true
  entity v1/$prec/r1/B_on_stale  $prec 180 name=B_on_stale  order=ions_first    policy=on       stale_prev=true
  entity v1/$prec/r1/A_neg_stale $prec 180 name=A_neg_stale order=leptons_first policy=negative stale_prev=true
done

J3=0.0127189; J6=0.389958; J10=1.09447
echo "==== V4"
for prec in single double; do
  entity v4/g3_ions_first_$prec  $prec 240 name=v4g3  order=ions_first fields=false neutralizer=false gauss=false isotropic=false n_ions=4200000 gamma_p=3.0  gamma_max=3.2  J_ref=$J3  p_ref=0.05 steps=5 diag_interval=1 dump_step=4 maxnpart_leptons=1500000 sort_interval=0
  entity v4/g10_ions_first_$prec $prec 240 name=v4g10 order=ions_first fields=false neutralizer=false gauss=false isotropic=false n_ions=4200000 gamma_p=10.0 gamma_max=12.0 J_ref=$J10 p_ref=0.05 steps=5 diag_interval=1 dump_step=4 maxnpart_leptons=1500000 sort_interval=0
done

echo "==== V2, V3"
COOL="fields=false neutralizer=false gauss=false recycle=true nx=32 ny=32 maxnpart_leptons=400000"
for prec in single double; do
  entity v2/g6_$prec $prec 900 name=v2g6 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
done
echo "runs finished after $((SECONDS - T0)) s"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

echo "==== analysis"
D2="$DATA/bhref-0.2.0/moments_mono05.txt"
sed 's/for ranks in (1, 2, 4):/for ranks in (1,):/' "$S/analyze_v1.py" > "$run/analyze_v1_one_rank.py"
{
  echo "---- V1 Gauss's law, one rank (A100)"
  timeout 120 python3 "$run/analyze_v1_one_rank.py" "$run/v1"
  for prec in single double; do
    lim=1e-5; [ $prec = double ] && lim=1e-10
    echo "######## $prec"
    echo "---- V2 mean cooling, V3 straggling"
    timeout 300 python3 "$S/analyze_cooling.py" "$run/v2/g6_$prec" --targets 0.9,0.7,0.5 --d2 "$D2" --tag "V2/V3 Gamma_p0=6 $prec A100"
    echo "---- energy ledger of the same run (analysis only)"
    timeout 120 python3 "$S/analyze_energy.py" "$run/v2/g6_$prec" --limit $lim --tag "fields off Gamma_p0=6 $prec A100"
    echo "---- V4 pair spectra"
    timeout 300 "$TOOL" "$DATA" 0.5 \
      3  "$run/v4/g3_ions_first_$prec/bhv_species_2_rank_0.bin"  "$run/v4/g3_ions_first_$prec/bhv_species_3_rank_0.bin" \
      10 "$run/v4/g10_ions_first_$prec/bhv_species_2_rank_0.bin" "$run/v4/g10_ions_first_$prec/bhv_species_3_rank_0.bin" \
      > "$run/v4/spectra_ions_first_$prec.txt" 2>&1
    echo "[ions_first] $(grep -E '^(PASS|FAIL) V4|^\(history\)' "$run/v4/spectra_ions_first_$prec.txt" | tr '\n' ' ')"
    grep -E "^  Gamma_p|^(PASS|FAIL) Gamma_p|^Gamma_p" "$run/v4/spectra_ions_first_$prec.txt" | sed 's/^/    /'
  done
} > "$run/analysis.txt" 2>&1
cat "$run/analysis.txt"
( cd "$run" && find . -name bhv_tables.txt -exec sha256sum {} + | sort -k2 > tables.sha256 )
( cd "$run" && find . -type f -printf '%P %s\n' | sort > files.txt )
date -u
echo "job finished after $((SECONDS - T0)) s, status $status"
exit "$status"
