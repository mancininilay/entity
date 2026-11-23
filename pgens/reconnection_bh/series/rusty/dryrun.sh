#!/usr/bin/env bash
# Dry run of the drivers of the resolution pair on a Rusty login node, end to end, through the
# sbatch files run as plain scripts (Claude 1, 2026-11-23). No Slurm, no GPU, no compiler.
#   dryrun.sh <a copy of the folder series/ of the branch>
# A temporary root with a one-commit source holding that folder; `sbatch` and `sacct` replaced by
# stubs on the PATH; stub executables that write the diagnostic files as the engine does (on rank
# 0 only). Cases:
#   build             build.sh with stubs: both trees, the capacity cases, the smoke run, BUILD set
#   healthy (s1-n-a)  runs to the end, no flag
#   tables (s1-n-b)   particle-steps above the tables: the flag must stop it
#   garbled (s1-n-c)  one unreadable line among healthy ones: the flag must not stop it
#   fill (s1-f-a)     a species above 90 percent of its capacity on one rank: the flag must stop it
#   hot (s1-n-long)   hot pairs: the flag must stop it
#   lf (lf-neg)       leptons first, five species: no flag
#   second (r-low-1card)  one rank, the executable of the decomposition runs (a stub, its hash given)
#   badinput          an input changed after registration: the driver must refuse it
#   testoption        a science input carrying a test-only option: the driver must refuse it
#   submit            submit.sh: the budget rule (one case within, one beyond) and the one-hour rule
set -uo pipefail
F="$(cd "${1:?folder series/}" && pwd)"
mkdir -p "$HOME/ceph/bh-resolution-pair-claude1-20261123/dryruns"; T=$(mktemp -d "$HOME/ceph/bh-resolution-pair-claude1-20261123/dryruns/dry.XXXXXX")
echo "dry-run root: $T"
mkdir -p "$T/src/pgens/reconnection_bh" "$T/stubs" "$T/bin" "$T/second/build/rec-single/src" "$T/slurm"
cp -r "$F" "$T/src/pgens/reconnection_bh/series"
( cd "$T/src" && git init -q . && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m stub )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
S="$T/src/pgens/reconnection_bh/series"
export BHP_ROOT="$T" BHP_FLAG_PERIOD=1 OMPI_MCA_rmaps_base_oversubscribe=1 BHP_SHARE_GPU=1

c24="100 8.63 50 0 0 0 0 90.0 37579570 37579570 193802117.4 264.6 37579575 37579575 193846309.3 264.6 150000 150000 450000.5 300.1 150000 150000 460000.25 310.2"
c28="100 8.63 50 0 0 0 0 90.0 150000 150000 450000.5 300.1 150000 150000 460000.25 310.2 37579570 37579570 193802117.4 264.6 37579575 37579575 193846309.3 264.6 0 0 0 0"
r4="100 8.63 RANK 9000000 58000000 9000000 58000000 40000 40000000 40000 40000000 50 0"
cat > "$T/stubs/entity.xc" <<'STUB'
#!/bin/bash
echo "stub engine: $*"
[[ "${OMPI_COMM_WORLD_RANK:-0}" == 0 ]] || exit 0
inp="$2"; name=$(grep -m1 'name ' "$inp" | cut -d'"' -f2)
mkdir -p "$name/fields" "$name/particles" "$name/spectra"
case="${BHP_DRY_CASE:-healthy}"
echo "# steps time pairs ..." > bh_counters.txt
echo "# steps time rank ..." > bh_ranks.txt
echo "# window" > bh_gauss.txt; echo "# steps time nodes max_residual" >> bh_gauss.txt; echo "1 0.08 100 1e-9 1e-11 1e-3 1e-4 1e-3 1e-4" >> bh_gauss.txt
echo "# rows" > bh_profile.txt
for i in $(seq 1 "${BHP_DRY_LINES:-6}"); do
  line="$BHP_DRY_C"
  case "$case" in
    tables) line=$(echo "$line" | awk '{$4 = 3; print}') ;;
    hot)    line=$(echo "$line" | awk '{$19 = 3.0e7; $23 = 3.0e7; print}') ;;
    garbled) [[ $i == 2 ]] && line=$(echo "$line" | awk '{$24 = "3.1e"; print}') ;;
  esac
  echo "$line" >> bh_counters.txt
  for k in 0 1 2 3; do
    r=$(echo "$BHP_DRY_R" | sed "s/RANK/$k/")
    [[ "$case" == fill && $k == 2 ]] && r=$(echo "$r" | awk '{$4 = 53000000; print}')
    echo "$r" >> bh_ranks.txt
  done
  sleep 1
done
echo "Bethe-Heitler emission (stub)"
exit 0
STUB
cat > "$T/stubs/test-framework-comm-capacity.xc" <<'STUB'
#!/bin/bash
n="${OMPI_COMM_WORLD_SIZE:-1}"
if [[ "$1" == over ]]; then echo "Too many particles to receive (cannot fit into maxptl)"; exit 15; fi
[[ "${OMPI_COMM_WORLD_RANK:-0}" == 0 ]] && echo "OK $1 on $n rank(s)"
exit 0
STUB
chmod +x "$T/stubs/"*
# Slurm stubs: sbatch records its arguments; sacct reports every listed job as ended after 0.5 h
# on 4 GPUs, except a job whose id starts with 55, which is pending with a limit of 300 minutes
cat > "$T/bin/sbatch" <<'STUB'
#!/bin/bash
echo "STUB sbatch $*" >> "$BHP_ROOT/sbatch.calls"
[[ "$1" == --test-only ]] && { echo "sbatch: Job 1 to start at 2026-11-23T14:00:00 using 32 processors on nodes workergpu000 in partition gpu"; exit 0; }
echo "99$RANDOM"
STUB
cat > "$T/bin/sacct" <<'STUB'
#!/bin/bash
for id in $(echo "$4" | tr , ' '); do
  if [[ $id == 55* ]]; then echo "PENDING 0 300 billing=32,cpu=32,gres/gpu=4,mem=256G,node=1"
  else echo "COMPLETED 1800 60 billing=32,cpu=32,gres/gpu=4,mem=256G,node=1"; fi
done
STUB
chmod +x "$T/bin/"*

show() { grep -v -E '^ *[0-9]+\)|^$|Where:|Sticky|Currently Loaded|NVIDIA-SMI has failed|nvidia-smi' "$1" | cut -c1-200; }
echo "================ build"
( export SLURM_JOB_ID=1000 BHP_STUB="$T/stubs" CUDA_VISIBLE_DEVICES=0; bash "$S/rusty/build.sbatch" )
b="$T/build-1000"; echo "build exit $(cat "$b/exit-status")"; grep -E '^u19|^RUN|BUILD |smoke|marked|status' "$b/driver.log" | head -30
echo "BUILD -> $(cat "$T/BUILD" 2>/dev/null)"

( cd "$S" && sha256sum s1-*.toml lf-*.toml r-low-1card.toml > INPUTS.sha256 )
( cd "$T/src" && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m inputs )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
# the second executable: a stub with its own hash
cp "$T/stubs/entity.xc" "$T/second/build/rec-single/src/entity.xc"
sha=$(sha256sum "$T/second/build/rec-single/src/entity.xc" | cut -d' ' -f1)
echo "$sha" > "$T/second/build/rec-single/BUILD_OK"; echo "$T/second/build" > "$T/second/BUILD"; echo b996538f > "$T/second/COMMIT-rec"
export BHP_SECOND="$T/second" BHP_SECOND_SHA="$sha"
k=2000
for c in healthy:s1-n-a:24 tables:s1-n-b:24 garbled:s1-n-c:24 fill:s1-f-a:24 hot:s1-n-long:24 lf:lf-neg:28 second:r-low-1card:24; do
  case="${c%%:*}"; rest="${c#*:}"; name="${rest%%:*}"; nc="${rest#*:}"
  k=$((k + 1))
  echo "================ $case ($name)"
  [[ $nc == 28 ]] && C="$c28" || C="$c24"
  ( export SLURM_JOB_ID=$k BHP_DRY_CASE="$case" BHP_DRY_C="$C" BHP_DRY_R="$r4" BHP_DRY_LINES=8 CUDA_VISIBLE_DEVICES=0,1,2,3
    [[ $name == r-low-1card ]] && export CUDA_VISIBLE_DEVICES=0
    bash "$S/rusty/run.sbatch" "$name" 120 )
  d="$T/runs/$name-$k"
  echo "exit status $(cat "$d/exit-status" 2>/dev/null); flag: $(cat "$d/STOPPED-BY-FLAG.txt" 2>/dev/null | sed -n 2p)"
  grep -E '^RUN|not |refuse|differ|GPUs visible|executable' "$d/driver.log" | head -5
done
echo "================ badinput"
echo "# changed" >> "$S/s1-n-b.toml"
( cd "$T/src" && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m changed )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
( export SLURM_JOB_ID=2999 BHP_DRY_C="$c24" BHP_DRY_R="$r4" CUDA_VISIBLE_DEVICES=0,1,2,3; bash "$S/rusty/run.sbatch" s1-n-b 60 )
echo "exit status $(cat "$T/runs/s1-n-b-2999/exit-status" 2>/dev/null)"; grep -E 'registered|not clean|FAILED' "$T/runs/s1-n-b-2999/driver.log"
echo "================ testoption (a science input carrying stale_prev = true, registered)"
sed -i 's/^  injector       = 2$/  injector       = 2\n  stale_prev     = true/' "$S/s1-n-c.toml"
( cd "$S" && sha256sum s1-*.toml lf-*.toml r-low-1card.toml > INPUTS.sha256 )
( cd "$T/src" && git add . && git -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m testoption )
git -C "$T/src" rev-parse HEAD > "$T/COMMIT"
( export SLURM_JOB_ID=2998 BHP_DRY_C="$c24" BHP_DRY_R="$r4" CUDA_VISIBLE_DEVICES=0,1,2,3; bash "$S/rusty/run.sbatch" s1-n-c 60 )
echo "exit status $(cat "$T/runs/s1-n-c-2998/exit-status" 2>/dev/null)"; grep -E 'launch validation' "$T/runs/s1-n-c-2998/driver.log"
grep -h 'launch validation' "$T"/runs/lf-neg-*/driver.log "$T"/runs/s1-n-a-*/driver.log
echo "================ submit (stubs of sbatch and sacct)"
sed -e "s|R=\"\$HOME/ceph/bh-resolution-pair-claude1-20261123\"|R=\"$T\"|" "$S/rusty/submit.sh" > "$T/submit-dry.sh"
( export PATH="$T/bin:$PATH"; cd "$T"
  : > jobs.txt
  bash "$T/submit-dry.sh" s1-n-a 01:00:00 gpupreempt --test-only
  bash "$T/submit-dry.sh" s1-n-a 01:00:00 gpupreempt
  bash "$T/submit-dry.sh" s1-n-long 02:30:00 gpupreempt
  echo "x 551" >> jobs.txt; echo "x 552" >> jobs.txt
  bash "$T/submit-dry.sh" s1-f-a 05:00:00 gpu
  echo "submit exit $? (expected 0: one ended job charged 2, two pending reserve 40, the new one 20)"
  echo "x 553" >> jobs.txt
  bash "$T/submit-dry.sh" s1-f-b 05:00:00 gpu
  echo "submit exit $? (expected 3: 2 + 60 + 20 above 75)" )
cat "$T/sbatch.calls"
echo "DRY RUN DONE ($T)"
