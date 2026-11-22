#!/usr/bin/env bash
# Build job of the shakedown: entity.xc with pgen reconnection_bh (CUDA,
# output on, no MPI, single precision; the Kokkos and ADIOS2 installs of
# ~/work/env_rusty_entity.sh), the tool bp2raw, and a smoke run of 58 steps
# of the toy input on the GPU. Each step under a time cap; nothing is retried.
# The build is marked usable (file BUILD_OK with the hash of the executable)
# only if it compiled, linked with every library resolved, and the smoke run
# ended with status 0 and wrote its counters and its three kinds of output.
#
# Started by build.sbatch from the clone of the commit (see prepare.sh).
# BHR_ROOT and BHR_STUB_RC exist only for the dry run on the login node
# (dryrun.sh): with BHR_STUB_RC set, no compiler is called and a stub
# executable is written instead; everything else runs as in the job.
set -uo pipefail
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || exit 1
: "${SLURM_JOB_ID:?run inside a Slurm allocation}"
R="${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}"
src="$R/src"
want=$(cat "$R/COMMIT") || exit 1
bld="$R/build-$SLURM_JOB_ID"
mkdir "$bld" || exit 1
exec > "$bld/driver.log" 2>&1
trap 'rc=$?; echo "Exit status: $rc"; echo "$rc" > "$bld/exit-status"' EXIT
T0=$SECONDS
hostname; date -u
module list 2>&1
g++ --version | head -1; nvcc --version | tail -2; cmake --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
grep Cpus_allowed_list /proc/self/status
[[ $(git -C "$src" rev-parse HEAD) == "$want" && -z $(git -C "$src" status --porcelain) ]] || { echo "source is not a clean $want"; exit 1; }
echo "source: $src at $want"
echo "Kokkos: $Kokkos_DIR"; echo "ADIOS2: $adios2_DIR"; echo "CXX: $CXX"
jobs="${SLURM_CPUS_PER_TASK:-8}"
x="$bld/tree/src/entity.xc"
rc=0
if [[ -n "${BHR_STUB_RC:-}" ]]; then
  mkdir -p "$bld/tree/src" "$bld/tools"
  echo "stub configure" > "$bld/configure.log"; echo "stub build" > "$bld/build.log"
  cat > "$x" <<STUB
#!/bin/bash
# stub of entity.xc for the dry run: writes what the checks of the drivers look at
name=\$(grep -m1 'name ' "\$2" | cut -d'"' -f2)
mkdir -p "\$name/fields" "\$name/particles" "\$name/spectra"
touch "\$name/fields/fields.00000001.bp" "\$name/particles/particles.00000001.bp" "\$name/spectra/spectra.00000001.bp"
printf '# steps time pairs\n1 0.1 0\n58 5.0 7\n' > bh_counters.txt
printf '# window\n# steps time\n1 0.1 0\n' > bh_gauss.txt
echo "stub run of \$name in \$(pwd)"
exit $BHR_STUB_RC
STUB
  chmod +x "$x"
  printf '#!/bin/bash\necho stub bp2raw\n' > "$bld/tools/bp2raw"; chmod +x "$bld/tools/bp2raw"
  sleep 1
else
  cmake -S "$src" -B "$bld/tree" -G Ninja -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DKokkos_DIR="$Kokkos_DIR" -Dadios2_DIR="$adios2_DIR" -Dmpi=OFF -Doutput=ON -Dprecision=single \
    -Dpgen=reconnection_bh > "$bld/configure.log" 2>&1 || rc=$?
  if [[ $rc == 0 ]]; then
    timeout --kill-after=20 1500 cmake --build "$bld/tree" -j "$jobs" > "$bld/build.log" 2>&1 || rc=$?
  fi
  mkdir -p "$bld/tools"
  g++ -std=c++17 -O2 "$src/pgens/reconnection_bh/tools/bp2raw.cpp" \
    $("$ENTITY_RUSTY_ROOT/install/adios2/bin/adios2-config" -s --cxx-flags --cxx-libs) -o "$bld/tools/bp2raw" > "$bld/build-bp2raw.log" 2>&1 \
    || { echo "bp2raw did not build"; cat "$bld/build-bp2raw.log"; rc=1; }
fi
nerr=$(grep -c -E '(^|[^a-zA-Z_])error([^a-zA-Z_]|$)' "$bld/build.log" 2>/dev/null); nwarn=$(grep -c -i 'warning' "$bld/build.log" 2>/dev/null)
echo "BUILD rc=$rc seconds=$((SECONDS - T0)) error_lines=${nerr:-0} warning_lines=${nwarn:-0}"
if [[ $rc != 0 || ! -x "$x" ]]; then
  tail -40 "$bld/build.log" 2>/dev/null || tail -40 "$bld/configure.log"
  exit 1
fi
grep -i 'warning' "$bld/build.log" | sort | uniq -c | head -20
ls -la "$x"; sha256sum "$x"
if [[ -z "${BHR_STUB_RC:-}" ]]; then
  missing=$(ldd "$x" | grep -c 'not found')
  echo "unresolved libraries: $missing"
  [[ $missing == 0 ]] || exit 1
  grep -E 'Git hash|Enabled Kokkos devices' "$bld/configure.log" | sed 's/\x1b\[[0-9;]*m//g'
fi
echo "==== smoke run (toy input, runtime 5)"
mkdir "$bld/smoke" || exit 1
sed 's/^  runtime = 500.0$/  runtime = 5.0/' "$src/pgens/reconnection_bh/toy-low.toml" > "$bld/smoke/toy-low.toml"
grep -q '^  runtime = 5.0$' "$bld/smoke/toy-low.toml" || { echo "smoke input not made"; exit 1; }
t=$SECONDS; rc=0
( cd "$bld/smoke" && timeout --kill-after=20 300 "$x" -input toy-low.toml > run.log 2>&1 ) || rc=$?
echo "SMOKE rc=$rc seconds=$((SECONDS - t)) cap=300"
tail -5 "$bld/smoke/run.log"
ok=1
[[ $rc == 0 ]] || ok=0
for f in bh_counters.txt bh_gauss.txt; do
  [[ -s "$bld/smoke/$f" ]] || { echo "smoke: no $f"; ok=0; }
done
for k in fields particles spectra; do
  n=$(ls "$bld/smoke/toy-low/$k" 2>/dev/null | wc -l)
  echo "smoke: $n outputs of $k"
  [[ $n -ge 1 ]] || ok=0
done
tail -1 "$bld/smoke/bh_counters.txt" 2>/dev/null | cut -c1-200
tail -1 "$bld/smoke/bh_gauss.txt" 2>/dev/null
grep -h 'Bethe-Heitler emission' "$bld/smoke/run.log" | head -1
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv || true
if [[ $ok == 1 ]]; then
  sha256sum "$x" | cut -d' ' -f1 > "$bld/BUILD_OK"
  echo "$bld" > "$R/BUILD"
  echo "BUILD_OK written; $R/BUILD points to $bld"
else
  echo "build NOT marked usable"
  exit 1
fi
echo "total seconds: $((SECONDS - T0))"
