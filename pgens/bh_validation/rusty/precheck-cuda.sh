#!/usr/bin/env bash
# Login-node pre-check (a build, no Slurm job, no GPU): compiles with the CUDA
# toolchain the two translation units that carry the new code, in single
# precision, so that a compiler-specific error is found before the allocation.
#   precheck-cuda.sh            (niced, 2 compile processes)
set -uo pipefail
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh"
B="$HOME/ceph/entity-builds/bh-emission-claude1-20261119"
P="$B/precheck-cuda"
mkdir -p "$P"
exec > "$P/precheck.log" 2>&1
hostname; date -u; g++ --version | head -1; nvcc --version | tail -2
git -C "$B/src-pr" rev-parse HEAD; git -C "$B/src" rev-parse HEAD
one() { # name src object-pattern cmake args...
  local name="$1" src="$2" pat="$3"; shift 3
  local t0=$SECONDS rc=0
  cmake -S "$src" -B "$P/$name" -G Ninja -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DKokkos_DIR="$Kokkos_DIR" -Dmpi=OFF -Doutput=OFF -Dprecision=single "$@" > "$P/configure-$name.log" 2>&1 || rc=$?
  echo "CONFIGURE $name rc=$rc"
  [[ $rc == 0 ]] || return
  local obj
  obj=$(ninja -C "$P/$name" -t targets all | grep -E "$pat" | head -1 | cut -d: -f1)
  echo "OBJECT $name: $obj"
  nice -n 19 ninja -C "$P/$name" -j 2 "$obj" > "$P/build-$name.log" 2>&1 || rc=$?
  echo "PRECHECK $name rc=$rc seconds=$((SECONDS - t0)) errors=$(grep -c 'error' "$P/build-$name.log") warnings=$(grep -ci 'warning' "$P/build-$name.log")"
}
one pr-single  "$B/src-pr" 'test-archetypes-bethe_heitler.xc.dir/bethe_heitler.cpp.o' -DTESTS=ON &
one val-single "$B/src"    'entity.xc.dir/entity.cpp.o' -Dpgen=bh_validation &
wait
date -u
echo PRECHECK DONE
