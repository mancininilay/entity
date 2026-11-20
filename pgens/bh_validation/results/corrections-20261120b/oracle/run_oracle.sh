#!/usr/bin/env bash
# The three oracle dumps of the first delivery against the headers of the pull-request worktree:
#   default: the dump programs unchanged (every builder with its default thread count);
#   caps 1 and 3: the capped copies with -DBH_THREADS=1 and 3.
# Every output is compared byte for byte with the output at 66aa6fb6 (../../oracle/*_branch.txt).
#   run_oracle.sh <master folder>/Entity Local Experiments <this directory>
set -uo pipefail
E="$1"; C="$2"
W="$E/worktrees/dev-bh-emission"; O="$E/p2-bh-emission-claude1-20261119/oracle"; K="$HOME/.entity/kokkos/5.0.1/openmp"
cd "$C" || exit 1
git -C "$W" rev-parse HEAD > source_head.txt
H=(-DBH_HEADER_PHYSICS='"archetypes/emission/bethe_heitler/bh_physics.h"' -DBH_HEADER_BORN='"archetypes/emission/bethe_heitler/bh_born_qk.h"' -DBH_HEADER_SAMPLER_HOST='"archetypes/emission/bethe_heitler/bh_sampler_host.h"' -DBH_HEADER_EVENT='"archetypes/emission/bethe_heitler/bh_event_host.h"' -DBH_HEADER_INTERPOLATE='"archetypes/emission/bethe_heitler/bh_interpolate.h"')
KO=(-I"$W/src" -I"$W/src/global" -isystem "$K/include" -Wno-c++11-narrowing -DNDEBUG -O3 -std=c++20 -arch arm64 -DSINGLE_PRECISION -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include)
KL=(-L"$K/lib" -lkokkoscore -lkokkoscontainers -L/opt/homebrew/opt/libomp/lib -lomp)
status=0
one() { # label source extra-flag reference kind
  local label="$1" src="$2" flag="$3" ref="$4" kind="$5" rc=0
  if [[ $kind == host ]]; then
    clang++ -std=c++20 -O2 -I "$W/src" "${H[@]}" $flag "$src" -o "bin_$label" 2> "build_$label.log" || rc=$?
  else
    /usr/bin/c++ "${KO[@]}" $flag "$src" -o "bin_$label" "${KL[@]}" 2> "build_$label.log" || rc=$?
  fi
  if [[ $rc != 0 ]]; then echo "BUILD FAILED $label"; tail -5 "build_$label.log"; status=1; return; fi
  local t=$SECONDS
  OMP_NUM_THREADS=4 OMP_PROC_BIND=false "./bin_$label" > "$label.txt" 2> "$label.stderr" || { echo "RUN FAILED $label"; status=1; return; }
  if cmp -s "$label.txt" "$ref"; then echo "IDENTICAL $label ($((SECONDS - t)) s) $(shasum -a 256 "$label.txt" | cut -c1-64)"; else echo "DIFFERENT $label"; status=1; fi
}
one tables_default       dump_tables.cpp              ""              "$O/oracle_branch.txt"       host
one tables_cap1          dump_tables_capped.cpp       -DBH_THREADS=1  "$O/oracle_branch.txt"       host
one tables_cap3          dump_tables_capped.cpp       -DBH_THREADS=3  "$O/oracle_branch.txt"       host
one draws_default        dump_draws.cpp               ""              "$O/draws_branch.txt"        kokkos
one draws_cap1           dump_draws_capped.cpp        -DBH_THREADS=1  "$O/draws_branch.txt"        kokkos
one draws_cap3           dump_draws_capped.cpp        -DBH_THREADS=3  "$O/draws_branch.txt"        kokkos
one device_rates_default dump_device_rates.cpp        ""              "$O/device_rates_branch.txt" kokkos
one device_rates_cap1    dump_device_rates_capped.cpp -DBH_THREADS=1  "$O/device_rates_branch.txt" kokkos
one device_rates_cap3    dump_device_rates_capped.cpp -DBH_THREADS=3  "$O/device_rates_branch.txt" kokkos
shasum -a 256 tables_*.txt draws_*.txt device_rates_*.txt > SHA256SUMS-oracle.txt
shasum -a 256 "$O/oracle_branch.txt" "$O/draws_branch.txt" "$O/device_rates_branch.txt" | sed "s|$E|<Entity Local Experiments>|" >> SHA256SUMS-oracle.txt
echo "oracle status $status"
exit $status
