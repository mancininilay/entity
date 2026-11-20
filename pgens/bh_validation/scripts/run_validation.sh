#!/usr/bin/env bash
# Registered validation runs of the Bethe-Heitler emission policy (engine level).
#   run_validation.sh <builds dir> <mpiexec> <output root> <make_input.py> <group>...
# Groups: v2 v4 v5 v6b v7 v9 v12 v13 (v1 has its own script). Builds used:
#   bhem-val-{single,double}/src/entity.xc      OpenMP
#   bhem-val-mpi-{single,double}/src/entity.xc  MPI + OpenMP
set -uo pipefail
BUILDS="$1"; MPIEXEC="$2"; ROOT="$3"; MK="$4"; shift 4
GROUPS_WANTED=" $* "
want() { [[ "$GROUPS_WANTED" == *" $1 "* ]]; }
J3=0.0127189; J6=0.389958; J10=1.09447
COOL="fields=false neutralizer=false gauss=false recycle=true nx=32 ny=32 maxnpart_leptons=400000"
run() { # dir prec key=value...
  local dir="$ROOT/$1" prec="$2"; shift 2
  mkdir -p "$dir"
  python3 "$MK" "$dir/input.toml" "$@" > "$dir/make_input.log"
  ( cd "$dir" && OMP_PROC_BIND=false OMP_NUM_THREADS=8 "$BUILDS/bhem-val-$prec/src/entity.xc" -input input.toml > run.log 2>&1 )
  echo "$1 rc=$?"
}
run_mpi() { # dir prec ranks decomposition key=value...
  local dir="$ROOT/$1" prec="$2" ranks="$3" dec="$4"; shift 4
  mkdir -p "$dir"
  python3 "$MK" "$dir/input.toml" decomposition="$dec" "$@" > "$dir/make_input.log"
  ( cd "$dir" && OMP_PROC_BIND=false OMP_NUM_THREADS=$((8 / ranks)) \
      "$MPIEXEC" -n "$ranks" "$BUILDS/bhem-val-mpi-$prec/src/entity.xc" -input input.toml > run.log 2>&1 )
  echo "$1 rc=$?"
}
for prec in single double; do
  if want v2; then
    # mean cooling and straggling: Gamma_p0 = 6 to <K>/K0 = 0.5, Gamma_p0 = 3 to 0.65
    run v2/g6_$prec $prec name=v2g6 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
    run v2/g3_$prec $prec name=v2g3 order=ions_first $COOL n_ions=100000 gamma_p=3.0 gamma_max=3.2 J_ref=$J3 p_ref=0.05 steps=132000 diag_interval=25
  fi
  if want v9; then
    # the other species order
    run v9/g6_leptons_first_$prec $prec name=v9g6 order=leptons_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
  fi
  if want v4; then
    # pair spectra at fixed Gamma_p: emitters along +x1, about 1.05e6 pairs in 5 steps
    for order in ions_first leptons_first; do
      run v4/g3_${order}_$prec $prec name=v4g3 order=$order fields=false neutralizer=false gauss=false isotropic=false n_ions=4200000 gamma_p=3.0 gamma_max=3.2 J_ref=$J3 p_ref=0.05 steps=5 diag_interval=1 dump_step=4 maxnpart_leptons=1500000 sort_interval=0
      run v4/g10_${order}_$prec $prec name=v4g10 order=$order fields=false neutralizer=false gauss=false isotropic=false n_ions=4200000 gamma_p=10.0 gamma_max=12.0 J_ref=$J10 p_ref=0.05 steps=5 diag_interval=1 dump_step=4 maxnpart_leptons=1500000 sort_interval=0
    done
  fi
  if want v5; then
    # the poster test: Gamma_p0 = 300, eps0 = 0.05, 1e4 emitters, dt = 0.5, to t = 23000
    run v5/g300_$prec $prec name=v5 order=ions_first $COOL n_ions=10000 gamma_p=300.0 gamma_max=0.0 eps0=0.05 dx=1.0 cfl=0.7071067811865476 tau=0.02827113042 steps=46000 diag_interval=23
  fi
  if want v6b; then
    # uniform magnetic field along x3, emitters in the plane: tan(theta) = B dt / (2 gamma)
    for th in 0.05 0.2; do
      bz=$(python3 -c "import math; g=1.0+1836.15*5.0; dt=0.5*0.1/math.sqrt(2.0); print(repr(2.0*g*math.tan($th)/dt))")
      run v6/b_theta${th}_$prec $prec name=v6b order=ions_first $COOL isotropic=false n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=2000 diag_interval=100 bz=$bz
    done
  fi
  if want v7; then
    # time step: the same optical depth, maximum probability 0.0125 to 0.1 over the
    # band up to gamma_max = 6.02. The probability at the top of the band is
    # 1.0014 times the nominal one (0.01252, 0.02503, 0.05007), so the largest
    # step is taken 0.2 % shorter (CFL 0.998, maximum probability 0.0999): with
    # CFL 1.0 the run is refused at startup (0.1001 > max_probability).
    for cfg in "0.0125:0.125:160000:80" "0.025:0.25:80000:40" "0.05:0.5:40000:20" "0.1:0.998:20040:10"; do
      IFS=: read -r pm cfl steps diag <<< "$cfg"
      # p_ref proportional to the step: the optical depth is the same in the four runs
      pref=$(python3 -c "print(repr(0.05 / 1.00754 * $cfl / 0.5))")
      run v7/p${pm}_$prec $prec name=v7 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.02 J_ref=$J6 p_ref=$pref cfl=$cfl steps=$steps diag_interval=$diag
    done
  fi
  if want v13; then
    # pair weight 2 and 4 (the same probability per step: the optical depth scales with a)
    for a in 2 4; do
      run v13/a${a}_$prec $prec name=v13 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 pair_weight=$a.0 use_weights=true steps=40000 diag_interval=25
    done
  fi
  if want v12; then
    # MPI: the Gamma_p0 = 6 cooling run on 1, 2 and 4 ranks
    run_mpi v12/r1_$prec $prec 1 "" name=v12 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
    run_mpi v12/r2_$prec $prec 2 "2, 1" name=v12 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
    run_mpi v12/r4_$prec $prec 4 "2, 2" name=v12 order=ions_first $COOL n_ions=100000 gamma_p=6.0 gamma_max=6.5 J_ref=$J6 p_ref=0.05 steps=40000 diag_interval=25
  fi
done
