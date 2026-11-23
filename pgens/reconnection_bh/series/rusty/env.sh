# Copied unchanged from pgens/bh_validation/mpi-gpu/rusty/env.sh of dev/bh-mpi-gpu (second local session, 2026-11-22),
# for the resolution pair (Claude 1, 2026-11-23).
# Environment of the builds and runs with CUDA and MPI on Rusty: the modules and
# installs of ~/work/env_rusty_entity.sh (GCC 13.3, CUDA 12.8, Open MPI with CUDA,
# Kokkos 5.2.1, ADIOS2 2.12.1) with UCX, as the P1.3 runs on A100s. Sourced.
export ENTITY_BACKEND=cuda
source "$HOME/work/env_rusty_entity.sh" || return 1
module load ucx/1.18.0 || return 1
unset LD_BIND_NOW
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
