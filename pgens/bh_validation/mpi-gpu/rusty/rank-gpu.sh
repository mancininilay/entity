#!/bin/bash
# One MPI rank on its own GPU: the index of the rank on its node picks one of
# the GPUs that Slurm gave the node (as P1.3's launcher does). With fewer GPUs
# than ranks (the smoke runs of the build job) the ranks share them.
#   mpirun ... rank-gpu.sh <executable> <arguments>
IFS=, read -ra devs <<< "${CUDA_VISIBLE_DEVICES:?no GPU visible}"
k="${OMPI_COMM_WORLD_LOCAL_RANK:?not started by Open MPI}"
export CUDA_VISIBLE_DEVICES="${devs[$((k % ${#devs[@]}))]}"
if [[ -n "${BHM_GPU_LOG:-}" ]]; then
  echo "rank ${OMPI_COMM_WORLD_RANK} local $k of ${#devs[@]} visible: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv,noheader 2>&1)" > "$BHM_GPU_LOG.${OMPI_COMM_WORLD_RANK}"
fi
exec "$@"
