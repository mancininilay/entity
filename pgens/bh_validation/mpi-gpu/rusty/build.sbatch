#!/bin/bash -l
# Submit from the root of the validation (~/ceph/bh-mpi-gpu-claude1-20261122):
#   sbatch src-val/pgens/bh_validation/mpi-gpu/rusty/build.sbatch
#SBATCH --job-name=bhmpi-build
#SBATCH --account=cca
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHM_ROOT:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}/src-val/pgens/bh_validation/mpi-gpu/rusty/build.sh"
