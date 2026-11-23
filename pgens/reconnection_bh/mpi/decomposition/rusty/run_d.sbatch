#!/bin/bash -l
# One run of the decomposition question. Submitted by submit_d.sh from the root of the
# validation, which sets the job name and the wall limit of the run:
#   sbatch --job-name=<name> --time=<limit> .../run_d.sbatch <name of the input> <cap in seconds>
#SBATCH --account=cca
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHM_ROOT:-$HOME/ceph/bh-mpi-gpu-claude1-20261122}/src-rec-d/pgens/reconnection_bh/mpi/decomposition/rusty/run_d.sh" "$1" "$2"
