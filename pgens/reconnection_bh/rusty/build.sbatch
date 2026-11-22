#!/bin/bash -l
# Submit from the root of the shakedown (~/ceph/bh-reconnection-claude1-20261121):
#   sbatch src/pgens/reconnection_bh/rusty/build.sbatch
# The log paths are relative to that directory.
#SBATCH --job-name=bhrec-build
#SBATCH --account=cca
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}/src/pgens/reconnection_bh/rusty/build.sh"
