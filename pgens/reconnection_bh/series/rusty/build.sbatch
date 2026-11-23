#!/bin/bash -l
# The build job of the resolution pair. Submitted by submit.sh from the root of the pair
# (~/ceph/bh-resolution-pair-claude1-20261123); the log paths are relative to it.
#SBATCH --job-name=bhpair-build
#SBATCH --account=cca
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHP_ROOT:-$HOME/ceph/bh-resolution-pair-claude1-20261123}/src/pgens/reconnection_bh/series/rusty/build.sh"
