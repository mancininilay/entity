#!/bin/bash -l
# Submit from the stage-C directory (~/ceph/bh-reconnection-claude1-20261121/stage-c):
#   sbatch --job-name=bhrec-<name> --time=<limit> jobs/run_c.sbatch <name> <cap in seconds>
# The log paths are relative to that directory.
#SBATCH --account=cca
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHR_ROOT:-$HOME/ceph/bh-reconnection-claude1-20261121}/stage-c/jobs/run_c.sh" "$1" "$2"
