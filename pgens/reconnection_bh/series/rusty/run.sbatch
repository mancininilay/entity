#!/bin/bash -l
# One run of the resolution pair. Submitted by submit.sh from the root of the pair, which sets
# the job name, the partition, the wall limit and, for the run on one card, the number of GPUs:
#   sbatch --job-name=bhpair-<name> --time=<limit> --partition=<p> [--qos=gpupreempt] \
#          [--ntasks=1 --gpus-per-node=1 --mem=96G] run.sbatch <name> <cap in seconds>
#SBATCH --account=cca
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
exec bash "${BHP_ROOT:-$HOME/ceph/bh-resolution-pair-claude1-20261123}/src/pgens/reconnection_bh/series/rusty/run.sh" "$1" "$2"
