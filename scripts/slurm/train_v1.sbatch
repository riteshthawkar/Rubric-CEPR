#!/usr/bin/env bash
# Submit from an inspected tmux session after CPU preflight has passed.
#SBATCH --job-name=accv_v1_reproduce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
: "${TMUX:?Submit from an inspected tmux session}"
: "${ACCV_RELEASE_ROOT:?Set the absolute release checkout path}"
: "${ACCV_ENV:?Set the conda environment prefix}"
: "${ACCV_ACTIVATE_SCRIPT:?Set the absolute activation script path}"
: "${HF_HOME:?Set the Hugging Face cache location}"
srun --gres=gpu:1 --export=ALL bash "$ACCV_RELEASE_ROOT/scripts/slurm/worker.sh" "$@"
