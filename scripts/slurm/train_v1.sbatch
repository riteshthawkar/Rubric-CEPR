#!/usr/bin/env bash
# Submit from an inspected tmux session after CPU preflight has passed.
#SBATCH --job-name=rubric_cepr_reproduce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
: "${TMUX:?Submit from an inspected tmux session}"
: "${RUBRIC_ROOT:?Set the absolute release checkout path}"
: "${RUBRIC_ENV_PREFIX:?Set the conda environment prefix}"
: "${RUBRIC_ACTIVATE_SCRIPT:?Set the absolute activation script path}"
: "${HF_HOME:?Set the Hugging Face cache location}"
srun --gres=gpu:1 --export=ALL bash "$RUBRIC_ROOT/scripts/slurm/worker.sh" "$@"
