#!/usr/bin/env bash
# The Slurm worker sources this file inside its numeric srun step.
set -euo pipefail
CONDA_COMMAND="${CONDA_EXE:-conda}"
eval "$("$CONDA_COMMAND" shell.bash hook)"
conda activate "${RUBRIC_ENV_PREFIX:?Set your conda environment prefix}"
