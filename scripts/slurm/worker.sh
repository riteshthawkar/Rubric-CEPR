#!/usr/bin/env bash
set -euo pipefail
source "${RUBRIC_ACTIVATE_SCRIPT:?Activation script must be specified}"
cd "${RUBRIC_ROOT:?Release checkout must be specified}"
export PYTHONPATH="$RUBRIC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${RUBRIC_ENV_PREFIX:?}/bin/python"
exec "$PYTHON_BIN" scripts/rubric_cepr.py train "$@"
