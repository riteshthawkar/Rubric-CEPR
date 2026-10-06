#!/usr/bin/env bash
set -euo pipefail
source "${ACCV_ACTIVATE_SCRIPT:?Activation script must be specified}"
cd "${ACCV_RELEASE_ROOT:?Release checkout must be specified}"
export PYTHONPATH="$ACCV_RELEASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${ACCV_ENV:?}/bin/python"
exec "$PYTHON_BIN" scripts/accv_v1.py train "$@"
