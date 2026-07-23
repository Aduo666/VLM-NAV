#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
ACTIVE_CONDA_PREFIX="${CONDA_PREFIX:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

SHOULD_ACTIVATE=1
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  ENV_NAME="${CONDA_ENV_NAME}"
elif [[ -n "${ACTIVE_CONDA_DEFAULT_ENV}" && "${ACTIVE_CONDA_DEFAULT_ENV}" != "base" ]]; then
  ENV_NAME="${ACTIVE_CONDA_DEFAULT_ENV}"
elif [[ -n "${ACTIVE_CONDA_PREFIX}" && "${ACTIVE_CONDA_PREFIX}" == */envs/* && -x "${ACTIVE_CONDA_PREFIX}/bin/python" ]]; then
  ENV_NAME="$(basename "${ACTIVE_CONDA_PREFIX}")"
  SHOULD_ACTIVATE=0
  export CONDA_PREFIX="${ACTIVE_CONDA_PREFIX}"
  export CONDA_DEFAULT_ENV="${ENV_NAME}"
  export PATH="${ACTIVE_CONDA_PREFIX}/bin:${PATH}"
else
  echo "No project conda environment is active." >&2
  echo "Activate it first, for example: conda activate vlmnav" >&2
  exit 1
fi

if [[ "${SHOULD_ACTIVATE}" == "1" ]]; then
  conda activate "${ENV_NAME}"
fi

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "Using conda environment: ${ENV_NAME}"
python -m vln_nav.app.g1_policy_test
