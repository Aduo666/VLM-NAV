#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/default.yaml}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/nav_latest.log}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "Logging to ${LOG_PATH}"

python -m vln_nav.app.nav --config "${CONFIG_PATH}" "$@"
