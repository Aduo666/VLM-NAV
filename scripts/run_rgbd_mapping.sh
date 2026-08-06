#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/default.yaml}"
MAPPING_CONFIG_PATH="${MAPPING_CONFIG_PATH:-${ROOT_DIR}/configs/rgbd_mapping.yaml}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/rgbd_mapping.log}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  echo "Activate the project conda environment before running this script." >&2
  echo "Example: conda activate vlmnav" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Standalone RGB-D mapping log: ${LOG_PATH}"
echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
python -m vln_nav.app.rgbd_mapping_test \
  --config "${CONFIG_PATH}" \
  --mapping-config "${MAPPING_CONFIG_PATH}" \
  "$@"
