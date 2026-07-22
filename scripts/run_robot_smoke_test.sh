#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/default.yaml}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  echo "Activate the project conda environment before running this script." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python -m vln_nav.app.robot_smoke_test --config "${CONFIG_PATH}" "$@"
