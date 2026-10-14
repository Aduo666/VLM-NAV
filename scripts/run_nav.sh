#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  echo "Activate the project conda environment before running navigation." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
exec "${CONDA_PREFIX}/bin/python" -m vln_nav.app.navigation "$@"
