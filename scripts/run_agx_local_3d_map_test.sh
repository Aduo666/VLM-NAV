#!/usr/bin/env bash
set -euo pipefail
ACTIVE_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ACTIVE_ROOT_DIR}"
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  echo "Activate your project conda environment before running this test." >&2
  exit 1
fi
export PYTHONPATH="${ACTIVE_ROOT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
exec "${CONDA_PREFIX}/bin/python" -m vln_nav.app.agx_local_3d_map_test
