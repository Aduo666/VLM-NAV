#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${PROJECT_ROOT}/weights/yoloe"

if ! command -v python >/dev/null 2>&1; then
  printf 'No active project Python was found. Activate the conda environment first.\n' >&2
  exit 1
fi

mkdir -p "${WEIGHTS_DIR}"
cd "${WEIGHTS_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/captainav-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

python -c 'from ultralytics import YOLOE; model = YOLOE("yoloe-26l-seg.pt", verbose=False); embeddings = model.get_text_pe(["chair"]); model.set_classes(["chair"], embeddings); print("YOLOE assets ready")'

test -f "${WEIGHTS_DIR}/yoloe-26l-seg.pt"
test -f "${WEIGHTS_DIR}/mobileclip2_b.ts"
printf 'YOLOE model: %s\n' "${WEIGHTS_DIR}/yoloe-26l-seg.pt"
printf 'Text encoder: %s\n' "${WEIGHTS_DIR}/mobileclip2_b.ts"
