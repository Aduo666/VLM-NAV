#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GROUNDING_DINO_DIR="${ROOT_DIR}/third_party/GroundingDINO"
WEIGHT_DIR="${ROOT_DIR}/weights/grounded_sam"

mkdir -p "${ROOT_DIR}/third_party" "${WEIGHT_DIR}"

if [[ ! -d "${GROUNDING_DINO_DIR}/.git" ]]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git "${GROUNDING_DINO_DIR}"
  cd "${GROUNDING_DINO_DIR}"
  git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
else
  cd "${GROUNDING_DINO_DIR}"
fi

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

path = Path("groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu")
text = path.read_text()
replacements = {
    "value.type().is_cuda()": "value.is_cuda()",
    "spatial_shapes.type().is_cuda()": "spatial_shapes.is_cuda()",
    "level_start_index.type().is_cuda()": "level_start_index.is_cuda()",
    "sampling_loc.type().is_cuda()": "sampling_loc.is_cuda()",
    "attn_weight.type().is_cuda()": "attn_weight.is_cuda()",
    "grad_output.type().is_cuda()": "grad_output.is_cuda()",
    "AT_DISPATCH_FLOATING_TYPES(value.type(),": "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),",
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text)
PY

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-grounding.txt"
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -e .
install -m 0644 \
  "${GROUNDING_DINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  "${WEIGHT_DIR}/GroundingDINO_SwinT_OGC.py"
"${PYTHON_BIN}" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir="${WEIGHT_DIR}/bert-base-uncased",
    local_dir_use_symlinks=False,
)
PY
"${PYTHON_BIN}" - <<'PY'
import numpy as np

if np.__version__ != "1.26.0":
    raise SystemExit(
        "Isaac Sim 5.1 requires numpy==1.26.0, "
        f"but the active environment has numpy=={np.__version__}."
    )
PY

cat <<EOF

GroundingDINO code is installed in:
  ${GROUNDING_DINO_DIR}

The model config was copied to:
  ${WEIGHT_DIR}/GroundingDINO_SwinT_OGC.py

Place the checkpoint here:
  ${WEIGHT_DIR}/groundingdino_swint_ogc.pth

The text encoder is stored under:
  ${WEIGHT_DIR}/bert-base-uncased/config.json

Then run:
  bash scripts/run_nav.sh
EOF
