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
"${PYTHON_BIN}" -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'

cat <<EOF

GroundingDINO code is installed in:
  ${GROUNDING_DINO_DIR}

Put or symlink these files here:
  ${WEIGHT_DIR}/GroundingDINO_SwinT_OGC.py
  ${WEIGHT_DIR}/groundingdino_swint_ogc.pth

SAM weights:
  ${WEIGHT_DIR}/sam_vit_h_4b8939.pth
  ${WEIGHT_DIR}/repvit_sam.pt

If you already have CA-Nav weights, this is enough:
  ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/GroundingDINO_SwinT_OGC.py ${WEIGHT_DIR}/GroundingDINO_SwinT_OGC.py
  ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/groundingdino_swint_ogc.pth ${WEIGHT_DIR}/groundingdino_swint_ogc.pth
  ln -sf /home/Lab/qiangsibin/CA-Nav-code/data/grounded_sam/sam_vit_h_4b8939.pth ${WEIGHT_DIR}/sam_vit_h_4b8939.pth

Then run:
  bash scripts/run_nav.sh
EOF
