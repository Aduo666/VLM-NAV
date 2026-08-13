#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  echo "Activate the project conda environment before collecting data." >&2
  exit 1
fi

SPLIT_NAME="${SPLIT:-train}"
START="${START_INDEX:-${EPISODE_INDEX:-0}}"
COUNT_VALUE="${COUNT:-1}"
HEADLESS_VALUE="${HEADLESS:-1}"
OVERWRITE_VALUE="${OVERWRITE:-0}"
CONFIG_VALUE="${CONFIG_PATH:-configs/default.yaml}"
TRAINING_CONFIG_VALUE="${TRAINING_CONFIG_PATH:-training/configs/expert_replay.yaml}"
LOG_PATH="${EXPERT_REPLAY_LOG_PATH:-/Data/qiang/data/CAPTAINav/expert_replay/logs/expert_replay_latest.log}"

if [[ "${SPLIT_NAME}" != "train" ]]; then
  echo "Expert replay collection is restricted to SPLIT=train; got ${SPLIT_NAME}." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

for ((offset=0; offset<COUNT_VALUE; offset++)); do
  index=$((START + offset))
  command=(
    python -m vln_nav.app.expert_replay_collect
    --config "${CONFIG_VALUE}"
    --training-config "${TRAINING_CONFIG_VALUE}"
    --split train
    --episode-index "${index}"
  )
  if [[ "${HEADLESS_VALUE}" == "0" ]]; then
    command+=(--no-headless)
  fi
  if [[ "${OVERWRITE_VALUE}" == "1" ]]; then
    command+=(--overwrite)
  fi
  echo "[ExpertReplay] collecting train episode_index=${index}"
  "${command[@]}"
done
