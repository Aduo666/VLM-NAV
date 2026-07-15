#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/default.yaml}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/task_editor_latest.log}"
ACTIVE_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
ACTIVE_CONDA_PREFIX="${CONDA_PREFIX:-}"

# Edit these constants when authoring tasks in another scene.
TASK_EDITOR_SCENE_USD="${TASK_EDITOR_SCENE_USD:-usd/hospital.usd}"
TASK_EDITOR_ROBOT_USD="${TASK_EDITOR_ROBOT_USD:-src/ros2_car_model_ws/src/1557/urdf/1557_all/1557_all.usd}"
TASK_EDITOR_AUTO_LOAD="${TASK_EDITOR_AUTO_LOAD:-1}"

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
  echo "Activate your environment first, for example:" >&2
  echo "  conda activate <your-project-env>" >&2
  echo "Or run with:" >&2
  echo "  CONDA_ENV_NAME=<your-project-env> bash scripts/run_task_editor.sh" >&2
  exit 1
fi

if [[ "${SHOULD_ACTIVATE}" == "1" ]]; then
  conda activate "${ENV_NAME}"
fi

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Logging to ${LOG_PATH}"
echo "Using conda environment: ${ENV_NAME}"
echo "DISPLAY=${DISPLAY:-<unset>} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ACTIVE_GPU=${ACTIVE_GPU:-<config>} PHYSICS_GPU=${PHYSICS_GPU:-<config>}"
echo "TASK_EDITOR_SCENE_USD=${TASK_EDITOR_SCENE_USD} TASK_EDITOR_ROBOT_USD=${TASK_EDITOR_ROBOT_USD} TASK_EDITOR_AUTO_LOAD=${TASK_EDITOR_AUTO_LOAD}"

python -m vln_nav.app.task_editor_launcher \
  --config "${CONFIG_PATH}" \
  --scene-usd "${TASK_EDITOR_SCENE_USD}" \
  --robot-usd "${TASK_EDITOR_ROBOT_USD}"
