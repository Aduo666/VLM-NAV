#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Edit these constants for the evaluation episode you want to run.
# The selected episode is copied to eval/current_episode.json, then nav.py
# auto-selects it because the file contains exactly one episode.
cd "${ROOT_DIR}"

VLNPE_SPLIT="${SPLIT:-val_seen}" \
VLNPE_EPISODE_INDEX="${EPISODE_INDEX:-0}" \
VLNPE_EPISODE_ID="${EPISODE_ID:-}" \
VLNPE_DATA_ROOT="/Data/qiang/InternData-N1/vln_pe" \
python eval/vlnpe/select_episode.py

TARGET_DEBUG_IMAGE_PATH="debug/eval/vln_debug.png" \
DEBUG_VIDEO_PATH="debug/eval/videos/vln_debug.mp4" \
DEPTH_DEBUG_IMAGE_PATH="debug/eval/depth_debug.png" \
bash scripts/run_nav.sh --tasks-file eval/current_episode.json
