#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Edit these constants for the evaluation episode you want to run.
# The selected episode is copied to eval/current_episode.json, then nav.py
# auto-selects it because the file contains exactly one episode.
SPLIT="val_seen"
EPISODE_INDEX="0"
EPISODE_ID=""

cd "${ROOT_DIR}"

INTERIORNAV_SPLIT="${SPLIT}" \
INTERIORNAV_EPISODE_INDEX="${EPISODE_INDEX}" \
INTERIORNAV_EPISODE_ID="${EPISODE_ID}" \
python eval/select_interiornav_episode.py

bash scripts/run_nav.sh --tasks-file eval/current_episode.json
