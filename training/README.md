# CAPTAINav Expert Replay Dataset

This directory contains the first, data-only stage of the learned VLN stack. It does not run the current CAPTAINav navigation policy and it does not train a model yet.

## Why `train` is used

InteriorNav's `train` split provides demonstrations for fitting learned components. Each episode contains a natural-language instruction, a scene, a start pose, a reference path and a goal. The reference path is privileged supervision. It is never copied into `StudentSample`.

`val_seen` and `val_unseen` remain evaluation splits. The collector rejects any split other than `train` to prevent accidental validation leakage.

## Collection pipeline

```text
InteriorNav train episode
        |
        v
canonical path poses (0.15 m translation, <=15 deg camera turns)
        |
        v
paused Isaac Sim replay with the AGX front RGB-D camera
        |
        +--> causal RGB-D occupancy map
        |        --> true free/unknown boundaries
        |        --> safe reachable navigation goals
        |
        +--> scheduled GroundingDINO --> low-confidence VLM crop verification
        |
        v
StudentSample + OracleTargets + DebugOracle
```

The replay clock is synthetic because InteriorNav does not provide expert controls or timestamps. Camera yaw is derived from the reference-path tangent. This is recorded explicitly rather than presented as an original dataset action sequence.

## Data boundary

- `StudentSample` contains only runtime-available information: raw instruction words, RGB history, causal local/global maps, past trajectory, observed semantic tracks and current frontier candidates. It never contains the current-frame Oracle selection.
- Each frontier records a true free/unknown boundary and a separate safe, reachable navigation goal moved into known free space.
- `OracleTargets` contains progress, event, mode and frontier labels derived from a reliable future-route crossing. Ambiguous frames remain unlabeled.
- `DebugOracle` contains the future reference route, crossing point and alignment evidence for auditing only.

## Collect one episode

```bash
conda activate vlmnav
SPLIT=train EPISODE_INDEX=0 bash training/scripts/collect_interiornav.sh
```

The collector intentionally supports AGX only. Set `robot.profile: "agx"` in `configs/default.yaml` before collection.

To inspect the replay in Isaac Sim:

```bash
HEADLESS=0 EPISODE_INDEX=0 bash training/scripts/collect_interiornav.sh
```

To collect a small validation batch of five training episodes:

```bash
START_INDEX=0 COUNT=5 bash training/scripts/collect_interiornav.sh
```

Existing episode directories are protected by default. Set `OVERWRITE=1` only when intentionally regenerating them.

## Inspect the result

Each episode is stored under:

```text
/Data/qiang/data/CAPTAINav/expert_replay/train/<scene>_<episode>/
├── episode.json
├── manifest.json
├── route_alignment.json
├── final_observed_global_map.npz
├── final_observed_global_map.png
├── timeline.html
└── frames/
    └── 00000/
        ├── rgb.png
        ├── depth.npz
        ├── local_map.npz
        ├── global_map.npz
        ├── frame.json
        ├── sample.json
        └── timeline.png
```

Open `timeline.html` directly in VS Code or a browser. Its three panels separate current RGB, the causal student map and the privileged future-route audit view. A cyan ring is the true free/unknown boundary, a cyan dot is the safe navigation goal, and a yellow goal appears only for a reliable Oracle route crossing. `final_observed_global_map.png` is the map accumulated over the complete replay and is for audit only; earlier student samples keep their causal partial maps.

Validate the path conversion and instruction alignment without starting Isaac Sim:

```bash
PYTHONPATH=src python training/scripts/validate_expert_replay.py --split train --episode-index 0
```

After collection, validate every referenced file and check that privileged fields did not leak into `StudentSample`:

```bash
PYTHONPATH=src python training/scripts/validate_expert_replay.py \
  --episode-dir /Data/qiang/data/CAPTAINav/expert_replay/train/kujiale_0067_955
```

## Configuration

Collection-specific settings live in `training/configs/expert_replay.yaml`. Generated RGB, depth,
map and sample files default to `/Data/qiang/data/CAPTAINav/expert_replay`, outside the project and
Home filesystems. Robot, camera, GroundingDINO and VLM credentials continue to come from the normal
project config so that replay uses the same calibrated AGX sensor.

Collector output is mirrored to
`/Data/qiang/data/CAPTAINav/expert_replay/logs/expert_replay_latest.log` and overwritten on each run.
