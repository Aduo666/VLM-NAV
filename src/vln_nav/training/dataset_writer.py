from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from vln_nav.navigation.occupancy_grid import OccupancyGridSnapshot

from .geometry import crop_egocentric_occupancy
from .schemas import CanonicalEpisode, ExpertReplayFrame, TrainingSampleV3
from .timeline import write_timeline_html


class ExpertReplayDatasetWriter:
    def __init__(
        self,
        *,
        output_root: str | Path,
        episode: CanonicalEpisode,
        overwrite: bool,
    ) -> None:
        root = Path(output_root).expanduser().resolve()
        self.episode_dir = root / episode.split / f"{episode.scene_id}_{episode.episode_id}"
        manifest_path = self.episode_dir / "manifest.json"
        has_existing_output = self.episode_dir.exists() and any(self.episode_dir.iterdir())
        if has_existing_output and manifest_path.is_file() and not overwrite:
            raise FileExistsError(
                f"Replay output already exists: {self.episode_dir}. Use --overwrite to replace files."
            )
        if self.episode_dir.exists() and (overwrite or not manifest_path.is_file()):
            shutil.rmtree(self.episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.episode_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_frames: list[dict] = []
        self._timeline_paths: list[str] = []
        self._timeline_metadata: list[dict] = []
        self._episode_artifacts: dict[str, str] = {}
        self.write_json(self.episode_dir / "episode.json", episode.to_dict())

    def write_frame_assets(
        self,
        *,
        frame: ExpertReplayFrame,
        rgb: np.ndarray,
        depth: np.ndarray,
        snapshot: OccupancyGridSnapshot,
        robot_pose_xyyaw: tuple[float, float, float],
        timeline_image: Image.Image,
        local_size_m: float,
        local_cells: int,
        global_cells: int,
        occupied_threshold: float,
        free_threshold: float,
    ) -> dict[str, str]:
        frame_dir = self.frames_dir / f"{frame.pose.frame_index:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = frame_dir / "rgb.png"
        depth_path = frame_dir / "depth.npz"
        local_map_path = frame_dir / "local_map.npz"
        global_map_path = frame_dir / "global_map.npz"
        timeline_path = frame_dir / "timeline.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(rgb_path)
        np.savez_compressed(depth_path, depth_m=np.asarray(depth, dtype=np.float32))
        local_map = crop_egocentric_occupancy(
            log_odds=snapshot.log_odds,
            origin_x=float(snapshot.origin_x),
            origin_y=float(snapshot.origin_y),
            source_resolution_m=float(snapshot.resolution_m),
            robot_x=float(robot_pose_xyyaw[0]),
            robot_y=float(robot_pose_xyyaw[1]),
            robot_yaw_rad=float(robot_pose_xyyaw[2]),
            output_size_m=float(local_size_m),
            output_cells=int(local_cells),
            occupied_threshold=float(occupied_threshold),
            free_threshold=float(free_threshold),
        )
        np.savez_compressed(
            local_map_path,
            occupancy=local_map,
            size_m=np.float32(local_size_m),
            robot_pose_xyyaw=np.asarray(robot_pose_xyyaw, dtype=np.float32),
        )
        coarse_global = _coarse_global_channels(snapshot=snapshot, output_cells=int(global_cells))
        np.savez_compressed(
            global_map_path,
            occupancy=coarse_global,
            origin_xy=np.asarray([snapshot.origin_x, snapshot.origin_y], dtype=np.float32),
            resolution_m=np.float32(snapshot.size_m / float(coarse_global.shape[-1])),
        )
        timeline_image.save(timeline_path)
        self.write_json(frame_dir / "frame.json", frame.to_dict())
        return {
            "rgb": str(rgb_path.relative_to(self.episode_dir)),
            "depth": str(depth_path.relative_to(self.episode_dir)),
            "local_map": str(local_map_path.relative_to(self.episode_dir)),
            "global_map": str(global_map_path.relative_to(self.episode_dir)),
            "timeline": str(timeline_path.relative_to(self.episode_dir)),
        }

    def write_sample(self, *, sample: TrainingSampleV3, paths: dict[str, str]) -> None:
        frame_index = int(sample.progress.observation.frame_index)
        sample_path = self.frames_dir / f"{frame_index:05d}" / "sample.json"
        self.write_json(sample_path, sample.to_dict())
        self._manifest_frames.append(
            {
                "frame_index": frame_index,
                "sample_path": str(sample_path.relative_to(self.episode_dir)),
                "paths": paths,
                "mode": sample.spatial_subgoal.supervision.navigation_mode,
                "selected_frontier_id": (
                    sample.spatial_subgoal.supervision.selected_frontier_id
                ),
                "progress_supervision_valid": (
                    sample.progress.supervision.supervision_valid
                ),
                "spatial_supervision_valid": (
                    sample.spatial_subgoal.supervision.supervision_valid
                ),
                "quality_flags": list(sample.quality_flags),
            }
        )
        self._timeline_paths.append(paths["timeline"])
        self._timeline_metadata.append(self._manifest_frames[-1])

    def write_final_observed_map(
        self,
        *,
        snapshot: OccupancyGridSnapshot,
        output_cells: int,
    ) -> None:
        """Save the map accumulated over the full replay for auditing only."""

        occupancy = _coarse_global_channels(snapshot=snapshot, output_cells=int(output_cells))
        data_path = self.episode_dir / "final_observed_global_map.npz"
        image_path = self.episode_dir / "final_observed_global_map.png"
        np.savez_compressed(
            data_path,
            occupancy=occupancy,
            origin_xy=np.asarray([snapshot.origin_x, snapshot.origin_y], dtype=np.float32),
            resolution_m=np.float32(snapshot.size_m / float(occupancy.shape[-1])),
        )
        visualization = np.full(occupancy.shape[1:], 92, dtype=np.uint8)
        visualization[occupancy[1].astype(bool)] = 240
        visualization[occupancy[0].astype(bool)] = 16
        Image.fromarray(visualization, mode="L").save(image_path)
        self._episode_artifacts = {
            "final_observed_global_map": str(data_path.relative_to(self.episode_dir)),
            "final_observed_global_map_image": str(image_path.relative_to(self.episode_dir)),
        }

    def finalize(self) -> Path:
        manifest_path = self.episode_dir / "manifest.json"
        self.write_json(
            manifest_path,
            {
                "episode_dir": str(self.episode_dir),
                "frame_count": len(self._manifest_frames),
                "episode_artifacts": self._episode_artifacts,
                "frames": self._manifest_frames,
            },
        )
        write_timeline_html(
            output_path=self.episode_dir / "timeline.html",
            episode_title=self.episode_dir.name,
            frame_paths=self._timeline_paths,
            metadata=self._timeline_metadata,
        )
        return manifest_path

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _coarse_global_channels(
    *,
    snapshot: OccupancyGridSnapshot,
    output_cells: int,
) -> np.ndarray:
    cells = max(16, int(output_cells))
    occupied = _max_pool_to_shape(np.asarray(snapshot.occupied_mask, dtype=bool), cells)
    free = _max_pool_to_shape(np.asarray(snapshot.free_mask, dtype=bool), cells)
    free &= ~occupied
    unknown = ~(occupied | free)
    return np.stack([occupied, free, unknown], axis=0).astype(np.uint8)


def _max_pool_to_shape(mask: np.ndarray, output_cells: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    factor = max(1, int(np.ceil(max(source.shape) / float(output_cells))))
    padded_rows = int(np.ceil(source.shape[0] / factor) * factor)
    padded_cols = int(np.ceil(source.shape[1] / factor) * factor)
    padded = np.zeros((padded_rows, padded_cols), dtype=bool)
    padded[: source.shape[0], : source.shape[1]] = source
    pooled = padded.reshape(
        padded_rows // factor,
        factor,
        padded_cols // factor,
        factor,
    ).max(axis=(1, 3))
    image = Image.fromarray((pooled.astype(np.uint8) * 255), mode="L")
    image = image.resize((output_cells, output_cells), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0
