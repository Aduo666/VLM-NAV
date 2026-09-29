from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class NavigationEpisode:
    episode_id: str
    name: str
    usd_path: Path
    start_x_m: float
    start_y_m: float
    start_yaw_deg: float
    instruction: str


def load_episodes(path: Path, *, project_root: Path) -> tuple[NavigationEpisode, ...]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "episodes"}:
        raise ValueError("episodes.json 必须包含 schema_version 和 episodes")
    if payload["schema_version"] != 1:
        raise ValueError(f"不支持的任务清单版本：{payload['schema_version']!r}")
    rows = payload["episodes"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("episodes.json 至少需要一条任务")

    episodes = tuple(
        _parse_episode(row, project_root=Path(project_root)) for row in rows
    )
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episodes.json 中的任务 id 不能重复")
    return episodes


def select_episode(
    episodes: tuple[NavigationEpisode, ...],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> NavigationEpisode:
    if not episodes:
        raise ValueError("没有可选择的导航任务")
    output_fn("可用导航任务：")
    for index, episode in enumerate(episodes, start=1):
        output_fn(f"  [{index}] {episode.name} ({episode.episode_id})")
        output_fn(f"      场景：{episode.usd_path}")
        output_fn(f"      指令：{episode.instruction}")

    while True:
        try:
            raw = input_fn(f"请选择任务编号 [1-{len(episodes)}]：").strip()
        except EOFError as exc:
            raise RuntimeError("终端没有提供任务编号") from exc
        try:
            index = int(raw)
        except ValueError:
            output_fn("请输入有效的整数编号。")
            continue
        if 1 <= index <= len(episodes):
            return episodes[index - 1]
        output_fn(f"任务编号必须位于 1 至 {len(episodes)}。")


def _parse_episode(row: object, *, project_root: Path) -> NavigationEpisode:
    expected = {"id", "name", "usd_path", "start_pose", "instruction"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"每条任务必须包含这些字段：{sorted(expected)}")
    pose = row["start_pose"]
    if not isinstance(pose, dict) or set(pose) != {"x_m", "y_m", "yaw_deg"}:
        raise ValueError("任务 start_pose 必须包含 x_m、y_m 和 yaw_deg")

    episode_id = str(row["id"]).strip()
    name = str(row["name"]).strip()
    instruction = str(row["instruction"]).strip()
    if not episode_id or not name or not instruction:
        raise ValueError("任务 id、name 和 instruction 不能为空")
    try:
        pose_values = tuple(
            float(pose[key]) for key in ("x_m", "y_m", "yaw_deg")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"任务 {episode_id} 的初始位姿必须是数值") from exc
    if not all(math.isfinite(value) for value in pose_values):
        raise ValueError(f"任务 {episode_id} 的初始位姿必须是有限数值")

    usd_path = Path(str(row["usd_path"]).strip()).expanduser()
    if not usd_path.is_absolute():
        usd_path = project_root / usd_path
    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"任务 {episode_id} 的场景文件不存在：{usd_path}")
    return NavigationEpisode(
        episode_id=episode_id,
        name=name,
        usd_path=usd_path,
        start_x_m=pose_values[0],
        start_y_m=pose_values[1],
        start_yaw_deg=pose_values[2],
        instruction=instruction,
    )
