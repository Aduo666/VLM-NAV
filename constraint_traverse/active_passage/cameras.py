CAMERA_PATHS = {
    "front": "pan_tilt_pitch_link/pan_tilt_surface/camera_mount/camera_body/camera_base/PanCamera",
    "arm": "link6/Camera",
    "down": "dummy_base_link/DownCamera",
}

DEFAULT_MAP_SOURCES = ("front", "down")


def normalize_map_sources(sources):
    if isinstance(sources, str):
        raise ValueError("地图相机来源需要使用列表或元组")
    values = tuple(sources)
    if len(set(values)) != len(values) or any(source not in CAMERA_PATHS for source in values):
        raise ValueError("地图相机来源必须为不重复的 front、arm、down")
    return tuple(source for source in CAMERA_PATHS if source in values)
