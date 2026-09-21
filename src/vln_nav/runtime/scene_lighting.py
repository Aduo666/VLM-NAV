from __future__ import annotations

import logging
import math


_FIXTURE_NAMES = ("ceiling_light", "spot_light", "panel_light")
_CEILING_NAMES = ("ceiling", "roof")


def add_indoor_lighting(
    *,
    stage,
    environment_prim_path: str,
    logger: logging.Logger,
) -> int:
    from pxr import Gf, Usd, UsdGeom, UsdLux

    environment = stage.GetPrimAtPath(environment_prim_path)
    if not environment.IsValid():
        logger.warning("[InteriorLighting] missing environment=%s", environment_prim_path)
        return 0

    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    scene_bounds = _world_bounds(bounds_cache, environment)
    if scene_bounds is None:
        logger.warning("[InteriorLighting] missing scene bounds")
        return 0

    scene_minimum, scene_maximum = scene_bounds
    fixture_minimum_z = scene_minimum[2] + 0.55 * (scene_maximum[2] - scene_minimum[2])
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    fixtures = []
    ceilings = []
    for prim in Usd.PrimRange(environment):
        name = prim.GetName().lower()
        if any(part in name for part in _FIXTURE_NAMES):
            location = transform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            if float(location[2]) >= fixture_minimum_z:
                fixtures.append(
                    (float(location[0]), float(location[1]), float(location[2]) - 0.08)
                )
        if not any(part in name for part in _CEILING_NAMES):
            continue
        if any(
            any(part in child.GetName().lower() for part in _CEILING_NAMES)
            for child in prim.GetChildren()
        ):
            continue
        bounds = _world_bounds(bounds_cache, prim)
        if bounds is None:
            continue
        minimum, maximum = bounds
        ceilings.append(
            (
                (minimum[0] + maximum[0]) / 2.0,
                (minimum[1] + maximum[1]) / 2.0,
                minimum[2] - 0.12,
            )
        )

    positions = _deduplicate_positions([*fixtures, *ceilings], spacing_m=0.75)
    source = "scene_ceiling"
    if not positions:
        positions = _fallback_grid(scene_bounds)
        source = "scene_bounds"

    lights_path = "/World/InteriorLights"
    if stage.GetPrimAtPath(lights_path).IsValid():
        stage.RemovePrim(lights_path)
    UsdGeom.Xform.Define(stage, lights_path)
    for index, position in enumerate(positions):
        light = UsdLux.RectLight.Define(stage, f"{lights_path}/Ceiling_{index:02d}")
        light.CreateIntensityAttr().Set(1500.0)
        light.CreateExposureAttr().Set(2.0)
        light.CreateWidthAttr().Set(1.5)
        light.CreateHeightAttr().Set(1.5)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr().Set(False)

    logger.info("[InteriorLighting] added %d lights source=%s", len(positions), source)
    return len(positions)


def _world_bounds(bounds_cache, prim):
    bounds = bounds_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    if not all(math.isfinite(float(value)) for value in (*minimum, *maximum)):
        return None
    if any(maximum[index] <= minimum[index] for index in range(2)):
        return None
    if maximum[2] < minimum[2]:
        return None
    return (
        tuple(float(value) for value in minimum),
        tuple(float(value) for value in maximum),
    )


def _deduplicate_positions(positions, *, spacing_m: float):
    selected = []
    for position in sorted(positions, key=lambda item: item[2]):
        if any(
            math.hypot(position[0] - previous[0], position[1] - previous[1]) < spacing_m
            for previous in selected
        ):
            continue
        selected.append(position)
    return selected


def _fallback_grid(bounds, *, spacing_m: float = 4.0):
    minimum, maximum = bounds
    width = maximum[0] - minimum[0]
    depth = maximum[1] - minimum[1]
    columns = max(1, min(6, math.ceil(width / spacing_m)))
    rows = max(1, min(6, math.ceil(depth / spacing_m)))
    height = maximum[2] - minimum[2]
    light_z = min(maximum[2] - 0.2, minimum[2] + min(2.6, 0.8 * height))
    return [
        (
            minimum[0] + width * (column + 0.5) / columns,
            minimum[1] + depth * (row + 0.5) / rows,
            light_z,
        )
        for row in range(rows)
        for column in range(columns)
    ]
