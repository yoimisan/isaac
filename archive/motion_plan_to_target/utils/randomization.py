import numpy as np
import random
import carb

from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

RANDOM_SEED = None
MAX_RANDOMIZATION_ATTEMPTS = 1000
TABLE_MARGIN = 0.03
OBJECT_SEPARATION_MARGIN = 0.04
FRANKA_MIN_REACH = 0.20
FRANKA_MAX_REACH = 0.78

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def _xy_half_extent(aabb):
    return 0.5 * (np.asarray(aabb[3:5]) - np.asarray(aabb[0:2]))


def _xy_bounds_inside_table(table_aabb, object_half_extent):
    min_xy = np.asarray(table_aabb[0:2]) + object_half_extent + TABLE_MARGIN
    max_xy = np.asarray(table_aabb[3:5]) - object_half_extent - TABLE_MARGIN
    if np.any(min_xy > max_xy):
        raise RuntimeError(
            "Randomization bounds are invalid. The table is too small for the object and configured margin."
        )
    return min_xy, max_xy


def _sample_xy(min_xy, max_xy):
    return np.array([
        random.uniform(min_xy[0], max_xy[0]),
        random.uniform(min_xy[1], max_xy[1]),
    ])


def _is_reachable_from_franka(xy, franka_base_xy):
    distance = np.linalg.norm(xy - franka_base_xy)
    return FRANKA_MIN_REACH <= distance <= FRANKA_MAX_REACH


def _xy_aabbs_overlap(center_a, half_a, center_b, half_b, margin):
    delta = np.abs(center_a - center_b)
    return np.all(delta <= half_a + half_b + margin)


def _set_xy_default_and_pose(prim, xy):
    position, orientation = prim.get_world_pose()
    position = np.asarray(position, dtype=np.float32)
    orientation = np.asarray(orientation, dtype=np.float32)
    position[0:2] = xy

    prim.set_default_state(position=position, orientation=orientation)
    prim.set_world_pose(position=position, orientation=orientation)


def randomize_scene_layout(franka, cube, target_region, table):
    bbox_cache = create_bbox_cache(use_extents_hint=False)
    table_aabb = compute_aabb(bbox_cache, table.prim_path, include_children=True)
    cube_aabb = compute_aabb(bbox_cache, cube.prim_path, include_children=True)
    target_aabb = compute_aabb(bbox_cache, target_region.prim_path, include_children=True)

    cube_half_extent = _xy_half_extent(cube_aabb)
    target_half_extent = _xy_half_extent(target_aabb)
    cube_min_xy, cube_max_xy = _xy_bounds_inside_table(table_aabb, cube_half_extent)
    target_min_xy, target_max_xy = _xy_bounds_inside_table(table_aabb, target_half_extent)

    franka_position, _ = franka.get_world_pose()
    franka_base_xy = np.asarray(franka_position[0:2])

    for _ in range(MAX_RANDOMIZATION_ATTEMPTS):
        target_xy = _sample_xy(target_min_xy, target_max_xy)
        if not _is_reachable_from_franka(target_xy, franka_base_xy):
            continue

        cube_xy = _sample_xy(cube_min_xy, cube_max_xy)
        if not _is_reachable_from_franka(cube_xy, franka_base_xy):
            continue

        if _xy_aabbs_overlap(cube_xy, cube_half_extent, target_xy, target_half_extent, OBJECT_SEPARATION_MARGIN):
            continue

        _set_xy_default_and_pose(target_region, target_xy)
        _set_xy_default_and_pose(cube, cube_xy)
        carb.log_info(
            f"Randomized cube xy={cube_xy.tolist()} and target_region xy={target_xy.tolist()}."
        )
        return

    raise RuntimeError(
        "Failed to sample a valid cube/target layout. Try reducing margins or widening the reachable/table region."
    )