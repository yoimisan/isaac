from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import random

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from curobo.types.math import Pose


RANDOM_SEED = None
MAX_RANDOMIZATION_ATTEMPTS = 1000
TABLE_MARGIN = 0.03
OBJECT_SEPARATION_MARGIN = 0.04
FRANKA_MIN_REACH = 0.20
FRANKA_MAX_REACH = 0.78


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

usd_path = "/home/yoisan/Documents/isaac-scenes/toy.usd"

success = open_stage(usd_path)
if not success:
    raise RuntimeError(f"Failed to open stage: {usd_path}")

simulation_app.update()
while is_stage_loading():
    simulation_app.update()

from isaacsim.core.prims import SingleGeometryPrim, SingleXFormPrim
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

world = World(stage_units_in_meters=1.0)

franka = world.scene.add(Robot(
    prim_path="/World/Franka",
    name="franka"
))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/Cube",
    name="cube"
))

target_region = world.scene.add(SingleGeometryPrim(
    prim_path="/World/TargetRegion",
    name="target_region"
))

table = world.scene.add(SingleGeometryPrim(
    prim_path="/World/Table",
    name="table"
))

randomize_scene_layout(franka, cube, target_region, table)
# Above is a very elegant way of building the world. I like it.

# Here we get the pregrasp frame with respect to franka tcp frame relative to cube local frame.
def get_local_pose():
    i = np.random.randint(3)
    
    d = cube.get_size()
    offset_length = 1.25 * (np.sqrt(2) * d / 2)
    theta = np.random.random() * np.pi / 2 + np.pi/4

    if i == 0:
        local_frame_translation = [
            offset_length * np.cos(theta),
            0,
            offset_length * np.sin(theta)
        ]

        local_frame_x = [np.cos(theta - np.pi/2), 0, np.sin(theta - np.pi/2)]
        local_frame_y = [0, -1, 0]
        local_frame_z = [np.cos(theta - np.pi), 0, np.sin(theta - np.pi)]
    else:
        local_frame_translation = [
            0,
            offset_length * np.cos(theta),
            offset_length * np.sin(theta),
        ]

        if i == 1:
            local_frame_x = [0, np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)]
            local_frame_y = [1, 0, 0]
            local_frame_z = [0, np.cos(theta - np.pi), np.sin(theta - np.pi)]
        else:
            local_frame_x = [0, -np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)]
            local_frame_y = [-1, 0, 0]
            local_frame_z = [0, -np.cos(theta - np.pi), np.sin(theta - np.pi)]

    local_rotation = np.column_stack((local_frame_x, local_frame_y, local_frame_z))

    local_transform = np.eye(4, dtype=np.float32)
    local_transform[:3, :3] = local_rotation
    local_transform[:3, 3] = local_frame_translation

    local_pose = Pose.from_matrix(local_transform)
    return local_pose


# import pdb; pdb.set_trace()
# simulation_app.close()
# import sys; sys.exit(0)

# Now you can find /World/Franka, /World/Target, etc.
world.reset()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()