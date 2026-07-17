"""Geometry helpers for the pick-and-place workflow."""

from __future__ import annotations

import numpy as np
from curobo.types.math import Pose as CuRoboPose
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import pose_from_tf_matrix


def sample_cube_pregrasp_pose(
    cube: DynamicCuboid,
    robot_world_position: np.ndarray,
    *,
    min_elevation: float = np.deg2rad(65.0),
    max_elevation: float = np.deg2rad(80.0),
) -> CuRoboPose:
    """Sample a gravity-aware pre-grasp and express it in the cube frame.

    The position and tool axes are constrained in the world frame first. The
    resulting pose is then converted to cube-local coordinates so a flipped
    cube cannot turn a nominally upward pre-grasp into a goal below the floor.
    """
    if not 0.0 < min_elevation <= max_elevation < np.pi / 2.0:
        raise ValueError(
            "Pre-grasp elevation must satisfy "
            "0 < min_elevation <= max_elevation < pi / 2."
        )

    cube_position, cube_orientation = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float64)
    robot_world_position = np.asarray(robot_world_position, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0])

    # Approach from the robot-facing side. In this task the Franka is behind
    # the cube on world -X, which makes both tool X and Z project onto +X.
    horizontal = robot_world_position - cube_position
    horizontal -= np.dot(horizontal, world_up) * world_up
    horizontal_norm = np.linalg.norm(horizontal)
    if horizontal_norm <= 1e-8:
        horizontal = np.array([-1.0, 0.0, 0.0])
    else:
        horizontal /= horizontal_norm
    if horizontal[0] >= -1e-6:
        horizontal = np.array([-1.0, 0.0, 0.0])

    elevation = np.random.uniform(min_elevation, max_elevation)
    cube_size = float(cube.get_size())
    offset_length = 1.5 * (np.sqrt(2.0) * cube_size / 2.0)
    radial = (
        np.cos(elevation) * horizontal
        + np.sin(elevation) * world_up
    )
    pregrasp_position = cube_position + offset_length * radial

    # Tool +Z points from the pre-grasp toward the cube. Tool +X is the
    # world-up projection orthogonal to Z, with a positive world-X component.
    tool_z = -radial
    tool_x = world_up - np.dot(world_up, tool_z) * tool_z
    tool_x /= np.linalg.norm(tool_x)
    if tool_x[0] < 0.0:
        tool_x = -tool_x
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)

    world_pregrasp = np.eye(4)
    world_pregrasp[:3, :3] = np.column_stack((tool_x, tool_y, tool_z))
    world_pregrasp[:3, 3] = pregrasp_position

    world_cube = pose_to_matrix(cube_position, cube_orientation)
    cube_pregrasp = np.linalg.inv(world_cube) @ world_pregrasp

    vertical_half_extent = (
        0.5 * cube_size * np.sum(np.abs(world_cube[2, :3]))
    )
    vertical_offset = float(np.dot(pregrasp_position - cube_position, world_up))
    minimum_clearance = 0.05 * cube_size
    if vertical_offset <= vertical_half_extent + minimum_clearance:
        raise RuntimeError(
            "Sampled pre-grasp does not clear the cube in world +Z: "
            f"offset={vertical_offset}, half_extent={vertical_half_extent}."
        )
    if tool_x[0] <= 0.0 or tool_z[0] <= 0.0:
        raise RuntimeError(
            "Sampled tool X and Z axes must have positive world-X components."
        )

    return CuRoboPose.from_matrix(cube_pregrasp.astype(np.float32))


def create_xform(
    world: World,
    prim_path: str,
    name: str,
    exist_ok: bool,
    *,
    position: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> SingleXFormPrim:
    """Create and register an Xform wrapper in the world scene."""
    if world.scene.object_exists(name) and exist_ok:
        world.scene.remove_object(name)
    return world.scene.add(
        SingleXFormPrim(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            position=position,
        )
    )


def pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Convert a pose using a WXYZ quaternion or rotation matrix to a homogeneous matrix."""
    orientation = np.asarray(orientation)
    transform = np.eye(4)
    if orientation.shape == (3, 3):
        transform[:3, :3] = orientation
    elif orientation.shape == (4,):
        w, x, y, z = orientation
        transform[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
    else:
        raise ValueError(f"Unsupported orientation shape: {orientation.shape}")
    transform[:3, 3] = position
    return transform


def compose_poses(
    parent_position: np.ndarray,
    parent_orientation: np.ndarray,
    local_position: np.ndarray,
    local_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a world parent pose with a parent-local child pose."""
    transform = pose_to_matrix(
        parent_position,
        parent_orientation,
    ) @ pose_to_matrix(local_position, local_orientation)
    return pose_from_tf_matrix(transform)


def relative_pose(
    parent_position: np.ndarray,
    parent_orientation: np.ndarray,
    child_position: np.ndarray,
    child_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame child pose in a world-frame parent's coordinates."""
    world_parent = pose_to_matrix(parent_position, parent_orientation)
    world_child = pose_to_matrix(child_position, child_orientation)
    return pose_from_tf_matrix(np.linalg.inv(world_parent) @ world_child)


def curobo_pose_to_numpy(pose: CuRoboPose) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single CuRobo pose to NumPy arrays using WXYZ quaternions."""
    return (
        pose.position.squeeze(0).detach().cpu().numpy(),
        pose.quaternion.squeeze(0).detach().cpu().numpy(),
    )
