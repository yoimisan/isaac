"""Geometry helpers for the pick-and-place workflow."""

from __future__ import annotations

import numpy as np
from curobo.types.math import Pose as CuRoboPose
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import pose_from_tf_matrix


def sample_cube_pregrasp_pose(cube: DynamicCuboid) -> CuRoboPose:
    """Sample a pre-grasp pose expressed in the cube's local frame."""
    face_index = np.random.randint(2)

    cube_size = cube.get_size()
    offset_length = 1.5 * (np.sqrt(2.0) * cube_size / 2.0)
    theta = np.random.random() * np.pi / 4.0 + np.pi / 3.0

    if face_index == 0:
        local_frame_translation = [
            offset_length * np.cos(theta),
            0.0,
            offset_length * np.sin(theta),
        ]
        local_frame_x = [np.cos(theta - np.pi / 2.0), 0.0, np.sin(theta - np.pi / 2.0)]
        local_frame_y = [0.0, -1.0, 0.0]
        local_frame_z = [np.cos(theta - np.pi), 0.0, np.sin(theta - np.pi)]
    else:
        local_frame_translation = [
            0.0,
            offset_length * np.cos(theta),
            offset_length * np.sin(theta),
        ]
        if theta < np.pi / 2:
            local_frame_x = [0.0, -np.sin(theta), np.cos(theta)]
            local_frame_y = [-1.0, 0.0, 0.0]
            local_frame_z = [0.0, -np.cos(theta), -np.sin(theta)]
        else:
            local_frame_x = [0.0, np.sin(np.pi - theta), np.cos(np.pi - theta)]
            local_frame_y = [1.0, 0.0, 0.0]
            local_frame_z = [0.0, np.cos(np.pi - theta), -np.sin(np.pi - theta)]

    local_transform = np.eye(4, dtype=np.float32)
    local_transform[:3, :3] = np.column_stack((local_frame_x, local_frame_y, local_frame_z))
    local_transform[:3, 3] = local_frame_translation
    return CuRoboPose.from_matrix(local_transform)


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
