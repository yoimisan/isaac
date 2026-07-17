"""Geometry helpers for the pick-and-place workflow."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim

from pick_place.transforms import matrix_to_pose, pose_to_matrix


def sample_cube_pregrasp_pose(
    cube: DynamicCuboid,
    franka_position: np.ndarray,
    franka_orientation: np.ndarray,
    *,
    min_grasp_angle: float = np.pi / 3.0,
    max_grasp_angle: float = np.pi / 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the tool-center pre-grasp from canonical cube axes."""
    if not 0.0 < min_grasp_angle <= max_grasp_angle <= np.pi / 2.0:
        raise ValueError(
            "Grasp angle must satisfy "
            "0 < min_grasp_angle <= max_grasp_angle <= pi / 2."
        )

    cube_position, cube_orientation = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float64)
    axis_i, axis_j, axis_k = get_cube_canonical_axes(
        cube_position,
        cube_orientation,
        franka_position,
        franka_orientation,
    )

    theta = np.random.uniform(min_grasp_angle, max_grasp_angle)
    tool_x = np.cos(theta) * axis_k + np.sin(theta) * axis_i
    tool_y = -axis_j
    tool_z = np.cos(theta) * axis_i - np.sin(theta) * axis_k

    cube_size = float(cube.get_size())
    offset_length = 1.5 * (np.sqrt(2.0) * cube_size / 2.0)
    pregrasp_position = cube_position - offset_length * tool_z

    world_pregrasp = np.eye(4)
    world_pregrasp[:3, :3] = np.column_stack((tool_x, tool_y, tool_z))
    world_pregrasp[:3, 3] = pregrasp_position
    world_cube = pose_to_matrix(cube_position, cube_orientation)
    cube_pregrasp = np.linalg.inv(world_cube) @ world_pregrasp
    return matrix_to_pose(cube_pregrasp)


def get_cube_canonical_axes(
    cube_position: np.ndarray,
    cube_orientation: np.ndarray,
    franka_position: np.ndarray,
    franka_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a right-handed cube frame with stable Franka-axis semantics.

    The returned vectors are signed cube-local axes expressed in world space.
    ``k`` is the cube axis most aligned with Franka +Z. Among the four rotations
    about ``k``, ``i`` and ``j`` are selected from the p/q quadrants described
    by the grasp formulation, with ties favoring Franka +X and +Y alignment.
    """
    cube_position = np.asarray(cube_position, dtype=np.float64)
    if cube_position.shape != (3,):
        raise ValueError(
            f"Cube position must have shape (3,), got {cube_position.shape}."
        )

    cube_rotation = pose_to_matrix(
        np.zeros(3),
        cube_orientation,
    )[:3, :3]
    franka_transform = pose_to_matrix(franka_position, franka_orientation)
    franka_rotation = franka_transform[:3, :3]
    cube_position_in_franka = (
        franka_rotation.T
        @ (cube_position - franka_transform[:3, 3])
    )
    cube_axes = [cube_rotation[:, index] for index in range(3)]
    franka_x = franka_rotation[:, 0]
    franka_y = franka_rotation[:, 1]
    franka_z = franka_rotation[:, 2]
    if cube_position_in_franka[1] >= 0.0:
        axis_p = franka_y
        axis_q = franka_x
    else:
        axis_p = franka_x
        axis_q = -franka_y

    target_i = axis_p + axis_q
    target_i /= np.linalg.norm(target_i)
    target_j = axis_p - axis_q
    target_j /= np.linalg.norm(target_j)

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i_index, base_i in enumerate(cube_axes):
        for i_sign in (-1.0, 1.0):
            axis_i = i_sign * base_i
            for j_index, base_j in enumerate(cube_axes):
                if j_index == i_index:
                    continue
                for j_sign in (-1.0, 1.0):
                    axis_j = j_sign * base_j
                    axis_k = np.cross(axis_i, axis_j)
                    candidates.append((axis_i, axis_j, axis_k))

    maximum_up_alignment = max(
        float(np.dot(axis_k, franka_z))
        for _, _, axis_k in candidates
    )
    upward_candidates = [
        (axis_i, axis_j, axis_k)
        for axis_i, axis_j, axis_k in candidates
        if np.dot(axis_k, franka_z) >= maximum_up_alignment - 1e-8
    ]

    def horizontal_score(
        axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[float, float]:
        axis_i, axis_j, _ = axes
        quadrant_alignment = float(
            np.dot(axis_i, target_i) + np.dot(axis_j, target_j)
        )
        franka_axis_alignment = float(
            np.dot(axis_i, franka_x) + np.dot(axis_j, franka_y)
        )
        return quadrant_alignment, franka_axis_alignment

    axis_i, axis_j, axis_k = max(upward_candidates, key=horizontal_score)
    return axis_i.copy(), axis_j.copy(), axis_k.copy()


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
