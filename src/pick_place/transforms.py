"""Pure NumPy rigid-transform helpers using WXYZ quaternions."""

from __future__ import annotations

import numpy as np


def pose_to_matrix(
    position: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    """Convert a position and WXYZ quaternion or rotation matrix to a transform."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"Position must have shape (3,), got {position.shape}.")

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.shape == (3, 3):
        rotation = orientation
    elif orientation.shape == (4,):
        quaternion_norm = np.linalg.norm(orientation)
        if quaternion_norm <= 1e-12:
            raise ValueError("Cannot build a transform from a zero quaternion.")
        w, x, y, z = orientation / quaternion_norm
        rotation = np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        )
    else:
        raise ValueError(f"Unsupported orientation shape: {orientation.shape}.")

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a homogeneous transform to position and a WXYZ quaternion."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4, 4), got {transform.shape}.")

    # Project small numerical drift back onto SO(3) before quaternion extraction.
    left, _, right_transpose = np.linalg.svd(transform[:3, :3])
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal_index = int(np.argmax(np.diag(rotation)))
        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            )
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            )
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            )
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )

    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return transform[:3, 3].copy(), quaternion


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
    return matrix_to_pose(transform)


def relative_pose(
    parent_position: np.ndarray,
    parent_orientation: np.ndarray,
    child_position: np.ndarray,
    child_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame child pose in a world-frame parent's coordinates."""
    world_parent = pose_to_matrix(parent_position, parent_orientation)
    world_child = pose_to_matrix(child_position, child_orientation)
    return matrix_to_pose(np.linalg.inv(world_parent) @ world_child)
