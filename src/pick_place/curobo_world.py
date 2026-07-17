"""Runtime synchronization between Isaac Sim rigid bodies and CuRobo obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.wrap.reacher.motion_gen import MotionGen
from isaacsim.robot.manipulators.examples.franka import Franka


class RigidPoseSource(Protocol):
    """PhysX-backed object that exposes an Isaac world pose."""

    @property
    def prim_path(self) -> str: ...

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class RegisteredObstacle:
    """One CuRobo collision object bound to a live Isaac rigid body."""

    name: str
    pose_source: RigidPoseSource
    enabled: bool = True


class CuroboWorldRegistry:
    """Keep CuRobo dynamic obstacle poses synchronized with PhysX state."""

    def __init__(
        self,
        *,
        motion_gen: MotionGen,
        robot: Franka,
        tensor_args: TensorDeviceType,
    ) -> None:
        self._motion_gen = motion_gen
        self._robot = robot
        self._tensor_args = tensor_args
        self._obstacles: dict[str, RegisteredObstacle] = {}

    def register(
        self,
        *,
        name: str,
        pose_source: RigidPoseSource,
        enabled: bool = True,
    ) -> None:
        """Bind an existing CuRobo obstacle to a PhysX-backed pose source."""
        if name in self._obstacles:
            raise ValueError(f"CuRobo obstacle is already registered: {name}")
        if self._motion_gen.world_model.get_obstacle(name) is None:
            available = [obstacle.name for obstacle in self._motion_gen.world_model.objects]
            raise ValueError(
                f"CuRobo obstacle does not exist: {name}; available={available}"
            )
        self._require_physics_handle(pose_source)
        self._obstacles[name] = RegisteredObstacle(
            name=name,
            pose_source=pose_source,
            enabled=enabled,
        )
        self.sync(name)
        self.set_enabled(name, enabled)

    def sync_enabled(self) -> None:
        """Update every enabled dynamic obstacle from its live Isaac pose."""
        for obstacle in self._obstacles.values():
            if obstacle.enabled:
                self.sync(obstacle.name)

    def sync(self, name: str) -> None:
        """Update one obstacle even when it is currently disabled."""
        obstacle = self._get(name)
        self._require_physics_handle(obstacle.pose_source)
        object_position, object_orientation = obstacle.pose_source.get_world_pose()
        base_position, base_orientation = self._robot.get_world_pose()

        world_object = self._make_pose(object_position, object_orientation)
        world_base = self._make_pose(base_position, base_orientation)
        base_object = world_base.inverse().multiply(world_object)
        self._motion_gen.world_collision.update_obstacle_pose(
            name=name,
            w_obj_pose=base_object,
            update_cpu_reference=True,
        )

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable one registered obstacle in CuRobo collision checking."""
        obstacle = self._get(name)
        self._motion_gen.world_collision.enable_obstacle(
            name=name,
            enable=enabled,
        )
        obstacle.enabled = enabled

    def is_enabled(self, name: str) -> bool:
        """Return the registry's collision-enabled state for an obstacle."""
        return self._get(name).enabled

    def _get(self, name: str) -> RegisteredObstacle:
        try:
            return self._obstacles[name]
        except KeyError as error:
            raise KeyError(f"CuRobo obstacle is not registered: {name}") from error

    def _make_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> CuRoboPose:
        return CuRoboPose(
            position=self._tensor_args.to_device(
                [np.asarray(position, dtype=np.float32)]
            ),
            quaternion=self._tensor_args.to_device(
                [np.asarray(orientation, dtype=np.float32)]
            ),
        )

    @staticmethod
    def _require_physics_handle(pose_source: RigidPoseSource) -> None:
        rigid_view = getattr(pose_source, "_rigid_prim_view", None)
        is_valid = getattr(rigid_view, "is_physics_handle_valid", None)
        if callable(is_valid) and not is_valid():
            raise RuntimeError(
                f"PhysX handle is not initialized for {pose_source.prim_path}. "
                "Create or refresh the registry after world.reset()."
            )
