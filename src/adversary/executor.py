"""Isaac Sim implementation of disturbance world mutations."""

from __future__ import annotations

from collections.abc import Mapping

import carb
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

from adversary.types import DisturbanceCommand, ObjectPoseOffset


class IsaacSimDisturbanceExecutor:
    """Apply typed disturbance commands through runtime physics objects."""

    def __init__(self, objects: Mapping[str, DynamicCuboid]) -> None:
        self._objects = dict(objects)

    def execute(self, command: DisturbanceCommand) -> None:
        if isinstance(command, ObjectPoseOffset):
            self._apply_object_pose_offset(command)
            return
        raise TypeError(f"Unsupported disturbance command: {type(command).__name__}")

    def _apply_object_pose_offset(self, command: ObjectPoseOffset) -> None:
        try:
            target = self._objects[command.target_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown disturbance target {command.target_name!r}"
            ) from error

        position, orientation = target.get_world_pose()
        previous_position = np.asarray(position, dtype=np.float64).copy()
        next_position = previous_position + np.asarray(command.position_offset)
        target.set_world_pose(position=next_position, orientation=orientation)
        target.set_linear_velocity(np.zeros(3))
        target.set_angular_velocity(np.zeros(3))

        carb.log_warn(
            f"Naughty ghost moved {command.target_name}: "
            f"{previous_position} -> {next_position}; reason={command.reason}"
        )
