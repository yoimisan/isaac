"""Grasp state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    CubeCollisionMode,
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class GraspState(PnPState):
    """Close the gripper and attach the cube to the CuRobo robot model."""

    phase = PickPlacePhase.GRASP
    cube_collision_mode = CubeCollisionMode.IGNORED

    def __init__(
        self,
        *,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        grasp_tolerance: float,
    ) -> None:
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._grasp_tolerance = grasp_tolerance

    def is_success(self) -> bool:
        """Return whether the cube remains within grasping distance."""
        cube_position, _ = self._cube.get_world_pose()
        tool_position, _ = self._planner.get_tool_world_pose()
        return bool(
            np.linalg.norm(cube_position - tool_position)
            <= self._grasp_tolerance
        )

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a cube that moved out of reach before the gripper closes."""
        cube_position, _ = self._cube.get_world_pose()
        tool_position, _ = self._planner.get_tool_world_pose()
        position_error = float(np.linalg.norm(cube_position - tool_position))
        if position_error <= self._grasp_tolerance:
            return None
        return Perturbation(
            reason="cube_moved_before_grasp",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Return directly to a fresh pre-grasp approach."""
        if perturbation.reason == "cube_moved_before_grasp":
            return PickPlacePhase.APPROACH
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Issue the grasp operations and advance directly to lift."""
        if not self.is_success():
            return StateStep(next_phase=PickPlacePhase.APPROACH)
        self._robot.gripper.close()
        self._planner.attach_cube()
        return StateStep(next_phase=PickPlacePhase.LIFT)
