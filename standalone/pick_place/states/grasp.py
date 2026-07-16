"""Grasp state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class GraspState(PnPState):
    """Close the gripper and attach the cube to the CuRobo robot model."""

    phase = PickPlacePhase.GRASP

    def __init__(
        self,
        *,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        tool_center_prim: SingleXFormPrim,
        grasp_tolerance: float,
    ) -> None:
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._tool_center_prim = tool_center_prim
        self._grasp_tolerance = grasp_tolerance

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a cube that moved out of reach before the gripper closes."""
        cube_position, _ = self._cube.get_world_pose()
        tool_position, _ = self._tool_center_prim.get_world_pose()
        position_error = float(np.linalg.norm(cube_position - tool_position))
        if position_error <= self._grasp_tolerance:
            return None
        return Perturbation(
            reason="cube_moved_before_grasp",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Wait for the cube and arm to settle before reacquiring the cube."""
        if perturbation.reason == "cube_moved_before_grasp":
            return PickPlacePhase.WAIT_FOR_STABLE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Issue the grasp operations and advance directly to lift."""
        self._robot.gripper.close()
        self._planner.attach_cube()
        return StateStep(next_phase=PickPlacePhase.LIFT)
