"""Constrained CuRobo lift state for pick-and-place."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class LiftState(PnPState):
    """Lift the grasped cube vertically while holding the tool orientation."""

    phase = PickPlacePhase.LIFT

    def __init__(
        self,
        *,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        lift_offset: float,
        approach_tolerance: float,
        grasp_tolerance: float = 0.06,
    ) -> None:
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._lift_offset = lift_offset
        self._approach_tolerance = approach_tolerance
        self._grasp_tolerance = grasp_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._target_position: np.ndarray | None = None
        self._recovering_from_cube_loss = False

    def enter(self) -> None:
        """Discard the previous lift so the next tick plans from live state."""
        self._trajectory = None
        self._trajectory_index = None
        self._target_position = None
        self._recovering_from_cube_loss = False

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        if self._recovering_from_cube_loss:
            self._robot.gripper.open()
            self._planner.detach_cube()
        self._trajectory = None
        self._trajectory_index = None

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a cube that is no longer moving with the tool center."""
        cube_position, _ = self._cube.get_world_pose()
        tool_position, _ = self._planner.get_tool_world_pose()
        position_error = float(np.linalg.norm(cube_position - tool_position))
        if position_error <= self._grasp_tolerance:
            return None
        return Perturbation(
            reason="cube_lost_during_lift",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Detach the lost cube, then wait before attempting a new grasp."""
        if perturbation.reason == "cube_lost_during_lift":
            self._recovering_from_cube_loss = True
            return PickPlacePhase.WAIT_FOR_STABLE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Execute one vertical-lift waypoint or advance to place."""
        if self._trajectory is None:
            self._start_plan()

        tool_position, _ = self._planner.get_tool_world_pose()
        if (
            np.linalg.norm(tool_position - self._target_position)
            <= self._approach_tolerance
        ):
            return StateStep(next_phase=PickPlacePhase.PLACE)

        self._trajectory_index = min(
            self._trajectory_index,
            len(self._trajectory) - 1,
        )
        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return StateStep(action=action)

    def _start_plan(self) -> None:
        tool_position, tool_orientation = self._planner.get_tool_world_pose()

        self._target_position = np.asarray(tool_position).copy()
        self._target_position[2] += self._lift_offset

        self._trajectory = self._planner.plan_linear_motion(
            self._target_position,
            tool_orientation,
            linear_axis=2,
            project_to_goal_frame=False,
        )
        if self._trajectory is None:
            raise RuntimeError(
                "CuRobo failed to generate a constrained lift trajectory."
            )
        self._trajectory_index = 0
