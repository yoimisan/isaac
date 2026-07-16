"""Return-home state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class ReturnState(PnPState):
    """Move the arm back to the joint pose captured at episode reset."""

    phase = PickPlacePhase.RETURN

    def __init__(
        self,
        *,
        robot: Franka,
        planner: CuroboPlanner,
        reset_arm_positions: np.ndarray,
        approach_tolerance: float,
    ) -> None:
        self._robot = robot
        self._planner = planner
        self._reset_arm_positions = reset_arm_positions.copy()
        self._approach_tolerance = approach_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._is_complete = False

    @property
    def is_complete(self) -> bool:
        """Return whether the arm reached its reset joint pose."""
        return self._is_complete

    def enter(self) -> None:
        """Discard the previous return plan and clear completion."""
        self._trajectory = None
        self._trajectory_index = None
        self._is_complete = False

    def exit(self) -> None:
        """Drop trajectory data while preserving the completion result."""
        self._trajectory = None
        self._trajectory_index = None

    def update(self) -> StateStep:
        """Execute one return waypoint or finish the episode in idle."""
        if self._trajectory is None:
            self._start_plan()

        self._trajectory_index = min(
            self._trajectory_index,
            len(self._trajectory) - 1,
        )
        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1

        arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ]
        next_phase = None
        if (
            np.linalg.norm(arm_positions - self._reset_arm_positions)
            <= self._approach_tolerance
        ):
            self._is_complete = True
            next_phase = PickPlacePhase.IDLE
        return StateStep(action=action, next_phase=next_phase)

    def _start_plan(self) -> None:
        self._trajectory = self._planner.plan_to_joint_positions(
            self._reset_arm_positions
        )
        if self._trajectory is None:
            raise RuntimeError("CuRobo failed to generate a return trajectory.")
        self._trajectory_index = 0
