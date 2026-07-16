"""Release state for the pick-and-place controller."""

from __future__ import annotations

from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class ReleaseState(PnPState):
    """Open the gripper and detach the cube from the CuRobo robot model."""

    phase = PickPlacePhase.RELEASE

    def __init__(self, *, robot: Franka, planner: CuroboPlanner) -> None:
        self._robot = robot
        self._planner = planner

    def update(self) -> StateStep:
        """Issue the release operations and advance directly to return."""
        self._robot.gripper.open()
        self._planner.detach_cube()
        return StateStep(next_phase=PickPlacePhase.RETURN)
