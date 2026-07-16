"""Grasp state for the pick-and-place controller."""

from __future__ import annotations

from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class GraspState(PnPState):
    """Close the gripper and attach the cube to the CuRobo robot model."""

    phase = PickPlacePhase.GRASP

    def __init__(self, *, robot: Franka, planner: CuroboPlanner) -> None:
        self._robot = robot
        self._planner = planner

    def update(self) -> StateStep:
        """Issue the grasp operations and advance directly to lift."""
        self._robot.gripper.close()
        self._planner.attach_cube()
        return StateStep(next_phase=PickPlacePhase.LIFT)
