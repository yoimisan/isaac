"""Return-home state for the pick-and-place controller."""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    CubeCollisionMode,
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class ReturnState(PnPState):
    """Move the arm back to the joint pose captured at episode reset."""

    phase = PickPlacePhase.RETURN
    cube_collision_mode = CubeCollisionMode.WORLD_OBSTACLE

    def __init__(
        self,
        *,
        world: World,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        reset_arm_positions: np.ndarray,
        approach_tolerance: float,
        placement_tolerance: float = 0.06,
    ) -> None:
        self._world = world
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._reset_arm_positions = reset_arm_positions.copy()
        self._approach_tolerance = approach_tolerance
        self._placement_tolerance = placement_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._final_waypoint_hold_steps = 0
        self._is_complete = False

    @property
    def is_complete(self) -> bool:
        """Return whether the arm reached its reset joint pose."""
        return self._is_complete

    def enter(self) -> None:
        """Discard the previous return plan and clear completion."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0
        self._is_complete = False

    def exit(self) -> None:
        """Drop trajectory data while preserving the completion result."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0

    def is_success(self) -> bool:
        """Return whether the arm reached its reset joint pose."""
        arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ]
        return bool(
            np.linalg.norm(arm_positions - self._reset_arm_positions)
            <= self._approach_tolerance
        )

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a released cube that leaves the target during return."""
        cube_position, _ = self._cube.get_world_pose()
        target_region = self._world.scene.get_object("target_region")
        target_position, _ = target_region.get_world_pose()
        position_error = float(
            np.linalg.norm(cube_position[:2] - target_position[:2])
        )
        if position_error <= self._placement_tolerance:
            return None
        return Perturbation(
            reason="cube_left_target_during_return",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Reacquire a cube that left the target during return."""
        if perturbation.reason == "cube_left_target_during_return":
            return PickPlacePhase.APPROACH
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Execute one return waypoint or finish the episode in idle."""
        if self.is_success():
            self._is_complete = True
            return StateStep(next_phase=PickPlacePhase.IDLE)

        if self._trajectory is None:
            self._start_plan()

        if self._trajectory_index >= len(self._trajectory):
            if (
                self._final_waypoint_hold_steps
                < self.max_final_waypoint_hold_steps
            ):
                self._final_waypoint_hold_steps += 1
                return StateStep(action=self._trajectory[-1])
            carb.log_warn("Return trajectory exhausted before success; replanning.")
            self._trajectory = None
            self._trajectory_index = None
            self._final_waypoint_hold_steps = 0
            return StateStep()

        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return StateStep(action=action)

    def _start_plan(self) -> None:
        self._trajectory = self._planner.plan_to_joint_positions(
            self._reset_arm_positions
        )
        if not self._trajectory:
            raise RuntimeError("CuRobo failed to generate a return trajectory.")
        self._trajectory_index = 0
        self._final_waypoint_hold_steps = 0
