"""Recovery wait state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class WaitForStableState(PnPState):
    """Wait until the cube and arm settle before replanning an approach."""

    phase = PickPlacePhase.WAIT_FOR_STABLE

    def __init__(
        self,
        *,
        robot: Franka,
        cube: DynamicCuboid,
        arm_joint_indices: list[int],
        cube_linear_velocity_tolerance: float = 0.05,
        cube_angular_velocity_tolerance: float = 0.10,
        arm_velocity_tolerance: float = 0.05,
    ) -> None:
        self._robot = robot
        self._cube = cube
        self._arm_joint_indices = arm_joint_indices
        self._cube_linear_velocity_tolerance = cube_linear_velocity_tolerance
        self._cube_angular_velocity_tolerance = cube_angular_velocity_tolerance
        self._arm_velocity_tolerance = arm_velocity_tolerance

    def detect_perturbation(self) -> Perturbation | None:
        """Treat motion as expected input while waiting rather than a disturbance."""
        return None

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Reject recovery requests because waiting already is a recovery state."""
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Continue waiting or request a fresh approach plan."""
        joint_state = self._robot.get_joints_state()
        arm_velocity = joint_state.velocities[self._arm_joint_indices]
        cube_is_stable = bool(
            np.linalg.norm(self._cube.get_linear_velocity())
            <= self._cube_linear_velocity_tolerance
            and np.linalg.norm(self._cube.get_angular_velocity())
            <= self._cube_angular_velocity_tolerance
        )
        arm_is_stable = bool(
            np.linalg.norm(arm_velocity) <= self._arm_velocity_tolerance
        )
        next_phase = (
            PickPlacePhase.APPROACH
            if cube_is_stable and arm_is_stable
            else None
        )
        return StateStep(next_phase=next_phase)
