"""Release state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class ReleaseState(PnPState):
    """Open the gripper and detach the cube from the CuRobo robot model."""

    phase = PickPlacePhase.RELEASE

    def __init__(
        self,
        *,
        world: World,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        placement_tolerance: float,
        grasp_tolerance: float = 0.06,
    ) -> None:
        self._world = world
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._placement_tolerance = placement_tolerance
        self._grasp_tolerance = grasp_tolerance
        self._recovering_from_cube_loss = False

    def enter(self) -> None:
        """Clear recovery data from an earlier release attempt."""
        self._recovering_from_cube_loss = False

    def exit(self) -> None:
        """Clean up CuRobo attachment when the cube was already lost."""
        if self._recovering_from_cube_loss:
            self._robot.gripper.open()
            self._planner.detach_cube()

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a cube that left the target pose before release."""
        cube_position, _ = self._cube.get_world_pose()
        target_region = self._world.scene.get_object("target_region")
        target_position, _ = target_region.get_world_pose()
        target_cube_position = np.asarray(target_position).copy()
        target_cube_position[2] += self._cube.get_size() / 2.0
        position_error = float(
            np.linalg.norm(cube_position - target_cube_position)
        )
        if position_error <= self._placement_tolerance:
            return None

        tool_position, _ = self._planner.get_tool_world_pose()
        grasp_error = float(np.linalg.norm(cube_position - tool_position))
        if grasp_error > self._grasp_tolerance:
            return Perturbation(
                reason="cube_lost_before_release",
                metrics={
                    "grasp_error": grasp_error,
                    "target_error": position_error,
                },
            )
        return Perturbation(
            reason="cube_moved_before_release",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Keep the grasp and generate a fresh place plan."""
        if perturbation.reason == "cube_lost_before_release":
            self._recovering_from_cube_loss = True
            return PickPlacePhase.WAIT_FOR_STABLE
        if perturbation.reason == "cube_moved_before_release":
            self._recovering_from_cube_loss = True
            return PickPlacePhase.WAIT_FOR_STABLE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Issue the release operations and advance directly to return."""
        self._robot.gripper.open()
        self._planner.detach_cube()
        return StateStep(next_phase=PickPlacePhase.RETURN)
