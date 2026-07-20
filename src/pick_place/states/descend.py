"""Constrained CuRobo descend state for pick-and-place."""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states.base import (
    CubeCollisionMode,
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class DescendState(PnPState):
    """Move the tool center to the cube along its local approach axis."""

    phase = PickPlacePhase.DESCEND
    cube_collision_mode = CubeCollisionMode.IGNORED

    def __init__(
        self,
        *,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        approach_tolerance: float,
        cube_motion_tolerance: float = 0.06,
    ) -> None:
        self._cube = cube
        self._planner = planner
        self._approach_tolerance = approach_tolerance
        self._cube_motion_tolerance = cube_motion_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._final_waypoint_hold_steps = 0
        self._planned_cube_position: np.ndarray | None = None

    def enter(self) -> None:
        """Discard the previous descent so the next tick plans from live state."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0
        self._planned_cube_position = None

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0

    def is_success(self) -> bool:
        """Return whether the tool center reached the cube."""
        tool_position, _ = self._planner.get_tool_world_pose()
        cube_position, _ = self._cube.get_world_pose()
        return bool(
            np.linalg.norm(tool_position - cube_position)
            <= self._approach_tolerance
        )

    def detect_perturbation(self) -> Perturbation | None:
        """Invalidate the descent when the cube leaves its planned pose."""
        if self._planned_cube_position is None:
            return None

        cube_position, _ = self._cube.get_world_pose()
        position_error = float(
            np.linalg.norm(np.asarray(cube_position) - self._planned_cube_position)
        )
        if position_error <= self._cube_motion_tolerance:
            return None
        return Perturbation(
            reason="cube_moved_during_descend",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Return directly to a fresh pre-grasp approach."""
        if perturbation.reason == "cube_moved_during_descend":
            return PickPlacePhase.APPROACH
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Execute a constrained waypoint or advance to grasp at the cube."""
        if self.is_success():
            return StateStep(next_phase=PickPlacePhase.GRASP)

        if self._trajectory is None:
            tool_position, tool_orientation = self._planner.get_tool_world_pose()
            cube_position, _ = self._cube.get_world_pose()
            self._start_plan(tool_position, tool_orientation, cube_position)

        if self._trajectory_index >= len(self._trajectory):
            if (
                self._final_waypoint_hold_steps
                < self.max_final_waypoint_hold_steps
            ):
                self._final_waypoint_hold_steps += 1
                return StateStep(action=self._trajectory[-1])
            carb.log_warn(
                "Descend trajectory exhausted before success; replanning."
            )
            self._trajectory = None
            self._trajectory_index = None
            self._final_waypoint_hold_steps = 0
            return StateStep()

        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return StateStep(action=action)

    def _start_plan(
        self,
        tool_position: np.ndarray,
        tool_orientation: np.ndarray,
        cube_position: np.ndarray,
    ) -> None:
        self._planned_cube_position = np.asarray(cube_position).copy()

        approach_distance = float(
            np.linalg.norm(np.asarray(tool_position) - cube_position)
        )
        self._trajectory = self._planner.plan_linear_approach(
            cube_position,
            tool_orientation,
            approach_distance=approach_distance,
            linear_axis=2,
        )
        if not self._trajectory:
            raise RuntimeError(
                "CuRobo failed to generate a constrained descend trajectory."
            )
        self._trajectory_index = 0
        self._final_waypoint_hold_steps = 0
