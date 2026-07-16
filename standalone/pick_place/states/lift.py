"""Constrained CuRobo lift state for pick-and-place."""

from __future__ import annotations

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import pose_from_tf_matrix
from isaacsim.core.utils.types import ArticulationAction

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import pose_to_matrix
from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class LiftState(PnPState):
    """Lift the grasped cube vertically while holding the tool orientation."""

    phase = PickPlacePhase.LIFT

    def __init__(
        self,
        *,
        planner: CuroboPlanner,
        base_prim: SingleXFormPrim,
        tool_center_prim: SingleXFormPrim,
        lift_offset: float,
        approach_tolerance: float,
    ) -> None:
        self._planner = planner
        self._base_prim = base_prim
        self._tool_center_prim = tool_center_prim
        self._lift_offset = lift_offset
        self._approach_tolerance = approach_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._target_position: np.ndarray | None = None

    def enter(self) -> None:
        """Discard the previous lift so the next tick plans from live state."""
        self._trajectory = None
        self._trajectory_index = None
        self._target_position = None

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        self._trajectory = None
        self._trajectory_index = None

    def update(self) -> StateStep:
        """Execute one vertical-lift waypoint or advance to place."""
        if self._trajectory is None:
            self._start_plan()

        tool_position, _ = self._tool_center_prim.get_world_pose()
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
        tool_position, tool_orientation = self._tool_center_prim.get_world_pose()
        base_position, base_orientation = self._base_prim.get_world_pose()

        self._target_position = np.asarray(tool_position).copy()
        self._target_position[2] += self._lift_offset
        world_goal = pose_to_matrix(self._target_position, tool_orientation)
        world_to_base = np.linalg.inv(
            pose_to_matrix(base_position, base_orientation)
        )
        goal_position, goal_orientation = pose_from_tf_matrix(
            world_to_base @ world_goal
        )

        self._trajectory = self._planner.plan_linear_motion(
            goal_position,
            goal_orientation,
            linear_axis=2,
            project_to_goal_frame=False,
        )
        if self._trajectory is None:
            raise RuntimeError(
                "CuRobo failed to generate a constrained lift trajectory."
            )
        self._trajectory_index = 0
