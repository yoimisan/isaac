"""Approach state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import create_cube_pregrasp_frame
from pick_place.states.base import Perturbation, PickPlacePhase, PnPState, StateStep


class ApproachState(PnPState):
    """Plan and execute motion to a cube-local pre-grasp pose."""

    phase = PickPlacePhase.APPROACH

    def __init__(
        self,
        *,
        world: World,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        base_prim: SingleXFormPrim,
        tool_center_prim: SingleXFormPrim,
        approach_tolerance: float,
        cube_motion_tolerance: float = 0.06,
    ) -> None:
        self._world = world
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._base_prim = base_prim
        self._tool_center_prim = tool_center_prim
        self._approach_tolerance = approach_tolerance
        self._cube_motion_tolerance = cube_motion_tolerance

        self._pregrasp_prim: SingleXFormPrim | None = None
        self._trajectory = None
        self._trajectory_index: int | None = None
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._planned_cube_position: np.ndarray | None = None

    def enter(self) -> None:
        """Sample a fresh pre-grasp pose and discard the previous plan."""
        self._pregrasp_prim = create_cube_pregrasp_frame(
            self._world,
            self._cube,
            exist_ok=True,
        )
        self._trajectory = None
        self._trajectory_index = None
        self._target_position = None
        self._target_orientation = None
        self._planned_cube_position = None

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        self._trajectory = None
        self._trajectory_index = None

    def detect_perturbation(self) -> Perturbation | None:
        """Invalidate the approach plan when the cube leaves its planned pose."""
        if self._planned_cube_position is None:
            return None

        cube_position, _ = self._cube.get_world_pose()
        position_error = float(
            np.linalg.norm(np.asarray(cube_position) - self._planned_cube_position)
        )
        if position_error <= self._cube_motion_tolerance:
            return None
        return Perturbation(
            reason="cube_moved_during_approach",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Wait for physics to settle before sampling and planning again."""
        if perturbation.reason == "cube_moved_during_approach":
            return PickPlacePhase.WAIT_FOR_STABLE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Return the next approach waypoint or finish at the pre-grasp pose."""
        if self._trajectory is None:
            self._start_plan()

        self._trajectory_index = min(
            self._trajectory_index,
            len(self._trajectory) - 1,
        )
        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1

        tool_center_position, _ = self._tool_center_prim.get_world_pose()
        target_error = np.linalg.norm(
            tool_center_position - self._target_position
        )

        next_phase = None
        if target_error <= self._approach_tolerance:
            next_phase = PickPlacePhase.DESCEND
        return StateStep(action=action, next_phase=next_phase)

    def _start_plan(self) -> None:
        if self._pregrasp_prim is None:
            raise RuntimeError("Approach state was updated before enter().")

        cube_position, _ = self._cube.get_world_pose()
        self._planned_cube_position = np.asarray(cube_position).copy()
        self._target_position, self._target_orientation = (
            self._pregrasp_prim.get_world_pose()
        )
        transform = get_relative_transform(
            source_prim=self._pregrasp_prim.prim,
            target_prim=self._base_prim.prim,
        )
        position, orientation = pose_from_tf_matrix(transform)
        self._trajectory = self._planner.plan_to_pose(position, orientation)
        if self._trajectory is None:
            raise RuntimeError("CuRobo failed to generate an approach trajectory.")
        self._trajectory_index = 0
        self._robot.gripper.open()
