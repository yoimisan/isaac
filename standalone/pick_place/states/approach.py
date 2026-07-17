"""Approach state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from curobo.types.math import Pose as CuRoboPose
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import (
    compose_poses,
    create_xform,
    curobo_pose_to_numpy,
    sample_cube_pregrasp_pose,
)
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
        approach_tolerance: float,
        cube_motion_tolerance: float = 0.06,
    ) -> None:
        self._world = world
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._approach_tolerance = approach_tolerance
        self._cube_motion_tolerance = cube_motion_tolerance

        self._pregrasp_local_pose: CuRoboPose | None = None
        self._pregrasp_marker: SingleXFormPrim | None = None
        self._trajectory = None
        self._trajectory_index: int | None = None
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._planned_cube_position: np.ndarray | None = None

    def enter(self) -> None:
        """Sample a fresh pre-grasp pose and discard the previous plan."""
        robot_position, _ = self._robot.get_world_pose()
        self._pregrasp_local_pose = sample_cube_pregrasp_pose(
            self._cube,
            robot_position,
        )
        position, orientation = self._compute_pregrasp_world_pose()
        self._pregrasp_marker = create_xform(
            world=self._world,
            prim_path="/World/CubePregrasp",
            name="cube_pregrasp",
            exist_ok=True,
            position=position,
            orientation=orientation,
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

        tool_center_position, _ = self._planner.get_tool_world_pose()
        target_error = np.linalg.norm(
            tool_center_position - self._target_position
        )

        next_phase = None
        if target_error <= self._approach_tolerance:
            next_phase = PickPlacePhase.DESCEND
        return StateStep(action=action, next_phase=next_phase)

    def _start_plan(self) -> None:
        if self._pregrasp_local_pose is None:
            raise RuntimeError("Approach state was updated before enter().")

        cube_position, _ = self._cube.get_world_pose()
        self._planned_cube_position = np.asarray(cube_position).copy()
        self._target_position, self._target_orientation = (
            self._compute_pregrasp_world_pose()
        )
        if self._pregrasp_marker is not None:
            self._pregrasp_marker.set_world_pose(
                self._target_position,
                self._target_orientation,
            )
        self._planner.prepare_obstacle_for_manipulation(self._cube.prim_path)
        self._trajectory = self._planner.plan_to_pose(
            self._target_position,
            self._target_orientation,
        )
        if self._trajectory is None:
            raise RuntimeError("CuRobo failed to generate an approach trajectory.")
        self._trajectory_index = 0
        self._robot.gripper.open()

    def _compute_pregrasp_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pregrasp_local_pose is None:
            raise RuntimeError("Pre-grasp pose has not been sampled.")
        cube_position, cube_orientation = self._cube.get_world_pose()
        local_position, local_orientation = curobo_pose_to_numpy(
            self._pregrasp_local_pose
        )
        return compose_poses(
            cube_position,
            cube_orientation,
            local_position,
            local_orientation,
        )
