"""Approach state for the pick-and-place controller."""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import (
    create_xform,
    sample_cube_pregrasp_pose,
)
from pick_place.states.base import (
    CubeCollisionMode,
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)
from pick_place.transforms import compose_poses


class ApproachState(PnPState):
    """Plan and execute motion to a cube-local pre-grasp pose."""

    phase = PickPlacePhase.APPROACH
    cube_collision_mode = CubeCollisionMode.WORLD_OBSTACLE

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

        self._pregrasp_local_position: np.ndarray | None = None
        self._pregrasp_local_orientation: np.ndarray | None = None
        self._pregrasp_marker: SingleXFormPrim | None = None
        self._trajectory = None
        self._trajectory_index: int | None = None
        self._final_waypoint_hold_steps = 0
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._planned_cube_position: np.ndarray | None = None

    def enter(self) -> None:
        """Sample a fresh pre-grasp pose and discard the previous plan."""
        franka_position, franka_orientation = self._robot.get_world_pose()
        (
            self._pregrasp_local_position,
            self._pregrasp_local_orientation,
        ) = sample_cube_pregrasp_pose(
            self._cube,
            franka_position,
            franka_orientation,
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
        self._final_waypoint_hold_steps = 0
        self._target_position = None
        self._target_orientation = None
        self._planned_cube_position = None

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0

    def is_success(self) -> bool:
        """Return whether the tool reached the current pre-grasp pose."""
        if self._target_position is None:
            return False
        tool_position, _ = self._planner.get_tool_world_pose()
        return bool(
            np.linalg.norm(tool_position - self._target_position)
            <= self._approach_tolerance
        )

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
        """Sample a fresh pre-grasp pose and plan again."""
        if perturbation.reason == "cube_moved_during_approach":
            return PickPlacePhase.APPROACH
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Return the next approach waypoint or finish at the pre-grasp pose."""
        if self.is_success():
            return StateStep(next_phase=PickPlacePhase.DESCEND)

        if self._trajectory is None:
            self._start_plan()

        if self._trajectory_index >= len(self._trajectory):
            if (
                self._final_waypoint_hold_steps
                < self.max_final_waypoint_hold_steps
            ):
                self._final_waypoint_hold_steps += 1
                return StateStep(action=self._trajectory[-1])
            carb.log_warn(
                "Approach trajectory exhausted before success; replanning."
            )
            self._trajectory = None
            self._trajectory_index = None
            self._final_waypoint_hold_steps = 0
            return StateStep()

        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return StateStep(action=action)

    def _start_plan(self) -> None:
        if (
            self._pregrasp_local_position is None
            or self._pregrasp_local_orientation is None
        ):
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
        self._trajectory = self._planner.plan_to_pose(
            self._target_position,
            self._target_orientation,
        )
        if not self._trajectory:
            raise RuntimeError("CuRobo failed to generate an approach trajectory.")
        self._trajectory_index = 0
        self._final_waypoint_hold_steps = 0
        self._robot.gripper.open()

    def _compute_pregrasp_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._pregrasp_local_position is None
            or self._pregrasp_local_orientation is None
        ):
            raise RuntimeError("Pre-grasp pose has not been sampled.")
        cube_position, cube_orientation = self._cube.get_world_pose()
        return compose_poses(
            cube_position,
            cube_orientation,
            self._pregrasp_local_position,
            self._pregrasp_local_orientation,
        )
