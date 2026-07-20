"""Place state for the pick-and-place controller."""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import create_xform, get_cube_canonical_axes
from pick_place.states.base import (
    CubeCollisionMode,
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)
from pick_place.transforms import (
    compose_poses,
    matrix_to_pose,
    pose_to_matrix,
    relative_pose,
)


class PlaceState(PnPState):
    """Plan and execute motion that carries the cube to the target region."""

    phase = PickPlacePhase.PLACE
    cube_collision_mode = CubeCollisionMode.ATTACHED

    def __init__(
        self,
        *,
        world: World,
        robot: Franka,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        approach_tolerance: float,
        target_motion_tolerance: float = 0.02,
        grasp_tolerance: float = 0.06,
    ) -> None:
        self._world = world
        self._robot = robot
        self._cube = cube
        self._planner = planner
        self._approach_tolerance = approach_tolerance
        self._target_motion_tolerance = target_motion_tolerance
        self._grasp_tolerance = grasp_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._final_waypoint_hold_steps = 0
        self._target_cube_marker: SingleXFormPrim | None = None
        self._target_tool_marker: SingleXFormPrim | None = None
        self._target_cube_position: np.ndarray | None = None
        self._target_cube_orientation: np.ndarray | None = None
        self._planned_target_position: np.ndarray | None = None
        self._recovering_from_cube_loss = False

    def enter(self) -> None:
        """Discard the previous place plan and target frame."""
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0
        self._target_cube_marker = None
        self._target_tool_marker = None
        self._target_cube_position = None
        self._target_cube_orientation = None
        self._planned_target_position = None
        self._recovering_from_cube_loss = False

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        if self._recovering_from_cube_loss:
            self._robot.gripper.open()
            self._planner.detach_cube()
        self._trajectory = None
        self._trajectory_index = None
        self._final_waypoint_hold_steps = 0

    def is_success(self) -> bool:
        """Return whether the cube reached the current placement target."""
        if self._target_cube_position is None:
            return False
        cube_position, _ = self._cube.get_world_pose()
        return bool(
            np.linalg.norm(cube_position - self._target_cube_position)
            <= self._approach_tolerance
        )

    def detect_perturbation(self) -> Perturbation | None:
        """Detect a lost cube or a target that invalidated the place plan."""
        cube_position, _ = self._cube.get_world_pose()
        tool_position, _ = self._planner.get_tool_world_pose()
        grasp_error = float(np.linalg.norm(cube_position - tool_position))
        if grasp_error > self._grasp_tolerance:
            return Perturbation(
                reason="cube_lost_during_place",
                metrics={"position_error": grasp_error},
            )

        if self._planned_target_position is None:
            return None

        target_region = self._world.scene.get_object("target_region")
        target_position, _ = target_region.get_world_pose()
        position_error = float(
            np.linalg.norm(
                np.asarray(target_position) - self._planned_target_position
            )
        )
        if position_error <= self._target_motion_tolerance:
            return None
        return Perturbation(
            reason="target_moved_during_place",
            metrics={"position_error": position_error},
        )

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Re-enter place so a fresh target pose and trajectory are generated."""
        if perturbation.reason == "cube_lost_during_place":
            self._recovering_from_cube_loss = True
            return PickPlacePhase.APPROACH
        if perturbation.reason == "target_moved_during_place":
            return PickPlacePhase.PLACE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Execute one place waypoint or advance to release at the target."""
        if self.is_success():
            return StateStep(next_phase=PickPlacePhase.RELEASE)

        if self._trajectory is None:
            self._start_plan()

        if self._trajectory_index >= len(self._trajectory):
            if (
                self._final_waypoint_hold_steps
                < self.max_final_waypoint_hold_steps
            ):
                self._final_waypoint_hold_steps += 1
                return StateStep(action=self._trajectory[-1])
            carb.log_warn("Place trajectory exhausted before success; replanning.")
            self._trajectory = None
            self._trajectory_index = None
            self._final_waypoint_hold_steps = 0
            return StateStep()

        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return StateStep(action=action)

    def _start_plan(self) -> None:
        target_position, target_orientation = self._create_target_tool_pose()
        self._trajectory = self._planner.plan_to_pose(
            target_position,
            target_orientation,
        )
        if not self._trajectory:
            raise RuntimeError("CuRobo failed to generate a place trajectory.")
        self._trajectory_index = 0
        self._final_waypoint_hold_steps = 0

    def _create_target_cube_pose(
        self,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_region = self._world.scene.get_object("target_region")
        position, _ = target_region.get_world_pose()
        self._planned_target_position = np.asarray(position).copy()
        self._target_cube_position = position + np.array(
            [0.0, 0.0, self._cube.get_size() / 2.0]
        )
        self._target_cube_orientation = np.asarray(orientation).copy()
        self._target_cube_marker = create_xform(
            self._world,
            "/World/TargetCube",
            "target_cube",
            True,
            position=self._target_cube_position,
            orientation=self._target_cube_orientation,
        )
        return self._target_cube_position, self._target_cube_orientation

    def _create_target_tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        cube_position, cube_orientation = self._cube.get_world_pose()
        franka_position, franka_orientation = self._robot.get_world_pose()
        tool_position, tool_orientation = self._planner.get_tool_world_pose()
        axis_i, axis_j, axis_k = get_cube_canonical_axes(
            cube_position,
            cube_orientation,
            franka_position,
            franka_orientation,
        )
        canonical_basis = np.column_stack((axis_i, axis_j, axis_k))
        current_cube_rotation = pose_to_matrix(
            np.zeros(3),
            cube_orientation,
        )[:3, :3]
        franka_rotation = pose_to_matrix(
            np.zeros(3),
            franka_orientation,
        )[:3, :3]
        canonical_axes_in_cube = current_cube_rotation.T @ canonical_basis
        target_cube_transform = np.eye(4)
        target_cube_transform[:3, :3] = (
            franka_rotation @ canonical_axes_in_cube.T
        )
        _, target_cube_orientation = matrix_to_pose(
            target_cube_transform
        )
        if not np.allclose(
            target_cube_transform[:3, :3] @ canonical_axes_in_cube,
            franka_rotation,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Failed to align canonical cube axes with the Franka frame."
            )

        relative_position, relative_orientation = relative_pose(
            cube_position,
            cube_orientation,
            tool_position,
            tool_orientation,
        )
        target_cube_position, target_cube_orientation = (
            self._create_target_cube_pose(target_cube_orientation)
        )
        position, orientation = compose_poses(
            target_cube_position,
            target_cube_orientation,
            relative_position,
            relative_orientation,
        )
        self._target_tool_marker = create_xform(
            self._world,
            "/World/TargetToolCenter",
            "target_tool_center",
            True,
            position=position,
            orientation=orientation,
        )
        return position, orientation
