"""Place state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import compose_poses, create_xform, relative_pose
from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class PlaceState(PnPState):
    """Plan and execute motion that carries the cube to the target region."""

    phase = PickPlacePhase.PLACE

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
            return PickPlacePhase.WAIT_FOR_STABLE
        if perturbation.reason == "target_moved_during_place":
            return PickPlacePhase.PLACE
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Execute one place waypoint or advance to release at the target."""
        if self._trajectory is None:
            self._start_plan()

        self._trajectory_index = min(
            self._trajectory_index,
            len(self._trajectory) - 1,
        )
        action = self._trajectory[self._trajectory_index]
        self._trajectory_index += 1

        cube_position, _ = self._cube.get_world_pose()
        if self._target_cube_position is None:
            raise RuntimeError("Place target pose was not created before execution.")
        next_phase = None
        if (
            np.linalg.norm(cube_position - self._target_cube_position)
            <= self._approach_tolerance
        ):
            next_phase = PickPlacePhase.RELEASE
        return StateStep(action=action, next_phase=next_phase)

    def _start_plan(self) -> None:
        target_position, target_orientation = self._create_target_tool_pose()
        self._trajectory = self._planner.plan_to_pose(
            target_position,
            target_orientation,
        )
        if self._trajectory is None:
            raise RuntimeError("CuRobo failed to generate a place trajectory.")
        self._trajectory_index = 0

    def _create_target_cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        target_region = self._world.scene.get_object("target_region")
        position, orientation = target_region.get_world_pose()
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
        tool_position, tool_orientation = self._planner.get_tool_world_pose()
        relative_position, relative_orientation = relative_pose(
            cube_position,
            cube_orientation,
            tool_position,
            tool_orientation,
        )
        target_cube_position, target_cube_orientation = (
            self._create_target_cube_pose()
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
