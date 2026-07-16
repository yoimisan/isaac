"""Place state for the pick-and-place controller."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    get_world_pose_from_relative,
    pose_from_tf_matrix,
)
from isaacsim.core.utils.types import ArticulationAction

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import create_xform
from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class PlaceState(PnPState):
    """Plan and execute motion that carries the cube to the target region."""

    phase = PickPlacePhase.PLACE

    def __init__(
        self,
        *,
        world: World,
        cube: DynamicCuboid,
        planner: CuroboPlanner,
        base_prim: SingleXFormPrim,
        tool_center_prim: SingleXFormPrim,
        approach_tolerance: float,
    ) -> None:
        self._world = world
        self._cube = cube
        self._planner = planner
        self._base_prim = base_prim
        self._tool_center_prim = tool_center_prim
        self._approach_tolerance = approach_tolerance

        self._trajectory: list[ArticulationAction] | None = None
        self._trajectory_index: int | None = None
        self._target_cube_prim: SingleXFormPrim | None = None

    def enter(self) -> None:
        """Discard the previous place plan and target frame."""
        self._trajectory = None
        self._trajectory_index = None
        self._target_cube_prim = None

    def exit(self) -> None:
        """Drop trajectory data that must not cross the state boundary."""
        self._trajectory = None
        self._trajectory_index = None

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
        target_cube_position, _ = self._target_cube_prim.get_world_pose()
        next_phase = None
        if (
            np.linalg.norm(cube_position - target_cube_position)
            <= self._approach_tolerance
        ):
            next_phase = PickPlacePhase.RELEASE
        return StateStep(action=action, next_phase=next_phase)

    def _start_plan(self) -> None:
        target_tool_center_prim = self._create_target_tool_center_prim()
        local_position, local_orientation = pose_from_tf_matrix(
            get_relative_transform(
                source_prim=target_tool_center_prim.prim,
                target_prim=self._base_prim.prim,
            )
        )
        self._trajectory = self._planner.plan_to_pose(
            local_position,
            local_orientation,
        )
        if self._trajectory is None:
            raise RuntimeError("CuRobo failed to generate a place trajectory.")
        self._trajectory_index = 0

    def _create_target_cube_prim(self) -> SingleXFormPrim:
        target_region = self._world.scene.get_object("target_region")
        position, orientation = target_region.get_world_pose()
        target_position = position + np.array(
            [0.0, 0.0, self._cube.get_size() / 2.0]
        )
        return create_xform(
            self._world,
            "/World/TargetCube",
            "target_cube",
            True,
            position=target_position,
            orientation=orientation,
        )

    def _create_target_tool_center_prim(self) -> SingleXFormPrim:
        relative_position, relative_orientation = pose_from_tf_matrix(
            get_relative_transform(
                source_prim=self._tool_center_prim.prim,
                target_prim=self._cube.prim,
            )
        )
        self._target_cube_prim = self._create_target_cube_prim()
        position, orientation = get_world_pose_from_relative(
            coord_prim=self._target_cube_prim.prim,
            relative_translation=relative_position,
            relative_orientation=relative_orientation,
        )
        return create_xform(
            self._world,
            "/World/TargetToolCenter",
            "target_tool_center",
            True,
            position=position,
            orientation=orientation,
        )
