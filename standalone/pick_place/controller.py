"""CuRobo-backed state-machine controller for pick-and-place."""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    get_world_pose_from_relative,
    pose_from_tf_matrix,
)
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import create_xform
from pick_place.states import (
    ApproachState,
    DescendState,
    LiftState,
    PickPlacePhase,
    PnPState,
    WaitForStableState,
)


class PnPController(BaseController):
    """Execute pick-and-place as a state machine backed by CuRobo plans."""

    _LIFT_OFFSET = 0.15

    def __init__(
        self,
        name: str,
        robot: Franka,
        cube: DynamicCuboid,
        world: World,
        approach_tolerance: float = 0.02,
    ) -> None:
        super().__init__(name)
        self._robot = robot
        self._cube = cube
        self._world = world
        self._approach_tolerance = approach_tolerance
        self._phase = PickPlacePhase.IDLE

        self._planner = CuroboPlanner(world.scene, robot, include_cube_in_collision=False)
        self._reset_phase_state()

    def reset(self) -> None:
        """Reset state-local data and begin a fresh approach."""
        super().reset()
        self._reset_phase_state()
        self._transition_to(PickPlacePhase.APPROACH)

    def forward(self) -> ArticulationAction | None:
        """Return the next articulation action for the active phase."""
        if self._phase is PickPlacePhase.IDLE:
            return None
        if self._phase in self._state_objects:
            return self._forward_state_object()
        if self._phase is PickPlacePhase.GRASP:
            return self._forward_grasp()
        if self._phase is PickPlacePhase.PLACE:
            return self._forward_place()
        if self._phase is PickPlacePhase.RELEASE:
            return self._forward_release()
        if self._phase is PickPlacePhase.RETURN:
            return self._forward_return()
        raise RuntimeError(f"Unsupported pick-and-place phase: {self._phase}")

    def is_complete(self) -> bool:
        """Return whether the cube was released and the arm returned to its reset pose."""
        return self._return_complete

    def _reset_phase_state(self) -> None:
        self._phase = PickPlacePhase.IDLE
        base_link = self._planner.robot_config["kinematics"]["base_link"]
        self._base_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/{base_link}",
            name="franka_base_link",
        )
        self._tool_center_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/panda_hand/tool_center",
            name="franka_tool_center",
        )
        self._trajectory: list[ArticulationAction] | None = None

        self._place_started = False
        self._place_index: int | None = None
        self._return_started = False
        self._return_index: int | None = None
        self._return_complete = False
        self._reset_arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ].copy()
        self._approach_state = ApproachState(
            world=self._world,
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            approach_tolerance=self._approach_tolerance,
        )
        wait_for_stable_state = WaitForStableState(
            robot=self._robot,
            cube=self._cube,
            arm_joint_indices=self._planner.isaac_arm_joint_indices,
        )
        descend_state = DescendState(
            cube=self._cube,
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            approach_tolerance=self._approach_tolerance,
        )
        lift_state = LiftState(
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            lift_offset=self._LIFT_OFFSET,
            approach_tolerance=self._approach_tolerance,
        )
        self._state_objects: dict[PickPlacePhase, PnPState] = {
            self._approach_state.phase: self._approach_state,
            descend_state.phase: descend_state,
            lift_state.phase: lift_state,
            wait_for_stable_state.phase: wait_for_stable_state,
        }

    def _forward_state_object(self) -> ArticulationAction | None:
        """Validate and advance the active state object by one tick."""
        state = self._state_objects[self._phase]
        perturbation = state.detect_perturbation()
        if perturbation is not None:
            recovery_phase = state.recovery_phase(perturbation)
            carb.log_warn(
                f"PnP state {self._phase.name} invalidated: "
                f"{perturbation.reason}; metrics={dict(perturbation.metrics)}"
            )
            self._transition_to(recovery_phase)
            return None

        step = state.update()
        if step.next_phase is not None:
            self._transition_to(step.next_phase)
        return step.action

    def _transition_to(self, next_phase: PickPlacePhase) -> None:
        """Perform state lifecycle hooks and record one state transition."""
        previous_phase = self._phase
        previous_state = self._state_objects.get(previous_phase)
        if previous_state is not None:
            previous_state.exit()

        self._phase = next_phase
        next_state = self._state_objects.get(next_phase)
        if next_state is not None:
            next_state.enter()

        if previous_phase is not next_phase:
            carb.log_info(
                f"PnP phase transition: {previous_phase.name} -> {next_phase.name}"
            )

    def _forward_grasp(self) -> None:
        self._robot.gripper.close()
        self._planner.attach_cube()
        self._transition_to(PickPlacePhase.LIFT)
        return None

    def _forward_place(self) -> ArticulationAction:
        if not self._place_started:
            target_prim = self._create_target_tool_center_prim()
            local_position, local_orientation = pose_from_tf_matrix(
                get_relative_transform(source_prim=target_prim.prim, target_prim=self._base_prim.prim)
            )
            self._trajectory = self._planner.plan_to_pose(local_position, local_orientation)
            if self._trajectory is None:
                raise RuntimeError("CuRobo failed to generate a place trajectory.")
            self._place_index = 0
            self._place_started = True

        self._place_index = min(self._place_index, len(self._trajectory) - 1)
        action = self._trajectory[self._place_index]
        self._place_index += 1
        cube_position, _ = self._cube.get_world_pose()
        target_cube_position, _ = self._world.scene.get_object("target_cube").get_world_pose()
        if np.linalg.norm(cube_position - target_cube_position) <= self._approach_tolerance:
            self._transition_to(PickPlacePhase.RELEASE)
        return action

    def _forward_release(self) -> None:
        self._robot.gripper.open()
        self._planner.detach_cube()
        self._transition_to(PickPlacePhase.RETURN)
        return None

    def _forward_return(self) -> ArticulationAction:
        if not self._return_started:
            self._trajectory = self._planner.plan_to_joint_positions(self._reset_arm_positions)
            if self._trajectory is None:
                raise RuntimeError("CuRobo failed to generate a return trajectory.")
            self._return_index = 0
            self._return_started = True

        self._return_index = min(self._return_index, len(self._trajectory) - 1)
        action = self._trajectory[self._return_index]
        self._return_index += 1
        arm_positions = self._robot.get_joints_state().positions[self._planner.isaac_arm_joint_indices]
        if np.linalg.norm(arm_positions - self._reset_arm_positions) <= self._approach_tolerance:
            self._return_complete = True
            self._transition_to(PickPlacePhase.IDLE)
        return action

    def _create_target_cube_prim(self) -> SingleXFormPrim:
        position, orientation = self._world.scene.get_object("target_region").get_world_pose()
        target_position = position + np.array([0.0, 0.0, self._cube.get_size() / 2.0])
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
            get_relative_transform(source_prim=self._tool_center_prim.prim, target_prim=self._cube.prim)
        )
        target_cube_prim = self._create_target_cube_prim()
        position, orientation = get_world_pose_from_relative(
            coord_prim=target_cube_prim.prim,
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
