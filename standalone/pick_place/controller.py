"""Hybrid CuRobo and RMPflow controller for the pick-and-place task."""

from __future__ import annotations

from enum import Enum

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
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.robot_motion.motion_generation.lula import RmpFlow

from pick_place.curobo_planner import CuroboPlanner
from pick_place.geometry import create_cube_pregrasp_frame, create_xform, pose_to_matrix


class PickPlacePhase(Enum):
    """Execution phases for one pick-and-place episode."""

    IDLE = 0
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    PLACE = 5
    RELEASE = 6
    RETURN = 7


class PnPController(BaseController):
    """Coordinate global CuRobo plans with local RMPflow manipulation."""

    _LIFT_OFFSET = 0.15

    def __init__(
        self,
        name: str,
        robot: Franka,
        cube: DynamicCuboid,
        world: World,
        physics_dt: float = 1.0 / 60.0,
        approach_tolerance: float = 0.02,
    ) -> None:
        super().__init__(name)
        self._robot = robot
        self._cube = cube
        self._world = world
        self._approach_tolerance = approach_tolerance
        self._phase = PickPlacePhase.IDLE

        rmpflow_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self._rmpflow = RmpFlow(**rmpflow_config)
        self._articulation_rmpflow = ArticulationMotionPolicy(robot, self._rmpflow, physics_dt)
        self._planner = CuroboPlanner(world.scene, robot, include_cube_in_collision=False)
        self._reset_phase_state()

    def reset(self) -> None:
        """Create a cube-local pre-grasp frame and begin the approach phase."""
        super().reset()
        self._reset_phase_state()
        self._rmpflow.set_robot_base_pose(
            robot_position=self._default_robot_position,
            robot_orientation=self._default_robot_orientation,
        )
        self._pregrasp_prim = create_cube_pregrasp_frame(self._world, self._cube, exist_ok=True)
        self._calibrate_rmpflow_tool_center_transform()
        self._phase = PickPlacePhase.APPROACH

    def forward(self) -> ArticulationAction | None:
        """Return the next articulation action for the active phase."""
        if self._phase is PickPlacePhase.IDLE:
            return None
        if self._phase is PickPlacePhase.APPROACH:
            return self._forward_approach()
        if self._phase is PickPlacePhase.DESCEND:
            return self._forward_descend()
        if self._phase is PickPlacePhase.GRASP:
            return self._forward_grasp()
        if self._phase is PickPlacePhase.LIFT:
            return self._forward_lift()
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
        self._pregrasp_prim: SingleXFormPrim | None = None
        self._default_robot_position, self._default_robot_orientation = self._robot.get_world_pose()
        base_link = self._planner.robot_config["kinematics"]["base_link"]
        self._base_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/{base_link}",
            name="franka_base_link",
        )
        self._tool_center_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/panda_hand/tool_center",
            name="franka_tool_center",
        )
        self._rmp_right_gripper_to_tool_center: np.ndarray | None = None
        self._trajectory: list[ArticulationAction] | None = None

        self._approach_started = False
        self._approach_index: int | None = None
        self._approach_target_position: np.ndarray | None = None
        self._approach_target_orientation: np.ndarray | None = None
        self._lift_started = False
        self._lift_target_position: np.ndarray | None = None
        self._lift_target_orientation: np.ndarray | None = None
        self._place_started = False
        self._place_index: int | None = None
        self._return_started = False
        self._return_index: int | None = None
        self._return_complete = False
        self._reset_arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ].copy()

    def _forward_approach(self) -> ArticulationAction:
        if not self._approach_started:
            self._approach_started = True
            self._approach_target_position, self._approach_target_orientation = self._pregrasp_prim.get_world_pose()
            transform = get_relative_transform(source_prim=self._pregrasp_prim.prim, target_prim=self._base_prim.prim)
            position, orientation = pose_from_tf_matrix(transform)
            self._trajectory = self._planner.plan_to_pose(position, orientation)
            if self._trajectory is None:
                raise RuntimeError("CuRobo failed to generate an approach trajectory.")
            self._approach_index = 0
            self._robot.gripper.open()

        self._approach_index = min(self._approach_index, len(self._trajectory) - 1)
        action = self._trajectory[self._approach_index]
        self._approach_index += 1
        tool_center_position, _ = self._tool_center_prim.get_world_pose()
        if np.linalg.norm(tool_center_position - self._approach_target_position) <= self._approach_tolerance:
            self._phase = PickPlacePhase.DESCEND
            self._trajectory = None
        return action

    def _forward_descend(self) -> ArticulationAction:
        desired_tool_transform = pose_to_matrix(
            self._cube.get_world_pose()[0],
            self._approach_target_orientation,
        )
        desired_rmp_transform = desired_tool_transform @ np.linalg.inv(self._rmp_right_gripper_to_tool_center)
        target_position, target_orientation = pose_from_tf_matrix(desired_rmp_transform)
        self._rmpflow.set_end_effector_target(
            target_position=target_position,
            target_orientation=target_orientation,
        )
        self._rmpflow.update_world()
        action = self._articulation_rmpflow.get_next_articulation_action()

        tool_center_position, _ = self._tool_center_prim.get_world_pose()
        if np.linalg.norm(tool_center_position - self._cube.get_world_pose()[0]) <= self._approach_tolerance:
            self._phase = PickPlacePhase.GRASP
        return action

    def _forward_grasp(self) -> None:
        self._robot.gripper.close()
        self._planner.attach_cube()
        self._phase = PickPlacePhase.LIFT
        return None

    def _forward_lift(self) -> ArticulationAction:
        if not self._lift_started:
            rmp_position, rmp_orientation = self._get_rmp_right_gripper_pose()
            self._lift_started = True
            self._lift_target_position = rmp_position + np.array([0.0, 0.0, self._LIFT_OFFSET])
            _, self._lift_target_orientation = pose_from_tf_matrix(pose_to_matrix(rmp_position, rmp_orientation))

        self._rmpflow.set_end_effector_target(
            target_position=self._lift_target_position,
            target_orientation=self._lift_target_orientation,
        )
        self._rmpflow.update_world()
        action = self._articulation_rmpflow.get_next_articulation_action()
        rmp_position, _ = self._get_rmp_right_gripper_pose()
        if np.linalg.norm(rmp_position - self._lift_target_position) <= self._approach_tolerance:
            self._phase = PickPlacePhase.PLACE
        return action

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
            self._phase = PickPlacePhase.RELEASE
        return action

    def _forward_release(self) -> None:
        self._robot.gripper.open()
        self._planner.detach_cube()
        self._phase = PickPlacePhase.RETURN
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
            self._phase = PickPlacePhase.IDLE
        return action

    def _calibrate_rmpflow_tool_center_transform(self) -> None:
        active_joint_indices = [
            self._robot.dof_names.index(joint_name) for joint_name in self._rmpflow.get_active_joints()
        ]
        active_joint_positions = self._robot.get_joints_state().positions[active_joint_indices]
        rmp_position, rmp_orientation = self._rmpflow.get_end_effector_pose(active_joint_positions)
        tool_position, tool_orientation = self._tool_center_prim.get_world_pose()
        self._rmp_right_gripper_to_tool_center = np.linalg.inv(
            pose_to_matrix(rmp_position, rmp_orientation)
        ) @ pose_to_matrix(tool_position, tool_orientation)

    def _get_rmp_right_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        active_joint_subset = self._articulation_rmpflow.get_active_joints_subset()
        return self._rmpflow.get_end_effector_pose(active_joint_subset.get_joint_positions())

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
