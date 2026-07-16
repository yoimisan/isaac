"""CuRobo motion-planning adapter for the Franka pick-and-place controller."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import carb
import numpy as np
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.types.state import JointState as CuRoboJointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.types import ArticulationAction, JointsState
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Usd, UsdGeom


def build_curobo_world_config(
    stage: Usd.Stage,
    robot_prim_path: str = "/World/Franka",
    include_cube: bool = False,
) -> Any:
    """Build a CuRobo collision world from the current USD stage."""
    usd_helper = UsdHelper()
    usd_helper.load_stage(stage)

    ignored_paths = [
        robot_prim_path,
        "/World/TargetRegion",
        "/World/Cube/cube_pregrasp",
        "/World/defaultGroundPlane",
        "/curobo",
    ]
    if not include_cube:
        ignored_paths.append("/World/Cube")

    return usd_helper.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=ignored_paths,
    ).get_collision_check_world()


def add_tool_link_to_robot_config(
    robot_config: dict[str, Any],
    stage: Usd.Stage,
    tool_center_path: str = "/World/Franka/panda_hand/tool_center",
    parent_link_name: str = "panda_hand",
    link_name: str = "tool_center",
) -> dict[str, Any]:
    """Add the USD tool-center frame as a fixed CuRobo kinematic link."""
    tool_center = stage.GetPrimAtPath(tool_center_path)
    if not tool_center.IsValid():
        raise RuntimeError(f"Tool center prim does not exist: {tool_center_path}")

    local_transform = UsdGeom.Xformable(tool_center).GetLocalTransformation()
    local_translation = local_transform.ExtractTranslation()
    local_orientation = local_transform.ExtractRotationQuat()
    fixed_transform = list(local_translation) + [local_orientation.GetReal()] + list(local_orientation.GetImaginary())

    updated_config = deepcopy(robot_config)
    updated_config["kinematics"].setdefault("extra_links", {})
    updated_config["kinematics"]["extra_links"][link_name] = {
        "parent_link_name": parent_link_name,
        "link_name": link_name,
        "fixed_transform": fixed_transform,
        "joint_type": "FIXED",
        "joint_name": f"{link_name}_joint",
    }
    return updated_config


class CuroboPlanner:
    """Own CuRobo configuration and convert planned trajectories to Isaac actions."""

    def __init__(
        self,
        scene: Scene,
        robot: Franka,
        include_cube_in_collision: bool = False,
    ) -> None:
        self._stage = scene.stage
        self._robot = robot
        self.tensor_args = TensorDeviceType()

        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_config = add_tool_link_to_robot_config(robot_config, self._stage)
        self.world_config = build_curobo_world_config(
            self._stage,
            robot.prim_path,
            include_cube=include_cube_in_collision,
        )

        collision_cache = self.world_config.get_cache_dict()
        collision_cache["obb"] = max(collision_cache["obb"], 32)
        collision_cache["mesh"] = max(collision_cache["mesh"], 32)
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            self.world_config,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache=collision_cache,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            interpolation_dt=0.01,
            ee_link_name="tool_center",
        )
        self._motion_gen = MotionGen(motion_gen_config)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        self.curobo_joint_names = self._motion_gen.joint_names
        self.isaac_arm_joint_indices = [
            self._robot.dof_names.index(joint_name) for joint_name in self.curobo_joint_names
        ]

    def plan_to_pose(
        self,
        base_position: np.ndarray,
        base_quaternion: np.ndarray,
    ) -> list[ArticulationAction] | None:
        """Plan to a CuRobo base-frame pose."""
        return self._plan_to_pose(base_position, base_quaternion)

    def plan_linear_approach(
        self,
        base_position: np.ndarray,
        base_quaternion: np.ndarray,
        *,
        approach_distance: float,
        linear_axis: int = 2,
        constraint_start_fraction: float = 0.8,
    ) -> list[ArticulationAction] | None:
        """Plan a final approach constrained to one axis in the goal frame."""
        if linear_axis not in (0, 1, 2):
            raise ValueError(f"linear_axis must be 0, 1, or 2; got {linear_axis}.")
        if approach_distance <= 0.0:
            raise ValueError(
                f"approach_distance must be positive; got {approach_distance}."
            )

        pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=approach_distance,
            linear_axis=linear_axis,
            tstep_fraction=constraint_start_fraction,
            project_to_goal_frame=True,
            tensor_args=self.tensor_args,
        )
        # CuRobo 0.7.8 accepts project_to_goal_frame in the helper but does not
        # copy it into the returned metric. Set it explicitly for stable semantics.
        pose_metric.project_to_goal_frame = True
        return self._plan_to_pose(
            base_position,
            base_quaternion,
            MotionGenPlanConfig(pose_cost_metric=pose_metric),
        )

    def _plan_to_pose(
        self,
        base_position: np.ndarray,
        base_quaternion: np.ndarray,
        plan_config: MotionGenPlanConfig | None = None,
    ) -> list[ArticulationAction] | None:
        """Run one pose query with an optional CuRobo planning configuration."""
        goal_pose = CuRoboPose(
            position=self.tensor_args.to_device([base_position]),
            quaternion=self.tensor_args.to_device([base_quaternion]),
        )
        start_state = self._get_current_joint_state()
        if plan_config is None:
            result = self._motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
            )
        else:
            result = self._motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=plan_config,
            )
        if not result.success.item():
            carb.log_warn("CuRobo motion plan failed.")
            return None
        return self._trajectory_to_actions(result.get_interpolated_plan())

    def plan_to_joint_positions(self, goal_positions: np.ndarray) -> list[ArticulationAction] | None:
        """Plan a collision-free joint-space trajectory."""
        start_state = self._get_current_joint_state()
        goal_state = CuRoboJointState.from_position(
            position=self.tensor_args.to_device([goal_positions]),
            joint_names=self.curobo_joint_names,
        )
        result = self._motion_gen.plan_single_js(start_state, goal_state)
        if not result.success.item():
            carb.log_warn("CuRobo failed to generate a return trajectory.")
            return None
        return self._trajectory_to_actions(result.get_interpolated_plan())

    def attach_cube(self) -> None:
        """Attach the cube to CuRobo's built-in attached-object link."""
        self._update_world(include_cube=True)
        attached = self._motion_gen.attach_objects_to_robot(
            joint_state=self._get_current_joint_state(),
            object_names=["/World/Cube"],
            link_name="attached_object",
        )
        if not attached:
            raise RuntimeError("Failed to attach the cube to CuRobo.")

    def detach_cube(self) -> None:
        """Detach the cube and restore it to the CuRobo collision world."""
        self._motion_gen.detach_object_from_robot(link_name="attached_object")
        self._update_world(include_cube=True)

    def _get_current_joint_state(self) -> CuRoboJointState:
        isaac_state: JointsState | None = self._robot.get_joints_state()
        if isaac_state is None:
            raise RuntimeError("Failed to read the Franka joint state before motion planning.")
        return CuRoboJointState.from_position(
            position=self.tensor_args.to_device([isaac_state.positions[self.isaac_arm_joint_indices]]),
            joint_names=self.curobo_joint_names,
        )

    def _trajectory_to_actions(self, trajectory: CuRoboJointState) -> list[ArticulationAction]:
        joint_positions = trajectory.get_ordered_joint_state(self.curobo_joint_names).position
        if joint_positions.ndim == 3:
            if joint_positions.shape[0] != 1:
                raise RuntimeError("Expected one CuRobo trajectory batch for this Franka.")
            joint_positions = joint_positions.squeeze(0)
        if joint_positions.ndim != 2:
            raise RuntimeError(f"Unexpected CuRobo trajectory shape: {tuple(joint_positions.shape)}")
        return [
            ArticulationAction(
                joint_positions=waypoint.detach().cpu().numpy(),
                joint_indices=self.isaac_arm_joint_indices,
            )
            for waypoint in joint_positions
        ]

    def _update_world(self, include_cube: bool) -> None:
        self.world_config = build_curobo_world_config(
            self._stage,
            self._robot.prim_path,
            include_cube=include_cube,
        )
        self._motion_gen.update_world(self.world_config)
