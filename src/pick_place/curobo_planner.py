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

from pick_place.curobo_world import CuroboWorldRegistry, RigidPoseSource


def build_curobo_world_config(
    stage: Usd.Stage,
    robot_prim_path: str = "/World/Franka",
) -> Any:
    """Build the initial CuRobo collision geometry from the USD stage."""
    usd_helper = UsdHelper()
    usd_helper.load_stage(stage)

    ignored_paths = [
        robot_prim_path,
        "/World/TargetRegion",
        "/World/CubePregrasp",
        "/World/TargetCube",
        "/World/TargetToolCenter",
        "/World/defaultGroundPlane",
        "/curobo",
    ]

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
    fixed_transform = [
        *[float(value) for value in local_translation],
        float(local_orientation.GetReal()),
        *[float(value) for value in local_orientation.GetImaginary()],
    ]

    updated_config = deepcopy(robot_config)
    updated_config["kinematics"].setdefault("extra_links", {})
    updated_config["kinematics"]["extra_links"][link_name] = {
        "parent_link_name": parent_link_name,
        "link_name": link_name,
        "fixed_transform": fixed_transform,
        "joint_type": "FIXED",
        "joint_name": f"{link_name}_joint",
    }
    attached_object = updated_config["kinematics"]["extra_links"].get(
        "attached_object"
    )
    if attached_object is None:
        raise RuntimeError("Franka CuRobo config has no attached_object link.")
    attached_object["parent_link_name"] = parent_link_name
    attached_object["fixed_transform"] = fixed_transform.copy()
    return updated_config


class CuroboPlanner:
    """Own CuRobo configuration and convert planned trajectories to Isaac actions."""

    def __init__(
        self,
        scene: Scene,
        robot: Franka,
    ) -> None:
        self._stage = scene.stage
        self._robot = robot
        self.tensor_args = TensorDeviceType()

        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_config = add_tool_link_to_robot_config(robot_config, self._stage)
        self.world_config = build_curobo_world_config(
            self._stage,
            robot.prim_path,
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
        self._world_registry = CuroboWorldRegistry(
            motion_gen=self._motion_gen,
            robot=self._robot,
            tensor_args=self.tensor_args,
        )

        self.curobo_joint_names = self._motion_gen.joint_names
        self.isaac_arm_joint_indices = [
            self._robot.dof_names.index(joint_name) for joint_name in self.curobo_joint_names
        ]

    def register_dynamic_obstacle(
        self,
        pose_source: RigidPoseSource,
        *,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Register a PhysX-backed obstacle already present in the CuRobo world."""
        self._world_registry.register(
            name=pose_source.prim_path if name is None else name,
            pose_source=pose_source,
            enabled=enabled,
        )

    def set_obstacle_collision_enabled(
        self,
        name: str,
        enabled: bool,
    ) -> None:
        """Synchronize an obstacle and set its CuRobo collision state."""
        self._world_registry.sync(name)
        self._world_registry.set_enabled(name, enabled)

    def reset_episode(self, attached_object_name: str = "/World/Cube") -> None:
        """Clear a stale attachment and restore the manipulated object as an obstacle."""
        self._motion_gen.detach_object_from_robot(link_name="attached_object")
        self._world_registry.sync(attached_object_name)
        self._world_registry.set_enabled(attached_object_name, True)

    def get_tool_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the live CuRobo tool pose from the current PhysX joint state."""
        kinematic_state = self._motion_gen.compute_kinematics(
            self._get_current_joint_state()
        )
        world_tool = self._get_world_base_pose().multiply(kinematic_state.ee_pose)
        return (
            world_tool.position.squeeze(0).detach().cpu().numpy(),
            world_tool.quaternion.squeeze(0).detach().cpu().numpy(),
        )

    def plan_to_pose(
        self,
        world_position: np.ndarray,
        world_quaternion: np.ndarray,
    ) -> list[ArticulationAction] | None:
        """Plan to a world-frame pose."""
        return self._plan_to_pose(world_position, world_quaternion)

    def plan_linear_approach(
        self,
        world_position: np.ndarray,
        world_quaternion: np.ndarray,
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
            world_position,
            world_quaternion,
            MotionGenPlanConfig(pose_cost_metric=pose_metric),
        )

    def plan_linear_motion(
        self,
        world_position: np.ndarray,
        world_quaternion: np.ndarray,
        *,
        linear_axis: int,
        project_to_goal_frame: bool,
    ) -> list[ArticulationAction] | None:
        """Plan while holding every pose dimension except one translation axis."""
        if linear_axis not in (0, 1, 2):
            raise ValueError(f"linear_axis must be 0, 1, or 2; got {linear_axis}.")

        hold_weight = self.tensor_args.to_device(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        hold_weight[3 + linear_axis] = 0.0
        pose_metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=hold_weight,
            project_to_goal_frame=project_to_goal_frame,
        )
        return self._plan_to_pose(
            world_position,
            world_quaternion,
            MotionGenPlanConfig(pose_cost_metric=pose_metric),
        )

    def _plan_to_pose(
        self,
        world_position: np.ndarray,
        world_quaternion: np.ndarray,
        plan_config: MotionGenPlanConfig | None = None,
    ) -> list[ArticulationAction] | None:
        """Synchronize the world and run one world-frame pose query."""
        self._world_registry.sync_enabled()
        goal_pose = self._world_to_base_pose(world_position, world_quaternion)
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
        self._world_registry.sync_enabled()
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
        self._world_registry.sync("/World/Cube")
        attached = self._motion_gen.attach_objects_to_robot(
            joint_state=self._get_current_joint_state(),
            object_names=["/World/Cube"],
            link_name="attached_object",
        )
        if not attached:
            raise RuntimeError("Failed to attach the cube to CuRobo.")
        self._world_registry.set_enabled("/World/Cube", False)

    def detach_cube(self) -> None:
        """Detach the cube and restore it to the CuRobo collision world."""
        self._motion_gen.detach_object_from_robot(link_name="attached_object")
        self._world_registry.sync("/World/Cube")
        self._world_registry.set_enabled("/World/Cube", True)

    def _get_world_base_pose(self) -> CuRoboPose:
        position, orientation = self._robot.get_world_pose()
        return self._make_pose(position, orientation)

    def _world_to_base_pose(
        self,
        world_position: np.ndarray,
        world_quaternion: np.ndarray,
    ) -> CuRoboPose:
        world_goal = self._make_pose(world_position, world_quaternion)
        return self._get_world_base_pose().inverse().multiply(world_goal)

    def _make_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> CuRoboPose:
        return CuRoboPose(
            position=self.tensor_args.to_device(
                [np.asarray(position, dtype=np.float32)]
            ),
            quaternion=self.tensor_args.to_device(
                [np.asarray(orientation, dtype=np.float32)]
            ),
        )

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
