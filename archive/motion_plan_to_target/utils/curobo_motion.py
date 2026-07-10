from __future__ import annotations

from copy import deepcopy

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.base_controller import BaseController
from pxr import Usd, UsdGeom


def build_curobo_world_cfg(
    stage: Usd.Stage,
    robot_prim_path: str = "/World/Franka",
    include_cube: bool = False,
):
    usd_helper = UsdHelper()
    usd_helper.load_stage(stage)

    ignore_substring = [
        robot_prim_path,
        "/World/TargetRegion",
        "/World/Cube/cube_pregrasp",
        "/World/defaultGroundPlane",
        "/curobo",
    ]
    if not include_cube:
        ignore_substring.append("/World/Cube")

    return usd_helper.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=ignore_substring,
    ).get_collision_check_world()


def add_tool_link_to_robot_cfg(
    robot_cfg: dict,
    stage: Usd.Stage,
    tool_center_path: str = "/World/Franka/panda_hand/tool_center",
    parent_link_name: str = "panda_hand",
    link_name: str = "tool_center",
) -> dict:
    tool_center: Usd.Prim = stage.GetPrimAtPath(tool_center_path)
    if not tool_center.IsValid():
        raise RuntimeError(f"Tool center prim does not exist: {tool_center_path}")

    tool_center_xform = UsdGeom.Xformable(tool_center)
    local_transform = tool_center_xform.GetLocalTransformation()
    local_translation = local_transform.ExtractTranslation()
    local_orientation = local_transform.ExtractRotationQuat()

    fixed_transform = (
        list(local_translation)
        + [local_orientation.GetReal()]
        + list(local_orientation.GetImaginary())
    )

    return_robot_cfg = deepcopy(robot_cfg)
    return_robot_cfg["kinematics"].setdefault("extra_links", {})
    return_robot_cfg["kinematics"]["extra_links"][link_name] = {
        "parent_link_name": parent_link_name,
        "link_name": link_name,
        "fixed_transform": fixed_transform,
        "joint_type": "FIXED",
        "joint_name": f"{link_name}_joint",
    }
    return return_robot_cfg


def build_goal_pose_from_frame(source_prim, target_prim, tensor_args: TensorDeviceType) -> Pose:
    transformation_matrix = get_relative_transform(
        source_prim=source_prim,
        target_prim=target_prim,
    )
    xyz, quat = pose_from_tf_matrix(transformation_matrix)
    return Pose(
        position=tensor_args.to_device([xyz]),
        quaternion=tensor_args.to_device([quat]),
    )


class FrankaController(BaseController):
    def __init__(
        self,
        stage: Usd.Stage,
        robot_prim_path: str = "/World/Franka",
        name: str = "franka_controller",
        include_cube_in_collision: bool = True,
    ):
        BaseController.__init__(self, name=name)

        self.robot_prim_path = robot_prim_path
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg = add_tool_link_to_robot_cfg(robot_cfg, stage)
        self.tensor_args = TensorDeviceType()
        self.world_cfg = build_curobo_world_cfg(
            stage,
            robot_prim_path,
            include_cube=include_cube_in_collision,
        )
        collision_cache = self.world_cfg.get_cache_dict()
        collision_cache["obb"] = max(collision_cache["obb"], 32)
        collision_cache["mesh"] = max(collision_cache["mesh"], 32)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache=collision_cache,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            interpolation_dt=0.01,
            ee_link_name="tool_center",
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        self.isaac_franka: Robot | None = None
        self.isaac_dof_names: list[str] | None = None
        self.isaac_joint_indices: list[int] | None = None
        self.curobo_dof_names: list[str] = self.motion_gen.joint_names

    def register_franka(self, isaac_franka: Robot):
        self.isaac_franka = isaac_franka
        self.isaac_dof_names = isaac_franka.dof_names
        self.isaac_joint_indices = [
            self.isaac_dof_names.index(name) for name in self.curobo_dof_names
        ]

    def _get_isaac_joint_indices(self) -> list[int]:
        if self.isaac_franka is None:
            raise RuntimeError("Isaac Franka is not registered. Call register_franka(franka) first.")
        if self.isaac_dof_names is None:
            self.isaac_dof_names = self.isaac_franka.dof_names
        if self.isaac_joint_indices is None:
            self.isaac_joint_indices = [
                self.isaac_dof_names.index(name) for name in self.curobo_dof_names
            ]
        return self.isaac_joint_indices

    def get_current_joint_state(self) -> JointState:
        if self.isaac_franka is None:
            raise RuntimeError("Isaac Franka is not registered. Call register_franka(franka) first.")

        isaac_joint_state = self.isaac_franka.get_joints_state()
        if isaac_joint_state is None:
            raise RuntimeError("Failed to read Isaac Franka joint state.")

        curobo_position = isaac_joint_state.positions[self._get_isaac_joint_indices()]

        return JointState.from_position(
            self.tensor_args.to_device([curobo_position]),
            joint_names=self.curobo_dof_names,
        )

    def _joint_tensor_to_numpy(self, joint_tensor):
        if joint_tensor is None:
            return None
        if len(joint_tensor.shape) == 2:
            if joint_tensor.shape[0] != 1:
                raise ValueError(
                    "Expected a single JointState command. Index the trajectory first, e.g. cmd_plan[i]."
                )
            joint_tensor = joint_tensor.squeeze(0)
        return joint_tensor.detach().cpu().numpy()

    def curobo_joint_state_to_isaac_action(
        self,
        joint_state: JointState,
        include_velocity: bool = False,
    ) -> ArticulationAction:
        if joint_state.joint_names is not None and joint_state.joint_names != self.curobo_dof_names:
            joint_state = joint_state.get_ordered_joint_state(self.curobo_dof_names)

        return ArticulationAction(
            joint_positions=self._joint_tensor_to_numpy(joint_state.position),
            joint_velocities=(
                self._joint_tensor_to_numpy(joint_state.velocity) if include_velocity else None
            ),
            joint_indices=self._get_isaac_joint_indices(),
        )

    def apply_curobo_joint_state(self, joint_state: JointState):
        if self.isaac_franka is None:
            raise RuntimeError("Isaac Franka is not registered. Call register_franka(franka) first.")

        action = self.curobo_joint_state_to_isaac_action(joint_state)
        self.isaac_franka.apply_action(action)

    def forward(self, goal_pose: Pose):
        start_state = self.get_current_joint_state()
        result = self.motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
        )
        if result.success.item():
            return result.get_interpolated_plan()
        raise RuntimeError("Motion planning failed.")

    def get_base_frame(self):
        base_link_name = self.robot_cfg["kinematics"]["base_link"]
        return SingleXFormPrim(
            prim_path=f"{self.robot_prim_path}/{base_link_name}",
            name="franka_base_link",
        )
