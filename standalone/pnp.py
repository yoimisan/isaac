from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import random

import carb
import numpy as np
from copy import deepcopy as dc
from omni.isaac.core.controllers import BaseController
from isaacsim.core.api import World
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from curobo.types.math import Pose

# Customize import 
import sys, os
sys.path.append(os.path.join(os.path.basename(__file__), 'utils'))

from utils.randomization import randomize_scene_layout 

usd_path = "/home/yoisan/Documents/isaac-scenes/toy.usd"

success = open_stage(usd_path)
if not success:
    raise RuntimeError(f"Failed to open stage: {usd_path}")

simulation_app.update()
while is_stage_loading():
    simulation_app.update()

from isaacsim.core.prims import SingleGeometryPrim, SingleXFormPrim
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.transformations import get_relative_transform, pose_from_tf_matrix

world = World(stage_units_in_meters=1.0)

franka = world.scene.add(Robot(
    prim_path="/World/Franka",
    name="franka"
))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/Cube",
    name="cube"
))

target_region = world.scene.add(SingleGeometryPrim(
    prim_path="/World/TargetRegion",
    name="target_region"
))

table = world.scene.add(SingleGeometryPrim(
    prim_path="/World/Table",
    name="table"
))

randomize_scene_layout(franka, cube, target_region, table)
# Above is a very elegant way of building the world. I like it.

# Here we get the pregrasp frame with respect to franka tcp frame relative to cube local frame.
def get_local_pose():
    i = np.random.randint(3)
    
    d = cube.get_size()
    offset_length = 1.25 * (np.sqrt(2) * d / 2)
    theta = np.random.random() * np.pi / 2 + np.pi/4

    if i == 0:
        local_frame_translation = [
            offset_length * np.cos(theta),
            0,
            offset_length * np.sin(theta)
        ]

        local_frame_x = [np.cos(theta - np.pi/2), 0, np.sin(theta - np.pi/2)]
        local_frame_y = [0, -1, 0]
        local_frame_z = [np.cos(theta - np.pi), 0, np.sin(theta - np.pi)]
    else:
        local_frame_translation = [
            0,
            offset_length * np.cos(theta),
            offset_length * np.sin(theta),
        ]

        if i == 1:
            local_frame_x = [0, np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)]
            local_frame_y = [1, 0, 0]
            local_frame_z = [0, np.cos(theta - np.pi), np.sin(theta - np.pi)]
        else:
            local_frame_x = [0, -np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)]
            local_frame_y = [-1, 0, 0]
            local_frame_z = [0, -np.cos(theta - np.pi), np.sin(theta - np.pi)]

    local_rotation = np.column_stack((local_frame_x, local_frame_y, local_frame_z))

    local_transform = np.eye(4, dtype=np.float32)
    local_transform[:3, :3] = local_rotation
    local_transform[:3, 3] = local_frame_translation

    local_pose = Pose.from_matrix(local_transform)
    return local_pose

local_frame_pose = get_local_pose()
pregrasp_xform = world.scene.add(SingleXFormPrim(
    prim_path="/World/Cube/cube_pregrasp",
    name="cube_pregrasp",
    translation=local_frame_pose.position.squeeze(0).detach().cpu().numpy(),
    orientation=local_frame_pose.quaternion.squeeze(0).detach().cpu().numpy()
))

# Above code has been verified
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)

from pxr import Usd, UsdGeom, Gf

def build_curobo_world_cfg(stage, robot_prim_path="/World/Franka", include_cube=False):
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

def add_tool_link_to_robot_cfg(robot_cfg: dict, stage: Usd.Stage):
    extra_tool_center_link = {}
    extra_tool_center_link['parent_link_name'] = "panda_hand"
    extra_tool_center_link['link_name'] = 'tool_center'

    tool_center: Usd.Prim = stage.GetPrimAtPath("/World/Franka/panda_hand/tool_center")
    geo_tool_center: UsdGeom.Xformable = UsdGeom.Xformable(tool_center)
    local_transformation: Gf.Matrix4d = geo_tool_center.GetLocalTransformation()

    local_translation: Gf.Vec3d = local_transformation.ExtractTranslation()
    local_orientation: Gf.Quatd = local_transformation.ExtractRotationQuat()

    local_pose: list[float] = list(local_translation) + [local_orientation.GetReal()] + list(local_orientation.GetImaginary())
    
    extra_tool_center_link['fixed_transform'] = local_pose
    extra_tool_center_link['joint_type'] = 'FIXED'
    extra_tool_center_link['joint_name'] = 'tool_link_joint'
    
    return_robot_cfg = dc(robot_cfg) 
    return_robot_cfg['kinematics']['extra_links']['tool_center'] = extra_tool_center_link
    return return_robot_cfg
class FrankaController(BaseController):
    def __init__(
        self,
        stage,
        robot_prim_path: str = "/World/Franka",
        name: str = "franka_controller",
    ):
        BaseController.__init__(self, name=name)

        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg = add_tool_link_to_robot_cfg(robot_cfg, stage)
        self.tensor_args = TensorDeviceType()
        self.world_cfg = build_curobo_world_cfg(stage, robot_prim_path, include_cube=True)
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
            interpolation_dt=0.002,
            ee_link_name='tool_center'
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        self.isaac_franka: Robot = None
        self.isaac_dof_names: list[str] = None
        self.isaac_joint_indices: list[int] = None
        self.curobo_dof_names: list[str] = self.motion_gen.joint_names
    
    def register_franka(self, isaac_franka: Robot):
        self.isaac_franka = isaac_franka
        self.isaac_dof_names = isaac_franka.dof_names
        self.isaac_joint_indices = [self.isaac_dof_names.index(name) for name in self.curobo_dof_names]

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
        action = self.curobo_joint_state_to_isaac_action(joint_state)
        self.isaac_franka.apply_action(action)
    
    def forward(self, goal_pose):
        start_state = self.get_current_joint_state()
        result = self.motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
        )
        if result.success.item():
            cmd_plan = result.get_interpolated_plan()
            return cmd_plan
        else:
            raise RuntimeError("Motion planning failed.")
    
    def get_base_frame(self):
        base_link_name = self.robot_cfg['kinematics']['base_link']
        base_link = SingleXFormPrim(
            prim_path=f"/World/Franka/{base_link_name}",
            name="franka_base_link"
        )
        return base_link





world.reset()
# import pdb; pdb.set_trace()
# simulation_app.close()
# import sys; sys.exit(0)


# Before simulation dirty work
franka_controller = FrankaController(world.stage)
franka_controller.register_franka(world.scene.get_object("franka"))

base_prim = franka_controller.get_base_frame()

transformation_matrix = get_relative_transform(
    source_prim=pregrasp_xform.prim,
    target_prim=base_prim.prim
)
xyz, quat = pose_from_tf_matrix(transformation_matrix)
goal_pose = Pose(
    position=franka_controller.tensor_args.to_device([xyz]),
    quaternion=franka_controller.tensor_args.to_device([quat])
)

# Now you can find /World/Franka, /World/Target, etc.
trajs = franka_controller.forward(goal_pose)

index = 0
length = len(trajs)

while simulation_app.is_running():
    franka_controller.apply_curobo_joint_state(trajs[index])
    index += 1

    if index >= length:
        index = length - 1


    world.step(render=True)

simulation_app.close()
