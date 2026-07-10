from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from copy import deepcopy 
import sys
from pathlib import Path
from typing import Optional
from pxr import Gf, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.types import JointsState, ArticulationAction
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from isaacsim.core.utils.extensions import enable_extension

# Keep this script on the legacy World/BaseTask API.  The matching Franka
# wrapper is provided by this extension rather than by the experimental API.
enable_extension("isaacsim.robot.manipulators.examples")
from isaacsim.robot.manipulators.examples.franka import Franka

# RMP Flow Related Import
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)
from isaacsim.robot_motion.motion_generation.lula import RmpFlow

# Motion Gen Related Import
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.state import JointState as CuRoboJointState
from curobo.types.math import Pose as CuRoboPose
from curobo.util.usd_helper import UsdHelper

from pxr import Usd, UsdGeom
from enum import Enum

def sample_cube_pregrasp_pose(cube) -> CuRoboPose:
    face_index = np.random.randint(2)

    cube_size = cube.get_size()
    offset_length = 1.5 * (np.sqrt(2.0) * cube_size / 2.0)
    theta = np.random.random() * np.pi / 4.0 + np.pi / 3.0

    if face_index == 0:
        local_frame_translation = [
            offset_length * np.cos(theta),
            0.0,
            offset_length * np.sin(theta),
        ]

        local_frame_x = [np.cos(theta - np.pi / 2.0), 0.0, np.sin(theta - np.pi / 2.0)]
        local_frame_y = [0.0, -1.0, 0.0]
        local_frame_z = [np.cos(theta - np.pi), 0.0, np.sin(theta - np.pi)]
    else:
        local_frame_translation = [
            0.0,
            offset_length * np.cos(theta),
            offset_length * np.sin(theta),
        ]

        if theta < np.pi / 2:
            local_frame_x = [0.0, -np.sin(theta), np.cos(theta)]
            local_frame_y = [-1.0, 0.0, 0.0]
            local_frame_z = [0.0, -np.cos(theta), -np.sin(theta)]
        else:
            local_frame_x = [0.0, np.sin(np.pi - theta), np.cos(np.pi - theta)]
            local_frame_y = [1.0, 0.0, 0.0]
            local_frame_z = [0.0, np.cos(np.pi - theta), -np.sin(np.pi - theta)]

    local_rotation = np.column_stack((local_frame_x, local_frame_y, local_frame_z))

    local_transform = np.eye(4, dtype=np.float32)
    local_transform[:3, :3] = local_rotation
    local_transform[:3, 3] = local_frame_translation

    return CuRoboPose.from_matrix(local_transform)


def create_cube_pregrasp_frame(
    world,
    cube,
    prim_path: str = "/World/Cube/cube_pregrasp",
    name: str = "cube_pregrasp",
    exist_ok: bool = False
) -> SingleXFormPrim:
    local_frame_pose = sample_cube_pregrasp_pose(cube)
    loginfo: str = f"Create local frame pose: {str(local_frame_pose.to_list())}."
    carb.log_info(loginfo)

    if world.scene.object_exists(name) and exist_ok:
        world.scene.remove_object(name)

    return world.scene.add(SingleXFormPrim(
        prim_path=prim_path,
        name=name,
        translation=local_frame_pose.position.squeeze(0).detach().cpu().numpy(),
        orientation=local_frame_pose.quaternion.squeeze(0).detach().cpu().numpy(),
    ))

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


def pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Convert a world pose with wxyz quaternion or 3x3 rotation orientation to a 4x4 matrix."""
    orientation = np.asarray(orientation)
    transform = np.eye(4)
    if orientation.shape == (3, 3):
        transform[:3, :3] = orientation
    elif orientation.shape == (4,):
        w, x, y, z = orientation
        transform[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
    else:
        raise ValueError(f"Unsupported orientation shape: {orientation.shape}")
    transform[:3, 3] = position
    return transform

class Status(Enum):
    HANG=0 # hanging (nothing to do)
    APPROACH=1 # Approach to pregrasp pose
    DEEP=2 # pregrasp to deep
    GRASP=3 # grasp (gripper close)
    PLACE=4 # place to target

class PnPController(BaseController):

    def __init__(
        self,
        name: str,
        robot: Franka,
        cube: DynamicCuboid,
        world: World,
        physics_dt: float = 1.0 / 60.0,
        approach_tolerance: float = 0.01,
    ):
        BaseController.__init__(self, name)
        self._robot = robot
        self._cube = cube
        self._world = world
        self._approach_tolerance = approach_tolerance
        self._status = Status.HANG

        # RMP Flow Initilization
        rmpflow_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self._rmpflow = RmpFlow(**rmpflow_config)
        self._articulation_rmpflow = ArticulationMotionPolicy(robot, self._rmpflow, physics_dt)

        # Motion Planning Initialization
        self.setup_curobo_motion_gen(world.scene, self._robot.prim_path, False)

        self.reset_buffer_variables()

    def reset_buffer_variables(self):
        # Set up variables that may be used in each status.
        self._pregrasp_prim = None
        self._default_robot_position, self._default_robot_orientation = self._robot.get_world_pose()
        self._base_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/{self._robot_cfg["kinematics"]["base_link"]}",
            name="franka_base_link"
        )
        self._tool_center_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/panda_hand/tool_center",
            name="franka_tool_center"
        )
        self._rmp_right_gripper_to_tool_center = None

        self._approach_start = False
        self._buffer_trajs: list[ArticulationAction] = None
        self._approach_idx = None

    
    def setup_curobo_motion_gen(self, scene: Scene, robot_prim_path: str, include_cube_in_collision: bool=True):
        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))['robot_cfg']
        robot_cfg = add_tool_link_to_robot_cfg(robot_cfg, scene.stage)
        world_cfg = build_curobo_world_cfg(scene.stage, robot_prim_path, include_cube=include_cube_in_collision)

        collision_cache = world_cfg.get_cache_dict()
        collision_cache["obb"] = max(collision_cache["obb"], 32)
        collision_cache["mesh"] = max(collision_cache["mesh"], 32)

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache=collision_cache,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            interpolation_dt=0.01,
            ee_link_name="tool_center",
        )
        self._motion_gen = MotionGen(self.motion_gen_config)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        self._curobo_joint_names = self._motion_gen.joint_names
        self._isaac_arm_joint_indices = [
            self._robot.dof_names.index(joint_name) for joint_name in self._curobo_joint_names
        ]

        self._robot_cfg = robot_cfg
        self._world_cfg = world_cfg
    
    def motion_plan_to_pregrasp(self) -> list[ArticulationAction] | None:
        transformation_matrix = get_relative_transform(
            source_prim=self._pregrasp_prim.prim,
            target_prim=self._base_prim.prim
        )
        xyz, quat = pose_from_tf_matrix(transformation_matrix)
        return self.motion_plan(
            base_position=xyz,
            base_quaternion=quat
        )
    def motion_plan(self, base_position, base_quaternion) -> list[ArticulationAction] | None:
        """Plan to a CuRobo base-frame pose and convert the trajectory to Isaac actions."""
        goal_pose: CuRoboPose = CuRoboPose(
            position=self.tensor_args.to_device([base_position]),
            quaternion=self.tensor_args.to_device([base_quaternion]),
        )

        start_state: JointsState = self._robot.get_joints_state()
        if start_state is None:
            raise RuntimeError("Failed to read the Franka joint state before motion planning.")

        curobo_start_state: CuRoboJointState = CuRoboJointState.from_position(
            position=self.tensor_args.to_device([start_state.positions[self._isaac_arm_joint_indices]]),
            joint_names=self._curobo_joint_names,
        )

        result = self._motion_gen.plan_single(
            start_state=curobo_start_state,
            goal_pose=goal_pose
        )

        if not result.success.item():
            carb.log_warn("CuRobo motion plan failed.")
            return None

        trajectory: CuRoboJointState = result.get_interpolated_plan()
        joint_positions = trajectory.get_ordered_joint_state(self._curobo_joint_names).position
        if joint_positions.ndim == 3:
            if joint_positions.shape[0] != 1:
                raise RuntimeError("Expected one CuRobo trajectory batch for this single Franka.")
            joint_positions = joint_positions.squeeze(0)
        if joint_positions.ndim != 2:
            raise RuntimeError(f"Unexpected CuRobo trajectory shape: {tuple(joint_positions.shape)}")

        return [
            ArticulationAction(
                joint_positions=waypoint.detach().cpu().numpy(),
                joint_indices=self._isaac_arm_joint_indices,
            )
            for waypoint in joint_positions
        ]

    def calibrate_rmpflow_tool_center_transform(self) -> None:
        """Measure the fixed RMPflow right_gripper-to-USD-tool_center transform."""
        active_joint_indices = [
            self._robot.dof_names.index(joint_name) for joint_name in self._rmpflow.get_active_joints()
        ]
        active_joint_positions = self._robot.get_joints_state().positions[active_joint_indices]
        rmp_position, rmp_orientation = self._rmpflow.get_end_effector_pose(active_joint_positions)
        tool_position, tool_orientation = self._tool_center_prim.get_world_pose()

        world_to_rmp = pose_to_matrix(rmp_position, rmp_orientation)
        world_to_tool = pose_to_matrix(tool_position, tool_orientation)
        self._rmp_right_gripper_to_tool_center = np.linalg.inv(world_to_rmp) @ world_to_tool
    
    def attach_cube_to_curobo_world(self):
        world_cfg = build_curobo_world_cfg(
            self._world.stage,
            self._robot.prim_path,
            include_cube=True
        ) 
        self._motion_gen.update_world(world_cfg)

        isaac_state = self._robot.get_joints_state()
        curobo_state = CuRoboJointState.from_position(
            position=self.tensor_args.to_device(
                [isaac_state.positions[self._isaac_arm_joint_indices]]
            ),
            joint_names=self._curobo_joint_names,
        )

        # 3. Convert the cube from a world obstacle into collision spheres attached
        #    to CuRobo's built-in `attached_object` link under `panda_hand`.
        attached = self._motion_gen.attach_objects_to_robot(
            joint_state=curobo_state,
            object_names=["/World/Cube"],
            link_name="attached_object",
        )
        if not attached:
            raise RuntimeError("Failed to attach the cube to CuRobo.")
    def deattach_cube_from_curobo_world(self):
        self._motion_gen.detach_object_from_robot(link_name="attached_object")
        world_cfg = build_curobo_world_cfg(
            self._world.stage,
            self._robot.prim_path,
            include_cube=True
        ) 
        self._motion_gen.update_world(world_cfg)


    def reset(self):
        """Create a cube-local pre-grasp frame and begin reactive approach control."""
        BaseController.reset(self)
        self.reset_buffer_variables()
        self._rmpflow.set_robot_base_pose(
            robot_position=self._default_robot_position,
            robot_orientation=self._default_robot_orientation,
        )
        self._pregrasp_prim: SingleXFormPrim = create_cube_pregrasp_frame(
            self._world,
            self._cube,
            exist_ok=True,
        )
        self.calibrate_rmpflow_tool_center_transform()
        self._status = Status.APPROACH


    def forward(self):
        """Return the next RMPflow action for the active pick-and-place phase."""
        if self._status == Status.HANG:
            return None
        action = None

        if self._status == Status.APPROACH:
            if not self._approach_start:
                self._approach_start = True
                self._approach_target_position, self._approach_target_orientation = self._pregrasp_prim.get_world_pose()
                self._buffer_trajs = self.motion_plan_to_pregrasp()
                if self._buffer_trajs is None:
                    raise RuntimeError("CuRobo failed to generate an approach trajectory.")

                self._approach_idx = 0
                self._robot.gripper.open()
            
            if self._approach_idx >= len(self._buffer_trajs): 
                self._approach_idx = len(self._buffer_trajs) - 1
            action = self._buffer_trajs[self._approach_idx]
            self._approach_idx += 1
                
            # # The frame is a child of the dynamic cube, so its world pose changes
            # # immediately with the cube and becomes RMPflow's target next step.
            # self._rmpflow.set_end_effector_target(
            #     target_position=target_position,
            #     target_orientation=target_orientation,
            # )
            # self._rmpflow.update_world()
            # action = self._articulation_rmpflow.get_next_articulation_action()
            # self._robot._gripper.open()



            tool_center_position, _ = self._tool_center_prim.get_world_pose()
            if np.linalg.norm(tool_center_position - self._approach_target_position) <= self._approach_tolerance:
                self._status = Status.DEEP
        
        elif self._status == Status.DEEP:
            desired_tool_transform = pose_to_matrix(
                self._cube.get_world_pose()[0],
                self._approach_target_orientation,
            )
            desired_rmp_transform = desired_tool_transform @ np.linalg.inv(
                self._rmp_right_gripper_to_tool_center
            )
            rmp_target_position, rmp_target_orientation = pose_from_tf_matrix(desired_rmp_transform)
            self._rmpflow.set_end_effector_target(
                target_position=rmp_target_position,
                target_orientation=rmp_target_orientation,
            )
            self._rmpflow.update_world()
            action = self._articulation_rmpflow.get_next_articulation_action()

            tool_center_position, _ = self._tool_center_prim.get_world_pose()
            if np.linalg.norm(tool_center_position - self._cube.get_world_pose()[0]) <= self._approach_tolerance:
                self._status = Status.GRASP
        
        elif self._status == Status.GRASP:
            self._robot._gripper.close()
            self.attach_cube_to_curobo_world()
            self._status = Status.HANG



        return action

class PickPlaceTask(BaseTask):
    CUBE_SIZE = 0.05
    TARGET_SIZE = 0.16
    WORKSPACE_X = (0.35, 0.65)
    WORKSPACE_Y = (-0.30, 0.30)
    MIN_CUBE_TARGET_DISTANCE = 0.20

    def __init__(
        self,
        name: str,
        offset: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        BaseTask.__init__(self, name, offset)
        self._rng = np.random.default_rng(seed)
        self._is_success = False

    def set_up_scene(self, scene: Scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # SingleGeometryPrim wraps a pre-existing USD geometry; create the
        # flattened Cube first, then wrap it as a non-colliding visual marker.
        target_prim = UsdGeom.Cube.Define(scene.stage, "/World/TargetRegion")
        target_prim.CreateSizeAttr(1.0)
        target_prim.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
        self._region = scene.add(SingleGeometryPrim(
            prim_path="/World/TargetRegion",
            name="target_region",
            position=np.array([0.45, 0.0, 0.002]),
            scale=np.array([self.TARGET_SIZE, self.TARGET_SIZE, 0.004]),
            collision=False,
        ))

        self._cube = scene.add(DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0.50, 0.0, self.CUBE_SIZE / 2.0]),
            size=self.CUBE_SIZE,
            color=np.array([0.0, 0.3, 1.0]),
        ))

        self._franka = scene.add(Franka(
            prim_path="/World/Franka",
            name="franka",
        ))

        self._task_objects = {
            "franka": self._franka,
            "cube": self._cube,
            "target_region": self._region,
        }

    def post_reset(self):
        """Sample reachable, separated pick and place locations for each episode."""
        self._is_success = False
        cube_xy = self._sample_workspace_xy()
        target_xy = self._sample_workspace_xy(min_distance_from=cube_xy)
        cube_yaw = self._rng.uniform(-np.pi/2, np.pi/2)

        self._cube.set_world_pose(
            position=np.array([cube_xy[0], cube_xy[1], self.CUBE_SIZE / 2.0]),
            orientation=np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)]),
        )
        self._region.set_world_pose(position=np.array([target_xy[0], target_xy[1], 0.002]))

    def is_done(self) -> bool:
        """Return True once the cube has been placed and settled inside the target region."""
        if self._is_success:
            return True

        cube_position, _ = self._cube.get_world_pose()
        target_position, _ = self._region.get_world_pose()

        # Require the whole cube footprint, not merely its centre, to be inside
        # the target square.  The small Z tolerance accepts a cube resting on
        # the ground plane while rejecting a cube held above the marker.
        half_clearance = (self.TARGET_SIZE - self.CUBE_SIZE) / 2.0
        footprint_inside = np.all(np.abs(cube_position[:2] - target_position[:2]) <= half_clearance)
        resting_on_ground = abs(cube_position[2] - self.CUBE_SIZE / 2.0) <= 0.02
        settled = np.linalg.norm(self._cube.get_linear_velocity()) <= 0.05

        self._is_success = bool(footprint_inside and resting_on_ground and settled)
        return self._is_success

    def _sample_workspace_xy(self, min_distance_from: Optional[np.ndarray] = None) -> np.ndarray:
        while True:
            xy = self._rng.uniform(
                low=np.array([self.WORKSPACE_X[0], self.WORKSPACE_Y[0]]),
                high=np.array([self.WORKSPACE_X[1], self.WORKSPACE_Y[1]]),
            )
            if min_distance_from is None or np.linalg.norm(xy - min_distance_from) >= self.MIN_CUBE_TARGET_DISTANCE:
                return xy
    
def main():
    world = World(stage_units_in_meters=1.0)

    task = PickPlaceTask(name="PNP")
    world.add_task(task)

    world.reset()
    world.step(render=True)

    franka = world.scene.get_object("franka")
    cube = world.scene.get_object("cube")
    controller = PnPController(
        name="pnp_controller",
        robot=franka,
        cube=cube,
        world=world,
    )
    articulation_controller = franka.get_articulation_controller()
    controller.reset()

    reset_needed = False
    while simulation_app.is_running():
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                world.reset()
                controller.reset()
                reset_needed = False

            action = controller.forward()
            if action is not None:
                articulation_controller.apply_action(action)

        world.step(render=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        simulation_app.close()
