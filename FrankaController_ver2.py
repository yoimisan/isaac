import carb
import torch
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from omni.kit.scripting import BehaviorScript
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.xforms import get_world_pose


from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class FrankaController(BehaviorScript):

    def on_init(self):
        self.franka_path = "/World/Franka"  # maybe different in your stage
        self.cube_path = "/World/Cube"
        self.robot = None
        self.ready = False
        self.t = 0.0
        self.arm_joint_indices = None

        self.target_position = np.array([1.0, -1.5, 0.0, -2.8, 0.0, 2.8, 1.2, 0.04, 0.04])

    def on_play(self):
        # reset your controller state here
        self.ready = False
        self.t = 0.0

    def on_update(self, current_time, delta_time):
        # first frame: find/initialize Franka
        # later frames: send joint commands
        if self.t == 0.0:
            self._setup()
            result = self.motion_gen.plan_single(
                self.start_state,
                self.goal_pose
            )
            if result.success:
                self.traj = result.get_interpolated_plan()
                self.traj_index = 0
            else:
                carb.log_error("Motion Planing result indicate an unsuccessful motion planning.")
        
        if self.traj_index < len(self.traj):
            cmd_state = self.traj[self.traj_index]
            action = ArticulationAction(
                joint_positions=cmd_state.position.detach().cpu().numpy(),
                joint_indices=self.arm_joint_indices,
            )
            self.robot.apply_action(action)
            self.traj_index += 1
        self.t += delta_time
        


    def on_stop(self):
        self.robot = None
        self.ready = False
    

    def _setup(self):
        self.cube = self.stage.GetPrimAtPath(Sdf.Path(self.cube_path))
        self.cube_position, self.cube_rotation = get_world_pose(self.cube_path)
        self.goal_pose = Pose.from_list(
            np.concatenate(
                [self.cube_position, self.cube_rotation],
                axis=0
            ),
            # q_xyzw=True
        )

        self.robot = SingleArticulation(
            prim_path = self.franka_path,
            name = "franka_controller_robot"
        )
        self.robot.initialize()
        print(self.robot.dof_names)

        self.motion_gen_config = MotionGenConfig.load_from_robot_config("franka.yml")
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()

        isaac_joint_names = self.robot.dof_names
        curobo_joint_names = self.motion_gen.joint_names
        self.arm_joint_indices = [isaac_joint_names.index(name) for name in curobo_joint_names]

        isaac_q = self.robot.get_joint_positions()
        q_for_curobo = isaac_q[self.arm_joint_indices]
        self.start_state = JointState.from_position(
            torch.tensor(q_for_curobo, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=curobo_joint_names,
        )

        carb.log_info(f"Start state: {self.start_state}\nGoal state: {self.goal_pose}")

        self.ready = True
