import numpy as np

from omni.kit.scripting import BehaviorScript
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

class FrankaController(BehaviorScript):

    def on_init(self):
        self.franka_path = "/World/Franka"  # maybe different in your stage
        self.robot = None
        self.ready = False
        self.t = 0.0

        self.target_position = np.array([0.0, -1.5, 0.0, -2.8, 0.0, 2.8, 1.2, 0.04, 0.04])

    def on_play(self):
        # reset your controller state here
        self.ready = False

    def on_update(self, current_time, delta_time):
        # first frame: find/initialize Franka
        # later frames: send joint commands
        if self.t == 0.0:
            self._setup()
        if self.ready:
            action = ArticulationAction(joint_positions=self.target_position)
            self.robot.apply_action(action)


    def on_stop(self):
        self.robot = None
        self.ready = False
    

    def _setup(self):
        self.robot = SingleArticulation(
            prim_path = self.franka_path,
            name = "franka_controller_robot"
        )
        self.robot.initialize()
        self.ready = True