"""Isaac Sim task definition for randomized Franka pick-and-place episodes."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, UsdGeom, UsdLux


class PickPlaceTask(BaseTask):
    """Create and evaluate a randomized single-cube pick-and-place task."""

    CUBE_SIZE = 0.05
    TARGET_SIZE = 0.16
    WORKSPACE_X = (0.35, 0.65)
    WORKSPACE_Y = (-0.30, 0.30)
    MIN_CUBE_TARGET_DISTANCE = 0.20

    def __init__(
        self,
        name: str,
        offset: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, offset)
        self._rng = np.random.default_rng(seed)
        self._is_success = False

    def set_up_scene(self, scene: Scene) -> None:
        """Add the ground, target region, cube, and Franka to the scene."""
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        UsdGeom.Scope.Define(scene.stage, "/World/Lights")
        dome_light = UsdLux.DomeLight.Define(
            scene.stage,
            "/World/Lights/Dome",
        )
        dome_light.CreateIntensityAttr(100.0)
        distant_light = UsdLux.DistantLight.Define(
            scene.stage,
            "/World/Lights/Distant",
        )
        distant_light.CreateIntensityAttr(500.0)

        target_prim = UsdGeom.Cube.Define(scene.stage, "/World/TargetRegion")
        target_prim.CreateSizeAttr(1.0)
        target_prim.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
        self._region = scene.add(
            SingleGeometryPrim(
                prim_path="/World/TargetRegion",
                name="target_region",
                position=np.array([0.45, 0.0, 0.002]),
                scale=np.array([self.TARGET_SIZE, self.TARGET_SIZE, 0.004]),
                collision=False,
            )
        )
        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.50, 0.0, self.CUBE_SIZE / 2.0]),
                size=self.CUBE_SIZE,
                color=np.array([0.0, 0.3, 1.0]),
            )
        )
        self._franka = scene.add(Franka(prim_path="/World/Franka", name="franka"))
        self._task_objects = {
            "franka": self._franka,
            "cube": self._cube,
            "target_region": self._region,
        }

    def post_reset(self) -> None:
        """Sample reachable, separated pick and place locations for an episode."""
        self._is_success = False
        cube_xy = self._sample_workspace_xy()
        target_xy = self._sample_workspace_xy(min_distance_from=cube_xy)
        cube_yaw = self._rng.uniform(-np.pi / 2, np.pi / 2)
        self._cube.set_world_pose(
            position=np.array([cube_xy[0], cube_xy[1], self.CUBE_SIZE / 2.0]),
            orientation=np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)]),
        )
        self._region.set_world_pose(position=np.array([target_xy[0], target_xy[1], 0.002]))

    def is_done(self) -> bool:
        """Return whether the cube has settled fully inside the target region."""
        if self._is_success:
            return True

        cube_position, _ = self._cube.get_world_pose()
        target_position, _ = self._region.get_world_pose()
        half_clearance = (self.TARGET_SIZE - self.CUBE_SIZE) / 2.0
        footprint_inside = np.all(np.abs(cube_position[:2] - target_position[:2]) <= half_clearance)
        resting_on_ground = abs(cube_position[2] - self.CUBE_SIZE / 2.0) <= 0.02
        settled = np.linalg.norm(self._cube.get_linear_velocity()) <= 0.05
        self._is_success = bool(footprint_inside and resting_on_ground and settled)
        return self._is_success

    def _sample_workspace_xy(self, min_distance_from: np.ndarray | None = None) -> np.ndarray:
        while True:
            xy = self._rng.uniform(
                low=np.array([self.WORKSPACE_X[0], self.WORKSPACE_Y[0]]),
                high=np.array([self.WORKSPACE_X[1], self.WORKSPACE_Y[1]]),
            )
            if min_distance_from is None or np.linalg.norm(xy - min_distance_from) >= self.MIN_CUBE_TARGET_DISTANCE:
                return xy
