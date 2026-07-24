"""Pick-and-place scene adapter for the shared episode replayer."""

from __future__ import annotations

from isaacsim.core.api import World

from data_collection.replay_runtime import ReplaySceneHandles
from pick_place.task import PickPlaceTask


def build_replay_scene() -> ReplaySceneHandles:
    """Build the PnP world and expose handles required by data replay."""
    world = World(stage_units_in_meters=1.0)
    world.add_task(PickPlaceTask(name="PNP", seed=0))
    world.reset()
    world.step(render=True)

    robot = world.scene.get_object("franka")
    cube = world.scene.get_object("cube")
    target_region = world.scene.get_object("target_region")
    return ReplaySceneHandles(
        world=world,
        robot=robot,
        articulation_controller=robot.get_articulation_controller(),
        objects={"cube": cube, "target_region": target_region},
    )
