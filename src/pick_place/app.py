"""Application loop for the standalone pick-and-place demo."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleGeometryPrim

from adversary.background import NoBackgroundDisturbance
from adversary.executor import IsaacSimDisturbanceExecutor
from adversary.ghost import NaughtyGhost
from adversary.types import (
    ObjectPoseView,
    TaskObjectDisturbanceContext,
    TaskStateView,
)
from pick_place.controller import PnPController
from pick_place.disturbance_policy import PickPlaceTaskObjectDisturbancePolicy
from pick_place.task import PickPlaceTask

if TYPE_CHECKING:
    from data_collection.config import DataCollectionConfig
    from data_collection.runtime import DataCollectionRuntime


def _create_naughty_ghost(
    cube: DynamicCuboid,
    *,
    seed: int | None,
    attack_count_range: tuple[int, int],
) -> NaughtyGhost:
    """Build the optional PnP task-object disturbance channel."""
    return NaughtyGhost(
        executor=IsaacSimDisturbanceExecutor(objects={"cube": cube}),
        background_policy=NoBackgroundDisturbance(),
        task_object_policy=PickPlaceTaskObjectDisturbancePolicy(
            target_name="cube",
            target_region_name="target_region",
            target_half_clearance=(
                PickPlaceTask.TARGET_SIZE - PickPlaceTask.CUBE_SIZE
            )
            / 2.0,
            workspace_x=PickPlaceTask.WORKSPACE_X,
            workspace_y=PickPlaceTask.WORKSPACE_Y,
            seed=seed,
            attack_count_range=attack_count_range,
        ),
    )


def _observe_task_object_context(
    controller: PnPController,
    cube: DynamicCuboid,
    target_region: SingleGeometryPrim,
) -> TaskObjectDisturbanceContext:
    """Build a pure-data snapshot for the PnP naughty policy."""
    cube_position, cube_orientation = cube.get_world_pose()
    target_position, target_orientation = target_region.get_world_pose()
    return TaskObjectDisturbanceContext(
        task_state=TaskStateView(
            task_name="pick_place",
            state_name=controller.phase.name,
            state_entry_id=controller.state_entry_id,
        ),
        objects={
            "cube": ObjectPoseView(
                position=tuple(float(value) for value in cube_position),
                orientation=tuple(float(value) for value in cube_orientation),
            ),
            "target_region": ObjectPoseView(
                position=tuple(float(value) for value in target_position),
                orientation=tuple(float(value) for value in target_orientation),
            ),
        },
    )


def run(
    simulation_app: SimulationApp,
    data_collection_config: DataCollectionConfig | None = None,
    *,
    enable_naughty_ghost: bool = False,
    naughty_seed: int | None = 0,
    naughty_attack_count_range: tuple[int, int] = (0, 5),
) -> None:
    """Run clean or perturbed pick-and-place, optionally recording episodes."""
    world = World(stage_units_in_meters=1.0)
    task = PickPlaceTask(name="PNP")
    world.add_task(task)

    world.reset()
    world.step(render=True)

    franka = world.scene.get_object("franka")
    cube = world.scene.get_object("cube")
    target_region = world.scene.get_object("target_region")
    controller = PnPController(
        name="pnp_controller",
        robot=franka,
        cube=cube,
        world=world,
    )
    articulation_controller = franka.get_articulation_controller()

    data_collection: DataCollectionRuntime | None = None
    if data_collection_config is not None and data_collection_config.enabled:
        from data_collection.runtime import DataCollectionRuntime

        data_collection = DataCollectionRuntime(
            data_collection_config,
            world,
            franka,
            articulation_controller,
            replay_objects={
                "cube": cube,
                "target_region": target_region,
            },
        )

    naughty_ghost = None
    if enable_naughty_ghost:
        naughty_ghost = _create_naughty_ghost(
            cube,
            seed=naughty_seed,
            attack_count_range=naughty_attack_count_range,
        )

    controller.reset()
    if naughty_ghost is not None:
        naughty_ghost.reset()
    if data_collection is not None:
        data_collection.begin_episode(world.current_time)

    reset_needed = False
    successful_episodes = 0
    try:
        while simulation_app.is_running():
            if world.is_stopped() and not reset_needed:
                reset_needed = True
            if world.is_playing():
                if reset_needed:
                    if data_collection is not None:
                        data_collection.before_world_reset()
                    world.reset()
                    if data_collection is not None:
                        data_collection.after_world_reset(world)
                    controller.reset()
                    if naughty_ghost is not None:
                        naughty_ghost.reset()
                    if data_collection is not None:
                        data_collection.begin_episode(world.current_time)
                    reset_needed = False

                if naughty_ghost is not None:
                    naughty_ghost.step(
                        _observe_task_object_context(
                            controller,
                            cube,
                            target_region,
                        )
                    )

                task_state = controller.phase.name
                action = controller.forward()
                if action is not None:
                    articulation_controller.apply_action(action)
                if data_collection is not None:
                    data_collection.record_frame(
                        world.current_time,
                        task_state=task_state,
                    )

            world.step(render=True)
            if data_collection is not None and controller.is_complete():
                data_collection.finish_successful_episode()
                successful_episodes += 1
                carb.log_info(
                    "Automatic collection completed "
                    f"episode {successful_episodes}/"
                    f"{data_collection.num_episodes}."
                )
                if successful_episodes >= data_collection.num_episodes:
                    break

                world.reset()
                data_collection.after_world_reset(world)
                controller.reset()
                if naughty_ghost is not None:
                    naughty_ghost.reset()
                data_collection.begin_episode(world.current_time)
    finally:
        if data_collection is not None:
            data_collection.close()
