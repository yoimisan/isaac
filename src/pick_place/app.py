"""Application loop for the standalone pick-and-place demo."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import numpy as np
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

from pick_place.controller import PnPController
from pick_place.states import PickPlacePhase
from pick_place.task import PickPlaceTask

if TYPE_CHECKING:
    from data_collection.config import DataCollectionConfig
    from data_collection.runtime import DataCollectionRuntime


_ENABLE_RECOVERY_TEST_PERTURBATION = False
_PERTURB_AFTER_APPROACH_PERCENT = 0.5
_PERTURB_CUBE_Y_OFFSET = 0.12


def _apply_recovery_test_perturbation(cube: DynamicCuboid) -> None:
    """Move the cube once to validate observation-driven state recovery."""
    position, orientation = cube.get_world_pose()
    perturbed_position = np.asarray(position).copy()
    direction = -1.0 if perturbed_position[1] > 0.0 else 1.0
    perturbed_position[1] += direction * _PERTURB_CUBE_Y_OFFSET
    cube.set_world_pose(
        position=perturbed_position,
        orientation=orientation,
    )
    carb.log_warn(
        "Applied recovery-test perturbation: "
        f"cube moved from {position} to {perturbed_position}."
    )


def run(
    simulation_app: SimulationApp,
    data_collection_config: DataCollectionConfig | None = None,
) -> None:
    """Run the interactive pick-and-place simulation."""
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

    data_collection: DataCollectionRuntime | None = None
    if data_collection_config is not None and data_collection_config.enabled:
        from data_collection.runtime import DataCollectionRuntime

        data_collection = DataCollectionRuntime(
            data_collection_config,
            world,
            franka,
            articulation_controller,
        )

    controller.reset()
    if data_collection is not None:
        data_collection.begin_episode(world.current_time)

    reset_needed = False
    perturbation_applied = False
    approach_ticks = 0
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
                    if data_collection is not None:
                        data_collection.begin_episode(world.current_time)
                    reset_needed = False
                    perturbation_applied = False
                    approach_ticks = 0

                if _ENABLE_RECOVERY_TEST_PERTURBATION and not perturbation_applied:
                    if controller.phase is PickPlacePhase.PLACE:
                        trajectory = controller._state_objects[controller.phase]._trajectory
                        if trajectory is not None:
                            approach_ticks += 1
                            length = len(trajectory)
                            if approach_ticks >= int(_PERTURB_AFTER_APPROACH_PERCENT * length):
                                _apply_recovery_test_perturbation(cube)
                                perturbation_applied = True
                    else:
                        approach_ticks = 0

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
            if (
                data_collection is not None
                and controller.is_complete()
                and task.is_done()
            ):
                data_collection.finish_successful_episode()
    finally:
        if data_collection is not None:
            data_collection.close()
