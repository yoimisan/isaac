"""Application loop for the standalone pick-and-place demo."""

from __future__ import annotations

import carb
from isaacsim import SimulationApp
from isaacsim.core.api import World

from pick_place.controller import PnPController
from pick_place.task import PickPlaceTask


def run(simulation_app: SimulationApp) -> None:
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
        # if world.is_playing() and controller.is_complete() and task.is_done():
        #     carb.log_info("Pick-and-place completed; the arm returned to its reset pose.")
        #     world.pause()

