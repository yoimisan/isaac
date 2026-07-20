"""Application loop for the standalone pick-and-place demo."""

from __future__ import annotations

from isaacsim import SimulationApp
from isaacsim.core.api import World

from adversary.background import NoBackgroundDisturbance
from adversary.executor import IsaacSimDisturbanceExecutor
from adversary.ghost import NaughtyGhost
from adversary.task_object import OneShotStateObjectPerturbation
from adversary.types import TaskStateView
from pick_place.controller import PnPController
from pick_place.states import PickPlacePhase
from pick_place.task import PickPlaceTask


_ENABLE_NAUGHTY_GHOST = True
_NAUGHTY_TRIGGER_STATE = PickPlacePhase.APPROACH.name
_NAUGHTY_TRIGGER_AFTER_STEPS = 30
_NAUGHTY_CUBE_POSITION_OFFSET = (0.0, 0.08, 0.0)


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
    naughty_ghost = NaughtyGhost(
        executor=IsaacSimDisturbanceExecutor(objects={"cube": cube}),
        background_policy=NoBackgroundDisturbance(),
        task_object_policy=OneShotStateObjectPerturbation(
            trigger_state=_NAUGHTY_TRIGGER_STATE,
            trigger_after_steps=_NAUGHTY_TRIGGER_AFTER_STEPS,
            target_name="cube",
            position_offset=_NAUGHTY_CUBE_POSITION_OFFSET,
        ),
    )
    naughty_ghost.reset()

    reset_needed = False
    while simulation_app.is_running():
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                world.reset()
                controller.reset()
                naughty_ghost.reset()
                reset_needed = False

            if _ENABLE_NAUGHTY_GHOST:
                naughty_ghost.step(
                    TaskStateView(
                        task_name="pick_place",
                        state_name=controller.phase.name,
                    )
                )

            action = controller.forward()
            if action is not None:
                articulation_controller.apply_action(action)

        world.step(render=True)
        # if world.is_playing() and controller.is_complete() and task.is_done():
        #     carb.log_info("Pick-and-place completed; the arm returned to its reset pose.")
        #     world.pause()
