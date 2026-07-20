"""Application loop for the standalone pick-and-place demo."""

from __future__ import annotations

from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

from adversary.background import NoBackgroundDisturbance
from adversary.executor import IsaacSimDisturbanceExecutor
from adversary.ghost import NaughtyGhost
from adversary.types import (
    RigidObjectView,
    TaskObjectDisturbanceContext,
    TaskStateView,
)
from pick_place.controller import PnPController
from pick_place.disturbance_policy import PickPlaceTaskObjectDisturbancePolicy
from pick_place.task import PickPlaceTask


_ENABLE_NAUGHTY_GHOST = True
_NAUGHTY_SEED = 1


def _observe_task_object_context(
    controller: PnPController,
    cube: DynamicCuboid,
) -> TaskObjectDisturbanceContext:
    """Build a pure-data snapshot for the PnP naughty policy."""
    cube_position, cube_orientation = cube.get_world_pose()
    return TaskObjectDisturbanceContext(
        task_state=TaskStateView(
            task_name="pick_place",
            state_name=controller.phase.name,
        ),
        objects={
            "cube": RigidObjectView(
                position=tuple(float(value) for value in cube_position),
                orientation=tuple(float(value) for value in cube_orientation),
            )
        },
    )


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
        task_object_policy=PickPlaceTaskObjectDisturbancePolicy(
            target_name="cube",
            workspace_x=PickPlaceTask.WORKSPACE_X,
            workspace_y=PickPlaceTask.WORKSPACE_Y,
            seed=_NAUGHTY_SEED,
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
                    _observe_task_object_context(controller, cube)
                )

            action = controller.forward()
            if action is not None:
                articulation_controller.apply_action(action)

        world.step(render=True)
        # if world.is_playing() and controller.is_complete() and task.is_done():
        #     carb.log_info("Pick-and-place completed; the arm returned to its reset pose.")
        #     world.pause()
