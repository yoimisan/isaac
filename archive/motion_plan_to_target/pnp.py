from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
from pathlib import Path

from isaacsim.core.api import World

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from archive.motion_plan_to_target.utils.curobo_motion import FrankaController, build_goal_pose_from_frame
from archive.motion_plan_to_target.utils.pregrasp import create_cube_pregrasp_frame
from archive.motion_plan_to_target.utils.randomization import randomize_scene_layout
from archive.motion_plan_to_target.utils.scene_setup import open_stage_and_wait, register_pnp_scene


USD_PATH = "/home/yoisan/Documents/isaac-scenes/toy.usd"


def main():
    open_stage_and_wait(simulation_app, USD_PATH)

    world = World(stage_units_in_meters=1.0)
    franka, cube, target_region, table = register_pnp_scene(world)

    randomize_scene_layout(franka, cube, target_region, table)
    pregrasp_xform = create_cube_pregrasp_frame(world, cube)

    world.reset()

    franka_controller = FrankaController(world.stage)
    franka_controller.register_franka(world.scene.get_object("franka"))

    base_prim = franka_controller.get_base_frame()
    goal_pose = build_goal_pose_from_frame(
        source_prim=pregrasp_xform.prim,
        target_prim=base_prim.prim,
        tensor_args=franka_controller.tensor_args,
    )

    trajs = franka_controller.forward(goal_pose)
    index = 0
    length = len(trajs)

    reset_needed = False
    while simulation_app.is_running():
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                randomize_scene_layout(franka, cube, target_region, table)
                world.reset()
                pregrasp_xform = create_cube_pregrasp_frame(world, cube, exist_ok=True)
                reset_needed = False
                goal_pose = build_goal_pose_from_frame(
                    source_prim=pregrasp_xform.prim,
                    target_prim=base_prim.prim,
                    tensor_args=franka_controller.tensor_args,
                )
                trajs = franka_controller.forward(goal_pose)
                length = len(trajs)
                index = 0
            else:
                franka_controller.apply_curobo_joint_state(trajs[index])
                index += 1

                if index >= length:
                    index = length - 1
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        simulation_app.close()
