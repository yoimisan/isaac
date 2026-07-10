from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import is_stage_loading, open_stage


def open_stage_and_wait(simulation_app, usd_path: str):
    success = open_stage(usd_path)
    if not success:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()


def register_pnp_scene(world):
    franka = world.scene.add(Robot(
        prim_path="/World/Franka",
        name="franka",
    ))

    cube = world.scene.add(DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
    ))

    target_region = world.scene.add(SingleGeometryPrim(
        prim_path="/World/TargetRegion",
        name="target_region",
    ))

    table = world.scene.add(SingleGeometryPrim(
        prim_path="/World/Table",
        name="table",
    ))

    return franka, cube, target_region, table
