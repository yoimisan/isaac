from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage, get_current_stage

usd_path = "/home/yoisan/Documents/isaac-scenes/toy.usd"

success = open_stage(usd_path)
if not success:
    raise RuntimeError(f"Failed to open stage: {usd_path}")

simulation_app.update()

from isaacsim.core.prims import SingleArticulation, SingleGeometryPrim
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot

world = World(stage_units_in_meters=1.0)

franka = world.scene.add(Robot(
    prim_path="/World/Franka",
    name="franka"
))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/Cube",
    name="cube"
))

target_region = world.scene.add(SingleGeometryPrim(
    prim_path="/World/TargetRegion",
    name="target_region"
))

table = world.scene.add(SingleGeometryPrim(
    prim_path="/World/Table",
    name="table"
))

world.reset()
# Above is a very elegant way of building the world. I like it.

# import pdb; pdb.set_trace()
# simulation_app.close()
# import sys; sys.exit(0)

# Now you can find /World/Franka, /World/Target, etc.
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()