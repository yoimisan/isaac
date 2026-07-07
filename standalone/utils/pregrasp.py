import numpy as np
import carb

from curobo.types.math import Pose
from isaacsim.core.prims import SingleXFormPrim


def sample_cube_pregrasp_pose(cube) -> Pose:
    face_index = np.random.randint(2)

    cube_size = cube.get_size()
    offset_length = 1.5 * (np.sqrt(2.0) * cube_size / 2.0)
    theta = np.random.random() * np.pi / 4.0 + np.pi / 4.0

    if face_index == 0:
        local_frame_translation = [
            offset_length * np.cos(theta),
            0.0,
            offset_length * np.sin(theta),
        ]

        local_frame_x = [np.cos(theta - np.pi / 2.0), 0.0, np.sin(theta - np.pi / 2.0)]
        local_frame_y = [0.0, -1.0, 0.0]
        local_frame_z = [np.cos(theta - np.pi), 0.0, np.sin(theta - np.pi)]
    else:
        local_frame_translation = [
            0.0,
            offset_length * np.cos(theta),
            offset_length * np.sin(theta),
        ]

        if theta < np.pi / 2:
            local_frame_x = [0.0, -np.sin(theta), np.cos(theta)]
            local_frame_y = [-1.0, 0.0, 0.0]
            local_frame_z = [0.0, -np.cos(theta), -np.sin(theta)]
        else:
            local_frame_x = [0.0, np.sin(np.pi - theta), np.cos(np.pi - theta)]
            local_frame_y = [1.0, 0.0, 0.0]
            local_frame_z = [0.0, np.cos(np.pi - theta), -np.sin(np.pi - theta)]

    local_rotation = np.column_stack((local_frame_x, local_frame_y, local_frame_z))

    local_transform = np.eye(4, dtype=np.float32)
    local_transform[:3, :3] = local_rotation
    local_transform[:3, 3] = local_frame_translation

    return Pose.from_matrix(local_transform)


def create_cube_pregrasp_frame(
    world,
    cube,
    prim_path: str = "/World/Cube/cube_pregrasp",
    name: str = "cube_pregrasp",
    exist_ok: bool = False
) -> SingleXFormPrim:
    local_frame_pose = sample_cube_pregrasp_pose(cube)
    loginfo: str = f"Create local frame pose: {str(local_frame_pose.to_list())}."
    carb.log_info(loginfo)

    if world.scene.object_exists(name) and exist_ok:
        world.scene.remove_object(name)

    return world.scene.add(SingleXFormPrim(
        prim_path=prim_path,
        name=name,
        translation=local_frame_pose.position.squeeze(0).detach().cpu().numpy(),
        orientation=local_frame_pose.quaternion.squeeze(0).detach().cpu().numpy(),
    ))
