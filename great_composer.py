import carb

import omni.kit.window.property
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    check_if_exposed_variables_should_be_removed,
    create_exposed_variables,
    get_exposed_variable,
    remove_exposed_variables,
)
from omni.kit.scripting import BehaviorScript
from pxr import Gf, Sdf, Usd, UsdGeom


class GreatComposer(BehaviorScript):
    BEHAVIOR_NS = "GreatComposer"
    VARIABLES_TO_EXPOSE = [
        {
            "attr_name": "targetCubePath",
            "attr_type": Sdf.ValueTypeNames.String,
            "default_value": "",
            "doc": "Path to target cube.",
        },
        {
            "attr_name": "pose0X",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 0.0,
            "doc": "X coordinate of candidate pose 0.",
        },
        {
            "attr_name": "pose0Y",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 0.0,
            "doc": "Y coordinate of candidate pose 0.",
        },
        {
            "attr_name": "pose1X",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 1.0,
            "doc": "X coordinate of candidate pose 1.",
        },
        {
            "attr_name": "pose1Y",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 0.0,
            "doc": "Y coordinate of candidate pose 1.",
        },
        {
            "attr_name": "pose2X",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 1.0,
            "doc": "X coordinate of candidate pose 2.",
        },
        {
            "attr_name": "pose2Y",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 1.0,
            "doc": "Y coordinate of candidate pose 2.",
        },
        {
            "attr_name": "pose3X",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 0.0,
            "doc": "X coordinate of candidate pose 3.",
        },
        {
            "attr_name": "pose3Y",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 1.0,
            "doc": "Y coordinate of candidate pose 3.",
        },
        {
            "attr_name": "transitionSpeed",
            "attr_type": Sdf.ValueTypeNames.Double,
            "default_value": 2.0,
            "doc": "Linear movement speed between candidate poses in scene units per second.",
        },
    ]

    def on_init(self):
        self._cube = None
        self._candidate_xy = []
        self._transition_speed = 2.0
        self._target_pose_index = 1
        self._fixed_z = 0.0

        create_exposed_variables(
            self.prim,
            EXPOSED_ATTR_NS,
            self.BEHAVIOR_NS,
            self.VARIABLES_TO_EXPOSE
        )

        omni.kit.window.property.get_window().request_rebuild()

    def on_destroy(self):
        self._reset()
        if check_if_exposed_variables_should_be_removed(self.prim, __file__):
            remove_exposed_variables(
                self.prim,
                EXPOSED_ATTR_NS,
                self.BEHAVIOR_NS,
                self.VARIABLES_TO_EXPOSE
            )

    def on_play(self):
        self._setup()

    def on_stop(self):
        self._reset()

    def on_update(self, current_time: float, delta_time: float):
        if delta_time <= 0:
            return
        self._move_cube(delta_time)

    def _setup(self):
        cube_path: str = self._get_exposed_variable("targetCubePath")
        self._candidate_xy = self._get_candidate_xy()
        self._transition_speed = max(0.0, self._as_float(self._get_exposed_variable("transitionSpeed"), 2.0))
        self._target_pose_index = 1
        self._cube = self._fetch_prim(cube_path)

        if self._cube:
            cube_location = self._get_location(self._cube)
            self._fixed_z = cube_location[2]
            self._set_location(self._cube, self._get_candidate_location(0))

    def _reset(self):
        self._cube = None
        self._candidate_xy = []
        self._target_pose_index = 1
        self._fixed_z = 0.0

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)

    def _as_float(self, value, default_value: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default_value

    def _get_candidate_xy(self):
        candidate_xy = []
        for index in range(4):
            x = self._as_float(self._get_exposed_variable(f"pose{index}X"), 0.0)
            y = self._as_float(self._get_exposed_variable(f"pose{index}Y"), 0.0)
            candidate_xy.append((x, y))
        return candidate_xy

    def _fetch_prim(self, prim_path: str):
        fetched_prim = None
        if not self.stage:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to access target prim '{prim_path}'.")
        else:  # Stage is valid
            _fetched_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            if _fetched_prim and _fetched_prim.IsValid() and _fetched_prim.IsA(UsdGeom.Xformable):
                fetched_prim = _fetched_prim
            else:
                carb.log_warn(
                    f"[{self.prim_path}] Target prim '{prim_path}' not found, not valid, or not Xformable."
                )
        return fetched_prim

    def _get_location(self, prim) -> Gf.Vec3d:
        # Get the location of the prim based on the available xformOps, create a default translation if none exists
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()

        for op in xform_ops:
            op_name = op.GetOpName()
            if op_name == "xformOp:translate":
                return op.Get()
            elif op_name == "xformOp:transform":
                transform_matrix = op.Get()
                return Gf.Transform(transform_matrix).GetTranslation()

        # If no translation op exists, create one with a default translation
        translate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
        default_translation = Gf.Vec3d(0.0, 0.0, 0.0)
        translate_op.Set(default_translation)
        return default_translation

    def _set_location(self, prim, location: Gf.Vec3d):
        # Set the location of the prim based on the available xformOps
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()

        # Look for a valid translation op to set the new location
        for op in xform_ops:
            op_name = op.GetOpName()
            if op_name == "xformOp:translate":
                op.Set(location)
                return
            elif op_name == "xformOp:transform":
                transform_matrix = op.Get()
                transform = Gf.Transform(transform_matrix)
                transform.SetTranslation(location)
                op.Set(transform.GetMatrix())
                return

        carb.log_warn(f"No valid location op found on {prim.GetPath()}")

    def _get_candidate_location(self, pose_index: int) -> Gf.Vec3d:
        x, y = self._candidate_xy[pose_index]
        return Gf.Vec3d(x, y, self._fixed_z)

    def _move_cube(self, delta_time: float):
        if not self._cube or not self._candidate_xy or self._transition_speed <= 0.0:
            return

        remaining_distance = self._transition_speed * delta_time
        cube_location = self._get_location(self._cube)
        skipped_targets = 0

        while remaining_distance > 0.0 and skipped_targets < len(self._candidate_xy):
            target_location = self._get_candidate_location(self._target_pose_index)
            offset_x = target_location[0] - cube_location[0]
            offset_y = target_location[1] - cube_location[1]
            distance_to_target = (offset_x * offset_x + offset_y * offset_y) ** 0.5

            if distance_to_target <= 1e-6:
                cube_location = target_location
                self._target_pose_index = (self._target_pose_index + 1) % len(self._candidate_xy)
                skipped_targets += 1
                continue

            skipped_targets = 0
            if remaining_distance >= distance_to_target:
                cube_location = target_location
                remaining_distance -= distance_to_target
                self._target_pose_index = (self._target_pose_index + 1) % len(self._candidate_xy)
            else:
                ratio = remaining_distance / distance_to_target
                cube_location = Gf.Vec3d(
                    cube_location[0] + offset_x * ratio,
                    cube_location[1] + offset_y * ratio,
                    self._fixed_z,
                )
                remaining_distance = 0.0

        self._set_location(self._cube, cube_location)
