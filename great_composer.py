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
            "doc": "Path to targt cube."
        },
    ]

    def on_init(self):
        self._cube = None

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
        self._move_cube()

    def _setup(self):
        cube_path: str = self._get_exposed_variable("targetCubePath")
        self._cube = self._fetch_prim(cube_path)

    def _reset(self):
        self._cube = None
    

    def _get_exposed_variable(self, attr_name):
        full_attr_name = f"{EXPOSED_ATTR_NS}:{self.BEHAVIOR_NS}:{attr_name}"
        return get_exposed_variable(self.prim, full_attr_name)

    def _fetch_prim(self, prim_path: str):
        fetched_prim = None
        if not self.stage:
            carb.log_warn(f"[{self.prim_path}] Stage is not valid to access target prim '{target_prim_path}'.")
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

        # Look for a valid translation op to set the new rotation
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

    def _move_cube(self):
        if self._cube:
            cube_location = self._get_location(self._cube)
            cube_location[0] += 0.01
            self._set_location(self._cube, cube_location)

            


