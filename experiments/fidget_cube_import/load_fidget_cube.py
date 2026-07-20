#!/usr/bin/env python3
"""Package, load, render, and physics-test the Fidget Cube in Isaac Sim 6.0.

The script consumes the visual USD produced by ``export_fidget_cube.py``.  It
creates a sim-ready package with a simple box collider, references that package
into an isolated validation scene, drops it onto a ground plane for at least
three simulated seconds, captures a render, and writes a machine-readable JSON
report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-usd", type=Path, required=True, help="Blender-exported fidget_cube_visual.usdc")
    parser.add_argument("--package-dir", type=Path, required=True, help="Generated sim-ready package directory")
    parser.add_argument("--report", type=Path, required=True, help="JSON validation report")
    parser.add_argument("--render", type=Path, required=True, help="Rendered PNG validation frame")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--duration", type=float, default=3.5, help="Simulated drop-test duration in seconds")
    return parser.parse_args()


ARGS = parse_args()
VISUAL_USD = ARGS.visual_usd.expanduser().resolve()
PACKAGE_DIR = ARGS.package_dir.expanduser().resolve()
REPORT_PATH = ARGS.report.expanduser().resolve()
RENDER_PATH = ARGS.render.expanduser().resolve()

if not VISUAL_USD.is_file():
    raise FileNotFoundError(f"Visual USD does not exist: {VISUAL_USD}")
if ARGS.width < 640 or ARGS.height < 480:
    raise ValueError("Validation rendering requires at least 640x480")
if ARGS.physics_hz < 120:
    raise ValueError("A 4.5 cm object requires at least 120 Hz physics for this validation")
if ARGS.duration < 3.0:
    raise ValueError("Physics validation must run for at least three simulated seconds")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": ARGS.width,
        "height": ARGS.height,
        "renderer": "RayTracedLighting",
    }
)

import carb
import numpy as np
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.experimental.prims import RigidPrim
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from PIL import Image
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


WRAPPER_NAME = "fidget_cube.usda"
VISUAL_NAME = "fidget_cube_visual.usdc"
MASS_KG = 0.08
EXPECTED_MATERIALS = {
    "Color_mat": {"diffuse_color": (0.020598998, 0.088981375, 0.0), "metallic": 0.0},
    "Cube_Color_mat": {"diffuse_color": (0.01960665, 0.01960665, 0.01960665), "metallic": 0.0},
    "Metal_mat": {"diffuse_color": (0.5, 0.5, 0.5), "metallic": 1.0},
}


def add_check(report: dict, name: str, passed: bool, detail: object) -> None:
    report["checks"][name] = {"passed": bool(passed), "detail": detail}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_asset_manifest(report: dict) -> None:
    """Record the validated package contract without absolute machine paths."""
    package_files = sorted(
        path for path in PACKAGE_DIR.rglob("*") if path.is_file() and path.name != "asset_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "asset": {
            "name": "Fidget Cube",
            "entrypoint": WRAPPER_NAME,
            "visual_layer": VISUAL_NAME,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "dimensions_m": report["asset"]["bounds"]["dimensions_m"],
            "mass_kg": MASS_KG,
            "collision": "box_proxy",
            "material_palette": report["asset"]["material_palette"],
        },
        "provenance": {
            "source": "BlenderKit",
            "source_asset_base_id": "e1548130-3cef-4cbd-bd4a-0b5b43b4f020",
            "author": "Klo Works",
            "license": "royalty_free",
            "source_url": "https://www.blenderkit.com/asset-gallery-detail/"
            "e1548130-3cef-4cbd-bd4a-0b5b43b4f020/",
        },
        "validation": {
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "physics_engine": "physx",
            "physics_hz": report["physics"]["physics_hz"],
            "duration_s": report["physics"]["duration_s"],
            "all_checks_passed": report["passed"],
            "render_mean_rgb": report["render"]["mean_rgb"],
            "render_max_rgb": report["render"]["max_rgb"],
        },
        "files": [
            {
                "path": str(path.relative_to(PACKAGE_DIR)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in package_files
        ],
    }
    (PACKAGE_DIR / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def copy_visual_payload() -> Path:
    """Copy the visual layer and its local dependencies into the package."""
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    destination = PACKAGE_DIR / VISUAL_NAME
    shutil.copy2(VISUAL_USD, destination)
    for child in VISUAL_USD.parent.iterdir():
        if child == VISUAL_USD or child.name.startswith("offline_inspection"):
            continue
        target = PACKAGE_DIR / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif child.is_file():
            shutil.copy2(child, target)
    return destination


def visual_bounds(visual_path: Path) -> tuple[list[float], list[float], list[float]]:
    stage = Usd.Stage.Open(str(visual_path))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"Cannot open visual asset with a default prim: {visual_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isclose(meters_per_unit, 1.0, abs_tol=1e-9):
        raise RuntimeError(f"Visual asset must use metersPerUnit=1.0, got {meters_per_unit}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    minimum = [float(v) for v in aligned.GetMin()]
    maximum = [float(v) for v in aligned.GetMax()]
    dimensions = [maximum[i] - minimum[i] for i in range(3)]
    return minimum, maximum, dimensions


def build_sim_ready_wrapper(visual_path: Path) -> tuple[Path, dict]:
    """Author physics as a non-destructive layer over the Blender visual USD."""
    minimum, maximum, dimensions = visual_bounds(visual_path)
    center = [(minimum[i] + maximum[i]) * 0.5 for i in range(3)]
    wrapper_path = PACKAGE_DIR / WRAPPER_NAME
    wrapper_path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(wrapper_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/FidgetCube")
    root.GetPrim().GetReferences().AddReference(f"./{VISUAL_NAME}")
    root.GetPrim().SetCustomDataByKey("sourceAsset", "BlenderKit Fidget Cube")
    root.GetPrim().SetCustomDataByKey("sourceLicense", "royalty_free")
    root.GetPrim().SetCustomDataByKey("massKg", MASS_KG)

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(MASS_KG)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx_body.CreateEnableCCDAttr().Set(True)
    physx_body.CreateSolverPositionIterationCountAttr().Set(16)
    physx_body.CreateSolverVelocityIterationCountAttr().Set(4)

    collision = UsdGeom.Cube.Define(stage, "/FidgetCube/Collision")
    collision.CreateSizeAttr().Set(1.0)
    collision.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    collision.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision_xform = UsdGeom.Xformable(collision.GetPrim())
    collision_xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    collision_xform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    # PXR's USDA writer leaves an extra blank line at EOF. Normalize the text
    # layer so regenerated packages remain clean in Git diffs.
    wrapper_path.write_text(wrapper_path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")

    source_doc = PACKAGE_DIR / "SOURCE.md"
    source_doc.write_text(
        "# Fidget Cube asset provenance\n\n"
        "- Source: BlenderKit\n"
        "- Asset: Fidget Cube\n"
        "- Author: Klo Works\n"
        "- License: BlenderKit Royalty Free\n"
        "- Source URL: https://www.blenderkit.com/asset-gallery-detail/"
        "e1548130-3cef-4cbd-bd4a-0b5b43b4f020/\n\n"
        "The visual layer was exported from Blender without changing its natural scale. "
        "The root layer adds an 80 g rigid body and a stable box collision proxy.\n",
        encoding="utf-8",
    )
    return wrapper_path, {"minimum_m": minimum, "maximum_m": maximum, "dimensions_m": dimensions, "center_m": center}


def bind_preview_material(stage: Usd.Stage, prim: Usd.Prim, path: str, color: tuple[float, float, float]) -> None:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def look_at_matrix(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Gf.Matrix4d:
    eye_vector = Gf.Vec3d(*eye)
    target_vector = Gf.Vec3d(*target)
    forward = (target_vector - eye_vector).GetNormalized()
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(forward * up) > 0.99:
        up = Gf.Vec3d(0.0, 1.0, 0.0)
    right = (forward ^ up).GetNormalized()
    camera_up = (right ^ forward).GetNormalized()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0.0))
    matrix.SetRow(1, Gf.Vec4d(camera_up[0], camera_up[1], camera_up[2], 0.0))
    matrix.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0.0))
    matrix.SetRow(3, Gf.Vec4d(eye_vector[0], eye_vector[1], eye_vector[2], 1.0))
    return matrix


def build_validation_stage(wrapper_path: Path, scene_path: Path) -> tuple[Usd.Stage, Usd.Prim]:
    stage = stage_utils.create_new_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube_prim = stage_utils.add_reference_to_stage(str(wrapper_path), "/World/FidgetCube")
    if not cube_prim or not cube_prim.IsValid():
        raise RuntimeError(f"Isaac Sim failed to reference {wrapper_path}")
    xform_api = UsdGeom.XformCommonAPI(cube_prim)
    if not xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.14)):
        raise RuntimeError("Failed to set the Fidget Cube validation pose")

    ground_xform = UsdGeom.Xform.Define(stage, "/World/Ground")
    UsdGeom.Xformable(ground_xform.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    ground = UsdGeom.Cube.Define(stage, "/World/Ground/Geometry")
    ground.CreateSizeAttr().Set(1.0)
    UsdGeom.Xformable(ground.GetPrim()).AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.01))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    bind_preview_material(stage, ground.GetPrim(), "/World/Looks/Ground", (0.12, 0.14, 0.18))

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(ARGS.physics_hz)
    physx_scene.CreateEnableCCDAttr().Set(True)
    physx_scene.CreateEnableStabilizationAttr().Set(True)
    physx_scene.CreateSolverTypeAttr().Set("TGS")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(300.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.95, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr().Set(1200.0)
    sun.CreateAngleAttr().Set(1.5)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 25.0))

    camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
    camera.CreateFocalLengthAttr().Set(55.0)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    # USD's default near clip is 1 stage unit.  That clips an entire centimeter-
    # scale asset viewed from 20-30 cm away, leaving only the clear color.
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 10.0))
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(
        look_at_matrix((0.16, -0.18, 0.12), (0.0, 0.0, 0.035))
    )

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.unlink(missing_ok=True)
    stage.GetRootLayer().Export(str(scene_path))
    return stage, cube_prim


def array_first(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array[0].astype(float)


def split_velocities(value: object) -> tuple[np.ndarray, np.ndarray]:
    """Normalize Isaac's (linear, angular) velocity result across backends."""
    if isinstance(value, tuple) and len(value) == 2:
        return array_first(value[0]), array_first(value[1])
    combined = array_first(value).reshape(-1)
    if combined.size != 6:
        raise RuntimeError(f"Unexpected rigid-body velocity shape: {combined.shape}")
    return combined[:3], combined[3:]


def read_source_palette(root: Usd.Prim) -> tuple[dict[str, dict[str, object]], bool]:
    """Read the composed Preview Surface palette and compare it to the source blend."""
    preview_by_material = {}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr() and str(shader.GetIdAttr().Get()) == "UsdPreviewSurface":
            preview_by_material[prim.GetParent().GetName()] = shader

    palette: dict[str, dict[str, object]] = {}
    all_match = True
    for material_name, expected in EXPECTED_MATERIALS.items():
        shader = preview_by_material.get(material_name)
        diffuse_value = shader.GetInput("diffuseColor").Get() if shader else None
        metallic_value = shader.GetInput("metallic").Get() if shader else None
        diffuse = [float(value) for value in diffuse_value] if diffuse_value is not None else None
        metallic = float(metallic_value) if metallic_value is not None else None
        matches = bool(
            diffuse is not None
            and metallic is not None
            and all(
                math.isclose(actual, target, abs_tol=1e-5)
                for actual, target in zip(diffuse, expected["diffuse_color"])
            )
            and math.isclose(metallic, expected["metallic"], abs_tol=1e-5)
        )
        palette[material_name] = {
            "diffuse_color_linear": diffuse,
            "metallic": metallic,
            "matches_source": matches,
        }
        all_match = all_match and matches
    return palette, all_match


def run_validation() -> dict:
    report: dict = {
        "schema_version": 1,
        "visual_usd": str(VISUAL_USD),
        "package_dir": str(PACKAGE_DIR),
        "checks": {},
    }
    packaged_visual = copy_visual_payload()
    wrapper_path, bounds = build_sim_ready_wrapper(packaged_visual)
    report["asset"] = {
        "wrapper": str(wrapper_path),
        "visual": str(packaged_visual),
        "mass_kg": MASS_KG,
        "bounds": bounds,
    }

    wrapper_stage = Usd.Stage.Open(str(wrapper_path))
    wrapper_root = wrapper_stage.GetDefaultPrim() if wrapper_stage else Usd.Prim()
    collision_prim = wrapper_stage.GetPrimAtPath("/FidgetCube/Collision") if wrapper_stage else Usd.Prim()
    add_check(report, "package_opens", bool(wrapper_stage and wrapper_root), str(wrapper_path))
    add_check(report, "package_default_prim", bool(wrapper_root) and str(wrapper_root.GetPath()) == "/FidgetCube", str(wrapper_root.GetPath()) if wrapper_root else None)
    add_check(report, "rigid_body_api", bool(wrapper_root) and wrapper_root.HasAPI(UsdPhysics.RigidBodyAPI), str(wrapper_root.GetPath()) if wrapper_root else None)
    add_check(report, "mass_api", bool(wrapper_root) and wrapper_root.HasAPI(UsdPhysics.MassAPI), MASS_KG)
    add_check(report, "collision_api", bool(collision_prim) and collision_prim.HasAPI(UsdPhysics.CollisionAPI), str(collision_prim.GetPath()) if collision_prim else None)

    switched = SimulationManager.switch_physics_engine("physx", verbose=True)
    active_engine = SimulationManager.get_active_physics_engine()
    add_check(report, "physics_engine", switched and active_engine == "physx", {"switched": switched, "active": active_engine})

    validation_scene_path = REPORT_PATH.parent / "validation_scene.usda"
    stage, cube_prim = build_validation_stage(wrapper_path, validation_scene_path)
    add_check(report, "isaac_reference", bool(cube_prim and cube_prim.IsValid()), str(cube_prim.GetPath()))
    add_check(report, "visual_geometry_composed", len([p for p in Usd.PrimRange(cube_prim) if p.IsA(UsdGeom.Mesh)]) == 9, [str(p.GetPath()) for p in Usd.PrimRange(cube_prim) if p.IsA(UsdGeom.Mesh)])
    add_check(report, "preview_materials_composed", len([p for p in Usd.PrimRange(cube_prim) if p.IsA(UsdShade.Material)]) >= 3, [str(p.GetPath()) for p in Usd.PrimRange(cube_prim) if p.IsA(UsdShade.Material)])
    material_palette, palette_matches = read_source_palette(cube_prim)
    report["asset"]["material_palette"] = material_palette
    add_check(report, "source_material_palette_composed", palette_matches, material_palette)

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/enabled", True)
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200.0)
    settings.set("/rtx/post/tonemap/whitepoint", 6500.0)
    settings.set("/rtx/post/aa/op", 3)

    # Create and attach the render graph before physics starts.  Replicator needs
    # several rendered frames to compile and populate its render variables.
    render_product = rep.create.render_product("/World/RenderCamera", (ARGS.width, ARGS.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])

    simulation_context = SimulationContext(
        physics_dt=1.0 / ARGS.physics_hz,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
        physics_prim_path="/World/PhysicsScene",
        set_defaults=False,
        backend="numpy",
        stage=stage,
    )
    rigid = RigidPrim("/World/FidgetCube")
    simulation_context.initialize_physics()
    simulation_context.play()
    simulation_context.step(render=True)
    initial_position = array_first(rigid.get_world_poses()[0])

    steps = int(round(ARGS.duration * ARGS.physics_hz))
    render_interval = max(1, ARGS.physics_hz // 60)
    for step in range(steps):
        simulation_context.step(render=False)
        if step % render_interval == 0:
            simulation_context.render()

    final_position = array_first(rigid.get_world_poses()[0])
    linear_velocity, angular_velocity = split_velocities(rigid.get_velocities())
    linear_speed = float(np.linalg.norm(linear_velocity))
    angular_speed_degrees = float(np.linalg.norm(angular_velocity))
    simulation_context.pause()

    report["physics"] = {
        "physics_hz": ARGS.physics_hz,
        "duration_s": ARGS.duration,
        "steps": steps + 1,
        "initial_position_m": initial_position.tolist(),
        "final_position_m": final_position.tolist(),
        "linear_velocity_mps": linear_velocity.tolist(),
        "angular_velocity_degrees_per_s": angular_velocity.tolist(),
        "linear_speed_mps": linear_speed,
        "angular_speed_degrees_per_s": angular_speed_degrees,
    }
    final_bottom_z = float(final_position[2] + bounds["minimum_m"][2])
    final_top_z = float(final_position[2] + bounds["maximum_m"][2])
    add_check(report, "drop_motion", float(initial_position[2] - final_position[2]) > 0.07, report["physics"])
    add_check(
        report,
        "rests_on_ground",
        -0.002 <= final_bottom_z <= 0.005 and final_top_z > 0.035,
        {"root_z_m": float(final_position[2]), "bottom_z_m": final_bottom_z, "top_z_m": final_top_z},
    )
    add_check(report, "settled_linear_speed", linear_speed < 0.08, linear_speed)
    add_check(report, "settled_angular_speed", angular_speed_degrees < 15.0, angular_speed_degrees)

    # Keep the settled physics pose and let Kit/Replicator finish compiling the
    # capture graph.  Reading the first frame immediately after attachment can
    # return the viewport clear color even when the camera is correct.
    for _ in range(64):
        simulation_app.update()
    image_data = None
    for _ in range(4):
        rep.orchestrator.step(pause_timeline=False)
        simulation_app.update()
        image_data = annotator.get_data()
    if isinstance(image_data, dict):
        image_data = image_data.get("data", image_data.get("image"))
    rgba = np.asarray(image_data)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise RuntimeError(f"Replicator returned an invalid RGB frame: shape={rgba.shape}")
    rgb = rgba[:, :, :3].astype(np.uint8)
    RENDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(RENDER_PATH, optimize=False)
    crop = rgb[rgb.shape[0] // 4 : 3 * rgb.shape[0] // 4, rgb.shape[1] // 4 : 3 * rgb.shape[1] // 4]
    render_stats = {
        "path": str(RENDER_PATH),
        "bytes": RENDER_PATH.stat().st_size,
        "shape": list(rgb.shape),
        "mean_rgb": float(rgb.mean()),
        "max_rgb": int(rgb.max()),
        "std_rgb": float(rgb.std()),
        "center_crop_std_rgb": float(crop.std()),
    }
    report["render"] = render_stats
    add_check(report, "render_not_black", render_stats["mean_rgb"] > 30.0 and render_stats["max_rgb"] > 200, render_stats)
    add_check(report, "render_not_overexposed", render_stats["mean_rgb"] < 220.0, render_stats["mean_rgb"])
    add_check(report, "render_has_detail", render_stats["center_crop_std_rgb"] > 12.0, render_stats["center_crop_std_rgb"])
    add_check(report, "render_file_size", render_stats["bytes"] > 150_000, render_stats["bytes"])

    annotator.detach([render_product])
    simulation_context.stop()
    report["passed"] = all(item["passed"] for item in report["checks"].values())
    if report["passed"]:
        write_asset_manifest(report)
    return report


def main() -> None:
    report: dict
    try:
        report = run_validation()
    except Exception as error:
        report = {
            "schema_version": 1,
            "visual_usd": str(VISUAL_USD),
            "package_dir": str(PACKAGE_DIR),
            "passed": False,
            "fatal_error": f"{type(error).__name__}: {error}",
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    else:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"FIDGET_ISAAC_VALIDATION={json.dumps({'passed': report['passed'], 'report': str(REPORT_PATH)}, sort_keys=True)}")
        if not report["passed"]:
            sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
