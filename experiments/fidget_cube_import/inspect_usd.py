#!/usr/bin/env python3
"""Offline structural validator for the Blender-exported Fidget Cube USD.

This script only needs the Isaac Sim Python environment's ``pxr`` package; it
does not launch Kit or a renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdShade


EXPECTED_MESH_NAMES = {
    "analog",
    "ball",
    "button",
    "click",
    "cube",
    "gears",
    "gears1",
    "gears2",
    "turn",
}
EXPECTED_MATERIALS = {
    "Color_mat": {"diffuse_color": (0.020598998, 0.088981375, 0.0), "metallic": 0.0},
    "Cube_Color_mat": {"diffuse_color": (0.01960665, 0.01960665, 0.01960665), "metallic": 0.0},
    "Metal_mat": {"diffuse_color": (0.5, 0.5, 0.5), "metallic": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True, help="Blender-exported visual USD")
    parser.add_argument("--report", type=Path, required=True, help="Path for the JSON inspection report")
    return parser.parse_args()


def check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report["checks"][name] = {"passed": bool(passed), "detail": detail}


def asset_paths(value: Any) -> list[Sdf.AssetPath]:
    if isinstance(value, Sdf.AssetPath):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Sdf.AssetPath)]
    try:
        return [item for item in value if isinstance(item, Sdf.AssetPath)]
    except TypeError:
        return []


def resolve_dependency(layer_dir: Path, asset_path: Sdf.AssetPath) -> tuple[str, str, bool]:
    authored = asset_path.path
    resolved = asset_path.resolvedPath
    if resolved:
        candidate = Path(resolved)
    elif authored and not authored.startswith(("http://", "https://", "omniverse://")):
        candidate = Path(authored)
        if not candidate.is_absolute():
            candidate = layer_dir / candidate
    else:
        return authored, resolved, bool(resolved)
    candidate = candidate.resolve()
    return authored, str(candidate), candidate.is_file()


def main() -> None:
    args = parse_args()
    asset = args.asset.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    report: dict[str, Any] = {"schema_version": 1, "asset": str(asset), "checks": {}}

    if not asset.is_file():
        raise FileNotFoundError(f"USD asset does not exist: {asset}")
    check(report, "file_size", asset.stat().st_size > 1024, {"bytes": asset.stat().st_size})

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open failed: {asset}")
    default_prim = stage.GetDefaultPrim()
    check(report, "default_prim", bool(default_prim), str(default_prim.GetPath()) if default_prim else None)
    check(
        report,
        "default_prim_path",
        bool(default_prim) and str(default_prim.GetPath()) == "/FidgetCube",
        str(default_prim.GetPath()) if default_prim else None,
    )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    check(report, "meters_per_unit", math.isclose(meters_per_unit, 1.0, abs_tol=1e-9), meters_per_unit)
    check(report, "up_axis", up_axis.upper() == "Z", up_axis)

    prims = list(stage.Traverse())
    meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
    # Blender exports each object as an Xform with an implementation-named Mesh
    # child (for example ``/Geometry/analog/Mesh_002``).  The stable asset object
    # name is therefore the parent Xform name, not the generated Mesh prim name.
    mesh_names = {prim.GetParent().GetName() for prim in meshes}
    materials = [prim for prim in prims if prim.IsA(UsdShade.Material)]
    shaders = [UsdShade.Shader(prim) for prim in prims if prim.IsA(UsdShade.Shader)]
    preview_shaders = [
        shader
        for shader in shaders
        if shader.GetIdAttr() and str(shader.GetIdAttr().Get()) == "UsdPreviewSurface"
    ]
    cameras = [prim for prim in prims if prim.IsA(UsdGeom.Camera)]
    lights = [prim for prim in prims if prim.GetTypeName().endswith("Light")]

    check(
        report,
        "mesh_set",
        mesh_names == EXPECTED_MESH_NAMES,
        {"found": sorted(mesh_names), "expected": sorted(EXPECTED_MESH_NAMES)},
    )
    check(report, "materials", len(materials) >= 3, [str(prim.GetPath()) for prim in materials])
    check(
        report,
        "usd_preview_surface",
        len(preview_shaders) >= 3,
        [str(shader.GetPath()) for shader in preview_shaders],
    )

    palette = {}
    palette_matches = True
    preview_by_material = {
        shader.GetPrim().GetParent().GetName(): shader for shader in preview_shaders
    }
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
        palette_matches = palette_matches and matches
    check(report, "source_material_palette", palette_matches, palette)
    check(report, "no_cameras", not cameras, [str(prim.GetPath()) for prim in cameras])
    check(report, "no_lights", not lights, [str(prim.GetPath()) for prim in lights])

    binding_details = {}
    bound_count = 0
    for mesh in meshes:
        material, relationship = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        path = str(material.GetPath()) if material else None
        binding_details[str(mesh.GetPath())] = path
        bound_count += int(bool(material and relationship))
    check(report, "material_bindings", bound_count == len(meshes), binding_details)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    minimum = [float(value) for value in aligned.GetMin()]
    maximum = [float(value) for value in aligned.GetMax()]
    dimensions = [(maximum[i] - minimum[i]) * meters_per_unit for i in range(3)]
    bbox_valid = all(math.isfinite(value) for value in [*minimum, *maximum])
    size_valid = bbox_valid and all(0.035 <= size <= 0.060 for size in dimensions)
    check(
        report,
        "world_bounds",
        size_valid,
        {"minimum_stage_units": minimum, "maximum_stage_units": maximum, "dimensions_m": dimensions},
    )

    dependencies = []
    for prim in prims:
        for attribute in prim.GetAttributes():
            if not attribute.HasAuthoredValueOpinion():
                continue
            for dependency in asset_paths(attribute.Get()):
                authored, resolved, exists = resolve_dependency(asset.parent, dependency)
                dependencies.append(
                    {
                        "attribute": str(attribute.GetPath()),
                        "authored": authored,
                        "resolved": resolved,
                        "exists": exists,
                    }
                )
    unresolved = [item for item in dependencies if item["authored"] and not item["exists"]]
    texture_dependencies = [
        item
        for item in dependencies
        if Path(item["authored"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff"}
    ]
    check(report, "resolved_dependencies", not unresolved, dependencies)
    check(report, "texture_dependency", bool(texture_dependencies), texture_dependencies)

    report["summary"] = {
        "prim_count": len(prims),
        "mesh_count": len(meshes),
        "material_count": len(materials),
        "preview_shader_count": len(preview_shaders),
        "dimensions_m": dimensions,
        "passed": all(item["passed"] for item in report["checks"].values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FIDGET_USD_INSPECTION={json.dumps(report['summary'], sort_keys=True)}")
    if not report["summary"]["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
