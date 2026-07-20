#!/usr/bin/env python3
"""Export the BlenderKit Fidget Cube as a self-contained visual USD asset.

Run this script with Blender, not with the system Python::

    blender --background --python export_fidget_cube.py -- \
        --input /path/to/fidget-cube.blend \
        --output-dir /path/to/work/export

The source .blend is opened read-only and is never saved.  The exporter writes a
visual-only USD plus an export manifest.  Physics is deliberately added later by
``load_fidget_cube.py`` so Blender conversion and simulator authoring remain
separate, observable stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


EXPECTED_ROOT_NAME = "Fidget Cube"
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
OUTPUT_FILENAME = "fidget_cube_visual.usdc"
RGB_NODE_MATERIALS = {"Color_mat", "Cube_Color_mat"}


def parse_args() -> argparse.Namespace:
    """Parse arguments placed after Blender's ``--`` separator."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source Fidget Cube .blend file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for USD, textures, and manifest")
    parser.add_argument(
        "--allow-mesh-set-change",
        action="store_true",
        help="Allow source mesh names to differ from the known BlenderKit asset",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descendants(root: bpy.types.Object) -> list[bpy.types.Object]:
    result: list[bpy.types.Object] = []

    def visit(obj: bpy.types.Object) -> None:
        result.append(obj)
        for child in sorted(obj.children, key=lambda item: item.name):
            visit(child)

    visit(root)
    return result


def evaluated_world_bounds(meshes: list[bpy.types.Object]) -> tuple[list[float], list[float]]:
    """Return render-evaluated world-space bounds in meters."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if not points:
        raise RuntimeError("The selected asset has no mesh bounds")
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return minimum, maximum


def id_property_dict(value: object) -> dict[str, object]:
    """Convert a Blender IDPropertyGroup to a normal dictionary when possible."""
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def bake_constant_rgb_base_colors(materials: list[bpy.types.Material]) -> dict[str, dict[str, object]]:
    """Bake Blender RGB-node colors into Principled inputs for USD export.

    The source asset exposes its body and accent colors through ShaderNodeRGB
    nodes.  Blender's USD Preview Surface conversion does not preserve those
    constant-node links and otherwise exports both materials as 50% gray.
    Baking only the constant input keeps the source-authored palette while
    leaving roughness, normal, and metallic networks untouched.
    """
    palette: dict[str, dict[str, object]] = {}
    baked_materials: set[str] = set()

    for material in sorted(materials, key=lambda item: item.name):
        if material.node_tree is None:
            raise RuntimeError(f"Material has no node tree: {material.name}")
        principled_nodes = [
            node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
        ]
        if len(principled_nodes) != 1:
            raise RuntimeError(
                f"Expected one Principled BSDF in {material.name}, found {len(principled_nodes)}"
            )
        base_color = principled_nodes[0].inputs.get("Base Color")
        if base_color is None:
            raise RuntimeError(f"Principled BSDF has no Base Color input: {material.name}")

        source = "principled_default"
        links = list(base_color.links)
        if links:
            if len(links) != 1 or links[0].from_node.bl_idname != "ShaderNodeRGB":
                raise RuntimeError(
                    f"Unsupported linked Base Color in {material.name}: "
                    f"{[link.from_node.bl_idname for link in links]}"
                )
            link = links[0]
            color = tuple(float(value) for value in link.from_socket.default_value)
            material.node_tree.links.remove(link)
            base_color.default_value = color
            material.diffuse_color = color
            source = "baked_shader_rgb"
            baked_materials.add(material.name)
        else:
            color = tuple(float(value) for value in base_color.default_value)

        palette[material.name] = {"linear_rgba": list(color), "source": source}

    if baked_materials != RGB_NODE_MATERIALS:
        raise RuntimeError(
            "Unexpected RGB-node material set; "
            f"expected={sorted(RGB_NODE_MATERIALS)}, found={sorted(baked_materials)}"
        )

    accent = palette["Color_mat"]["linear_rgba"]
    body = palette["Cube_Color_mat"]["linear_rgba"]
    if not (accent[1] > accent[0] * 2.0 and accent[1] > accent[2] + 0.05):
        raise RuntimeError(f"Source accent material is no longer green: {accent}")
    if max(body[:3]) - min(body[:3]) > 1e-4 or max(body[:3]) > 0.1:
        raise RuntimeError(f"Source body material is no longer dark neutral gray: {body}")
    return palette


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output = output_dir / OUTPUT_FILENAME
    manifest_path = output_dir / "export_manifest.json"

    if not source.is_file():
        raise FileNotFoundError(f"Source blend file does not exist: {source}")
    if source.suffix.lower() != ".blend":
        raise ValueError(f"Expected a .blend source, got: {source}")

    if Path(bpy.data.filepath).resolve() != source:
        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)

    root = bpy.data.objects.get(EXPECTED_ROOT_NAME)
    if root is None:
        raise RuntimeError(f"Expected root object {EXPECTED_ROOT_NAME!r} was not found")
    if root.parent is not None or root.type != "EMPTY":
        raise RuntimeError("Fidget Cube root must be an unparented Blender Empty")

    hierarchy = descendants(root)
    meshes = [obj for obj in hierarchy if obj.type == "MESH"]
    mesh_names = {obj.name for obj in meshes}
    if mesh_names != EXPECTED_MESH_NAMES and not args.allow_mesh_set_change:
        missing = sorted(EXPECTED_MESH_NAMES - mesh_names)
        unexpected = sorted(mesh_names - EXPECTED_MESH_NAMES)
        raise RuntimeError(f"Unexpected source hierarchy; missing={missing}, unexpected={unexpected}")
    if any(obj.scale[:] != (1.0, 1.0, 1.0) for obj in hierarchy):
        raise RuntimeError("The source contains non-unit object scale; refusing an ambiguous export")

    scene = bpy.context.scene
    if scene.unit_settings.system != "METRIC" or scene.unit_settings.scale_length != 1.0:
        raise RuntimeError(
            "Expected a metric Blender scene with scale_length=1.0; "
            f"got system={scene.unit_settings.system!r}, scale_length={scene.unit_settings.scale_length}"
        )

    minimum, maximum = evaluated_world_bounds(meshes)
    dimensions = [maximum[i] - minimum[i] for i in range(3)]
    if any(size < 0.035 or size > 0.060 for size in dimensions):
        raise RuntimeError(f"Fidget Cube dimensions are outside the expected 3.5-6.0 cm range: {dimensions}")

    material_objects = list(
        {slot.material for obj in meshes for slot in obj.material_slots if slot.material}
    )
    materials = sorted(material.name for material in material_objects)
    if len(materials) < 3:
        raise RuntimeError(f"Expected at least three materials, found: {materials}")
    material_palette = bake_constant_rgb_base_colors(material_objects)

    image_info = []
    for image in bpy.data.images:
        if image.source == "VIEWER":
            continue
        image_info.append(
            {
                "name": image.name,
                "packed": bool(image.packed_file),
                "size": [int(image.size[0]), int(image.size[1])],
            }
        )

    asset_data = id_property_dict(root.get("asset_data", {}))

    # Use a stable USD hierarchy and omit BlenderKit's large custom-property payload.
    root.name = "Geometry"
    bpy.ops.object.select_all(action="DESELECT")
    for obj in hierarchy:
        obj.select_set(True)
        obj.hide_render = False
    bpy.context.view_layer.objects.active = root

    output_dir.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.usd_export(
        filepath=str(output),
        check_existing=False,
        selected_objects_only=True,
        export_animation=False,
        export_hair=False,
        export_uvmaps=True,
        rename_uvmaps=True,
        export_mesh_colors=True,
        export_normals=True,
        export_materials=True,
        export_subdivision="BEST_MATCH",
        export_armatures=False,
        only_deform_bones=False,
        export_shapekeys=False,
        use_instancing=False,
        evaluation_mode="RENDER",
        generate_preview_surface=True,
        generate_materialx_network=False,
        convert_orientation=False,
        export_textures_mode="NEW",
        overwrite_textures=True,
        relative_paths=True,
        xform_op_mode="TRS",
        root_prim_path="/FidgetCube",
        export_custom_properties=False,
        author_blender_name=True,
        convert_world_material=False,
        export_meshes=True,
        export_lights=False,
        export_cameras=False,
        export_curves=False,
        export_points=False,
        export_volumes=False,
        triangulate_meshes=False,
        merge_parent_xform=False,
        convert_scene_units="METERS",
        meters_per_unit=1.0,
    )
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError(f"Blender USD export failed: result={result}, output={output}")

    texture_files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path != output and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "asset": {
            "name": "Fidget Cube",
            "source": "BlenderKit",
            "source_asset_id": asset_data.get("id", "483121ec-975c-49df-9633-4b7deb13eadd"),
            "source_asset_base_id": asset_data.get("assetBaseId", "e1548130-3cef-4cbd-bd4a-0b5b43b4f020"),
            "author": (asset_data.get("author") or {}).get("firstName", "Klo")
            + " "
            + (asset_data.get("author") or {}).get("lastName", "Works"),
            "license": asset_data.get("license", "royalty_free"),
            "source_url": "https://www.blenderkit.com/asset-gallery-detail/e1548130-3cef-4cbd-bd4a-0b5b43b4f020/",
        },
        "source": {
            "filename": source.name,
            "sha256": sha256(source),
        },
        "export": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "blender_version": bpy.app.version_string,
            "usd": output.name,
            "usd_sha256": sha256(output),
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "root_prim": "/FidgetCube",
            "bounds_min_m": minimum,
            "bounds_max_m": maximum,
            "dimensions_m": dimensions,
            "mesh_names": sorted(mesh_names),
            "materials": materials,
            "material_palette": material_palette,
            "material_conversion": "constant ShaderNodeRGB inputs baked into Principled Base Color",
            "images": image_info,
            "files": [output.name, *texture_files],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FIDGET_EXPORT_OK={json.dumps(manifest['export'], sort_keys=True)}")


if __name__ == "__main__":
    main()
