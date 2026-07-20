# Fidget Cube import worklog

## Feature-to-skill mapping

| Capability | Project skill | Validation evidence |
|---|---|---|
| Blender source discovery and natural scale | `usd-pipeline` | Blender hierarchy inspection and `export_manifest.json` |
| USD materials, dependencies, default Prim, and bounds | `usd-pipeline` | `offline_inspection.json` |
| Rigid body, collision proxy, drop stability | `physics-simulation` | Physics section of `isaac_validation.json` |
| Headless RT2 visibility and lighting | `isaac-sim-rendering` | `fidget_cube_render.png` and render metrics |
| Final delivery gate | `isaac-sim-validator` | All report checks pass before promotion to `assets/` |

## Pipeline contract

1. `export_fidget_cube.py` opens the source `.blend` without saving it.
2. The Blender stage emits only visual geometry and UsdPreviewSurface materials.
3. `inspect_usd.py` is an offline gate and does not start Isaac Sim.
4. `load_fidget_cube.py` copies the visual payload into a package, authors physics in a separate root layer, and tests the package in isolation.
5. The package is written to `~/Documents/isaac-scenes-assets-importer/assets/fidget_cube` only after every structural, physics, material-palette, and rendering check passes.

## Asset-specific expectations

- Source asset: BlenderKit Fidget Cube by Klo Works.
- License: BlenderKit Royalty Free.
- Source scene unit: meters.
- Natural size: approximately 4.5 cm on each axis.
- Visual hierarchy: one root Empty and nine mesh children.
- Physics representation: 80 g rigid body with a box collision proxy.
- This asset is a single rigid object; its decorative buttons and gears are not imported as independent articulations.
- The dark-gray body and green controls come from source `ShaderNodeRGB` values. These constant values are baked into Principled Base Color inputs because Blender's USD converter otherwise emits default 50% gray.

## Status

- Source located and opened with Blender 5.2: complete.
- Export script: complete; emitted a self-contained visual USD and relative texture.
- Offline USD inspection: complete; all structure, scale, material, and dependency checks passed.
- Isaac Sim 6.0.1 package/load test: complete; all composition, PhysX, and RT2 render checks passed.
- Final promotion: complete at `~/Documents/isaac-scenes-assets-importer/assets/fidget_cube`.

## Validated result

- Default Prim: `/FidgetCube`.
- Visual geometry: 9 meshes and 3 bound UsdPreviewSurface materials with the source dark-gray, green, and metallic palette.
- Dimensions: `0.044894 × 0.045594 × 0.045665 m`.
- Physics: 80 g rigid body, box proxy collider, CCD enabled.
- Drop test: 3.5 seconds at 240 Hz on PhysX; the body settled on the ground with zero measured final velocity.
- Render: 1024×1024 RT2 capture, mean RGB about 129, max RGB 241, visibly dark-gray/green/silver, and visually reviewed against the Blendkit reference.
- Final package: every file matched the SHA-256 recorded by `asset_manifest.json`.

## Iteration notes

- Blender object names are exported on parent Xforms; generated Mesh prims use names such as `Mesh_002`.
- For centimeter-scale assets, the USD camera near clip must be explicitly reduced below the default 1 m. A 5 mm near clip was used here.
