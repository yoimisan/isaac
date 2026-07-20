# Fidget Cube Blender → Isaac Sim experiment

This experiment converts the BlenderKit **Fidget Cube** into a self-contained,
sim-ready USD package and proves that it works in Isaac Sim 6.0.1.

Source and appearance reference: [Fidget Cube by Klo Works on Blendkit](https://www.blendkit.com/asset-gallery-detail/e1548130-3cef-4cbd-bd4a-0b5b43b4f020/).

The scripts accept paths as arguments; they do not bake a user home directory
into generated assets.

## 1. Export the visual USD with Blender

```bash
REPO_ROOT="$HOME/Documents/isaac-scenes-assets-importer"
EXPERIMENT_DIR="$REPO_ROOT/experiments/fidget_cube_import"
WORK_DIR="$REPO_ROOT/logs/asset_imports/fidget_cube"
ASSET_DIR="$REPO_ROOT/assets/fidget_cube"
SOURCE_BLEND="$HOME/blenderkit_data/models/fidget-cube_483121ec-975c-49df-9633-4b7deb13eadd/fidget-cube_5955b1b3-ad3e-4e13-9586-72d56821959f.blend"

blender --background --python "$EXPERIMENT_DIR/export_fidget_cube.py" -- \
  --input "$SOURCE_BLEND" \
  --output-dir "$WORK_DIR/export"
```

The source uses constant Blender RGB nodes for its dark-gray body and green
controls. The exporter bakes those source-authored values into the Principled
Base Color inputs before USD conversion so the asset does not become gray-white.
The metallic ball remains a neutral metallic Preview Surface.

## 2. Inspect the USD without launching Kit

```bash
ISAAC_PYTHON="$HOME/miniconda3/envs/isaacsim/bin/python"

"$ISAAC_PYTHON" "$EXPERIMENT_DIR/inspect_usd.py" \
  --asset "$WORK_DIR/export/fidget_cube_visual.usdc" \
  --report "$WORK_DIR/offline_inspection.json"
```

## 3. Package and validate in Isaac Sim

```bash
"$ISAAC_PYTHON" "$EXPERIMENT_DIR/load_fidget_cube.py" \
  --visual-usd "$WORK_DIR/export/fidget_cube_visual.usdc" \
  --package-dir "$ASSET_DIR" \
  --report "$WORK_DIR/isaac_validation.json" \
  --render "$WORK_DIR/fidget_cube_render.png" \
  --headless
```

The Isaac script:

- references the Blender visual layer without flattening it;
- adds an 80 g rigid body and a stable box collider in `fidget_cube.usda`;
- drops the asset onto a ground plane for 3.5 simulated seconds at 240 Hz;
- verifies live PhysX pose and velocity;
- checks that all nine meshes and at least three UsdPreviewSurface materials compose;
- captures and numerically validates a 1024×1024 RT2 render.

## Outputs

Intermediate exports, reports, and renders remain under the ignored `logs/`
tree. The complete validated package is written to:

```text
$HOME/Documents/isaac-scenes-assets-importer/assets/fidget_cube/
├── fidget_cube.usda
├── fidget_cube_visual.usdc
├── textures/
├── export_manifest.json
├── asset_manifest.json
└── SOURCE.md
```

The intended entry point for downstream tasks is `fidget_cube.usda`.
