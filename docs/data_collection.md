# Data collection and LeRobot v3.0 export

## Design boundary

Data collection is a sidecar to task execution. It does not modify
`PnPController`, its state objects, CuRobo plans, or recovery decisions. The
application loop only supplies four lifecycle events:

1. create the camera and recorder after the world is initialized;
2. begin or abort an episode when the simulation resets;
3. sample the observation and the already-applied action at a fixed rate;
4. finish a successful episode and close the recorder at shutdown.

The RGB camera uses Isaac Sim 6's `RtxCamera` and `CameraSensor`. The existing
`World.step(render=True)` drives its render product; the collector must not call
`rep.orchestrator.step()`, because that would add a second simulation/render
advance to the task loop.

The first camera is a fixed `overview` camera. The camera rig already accepts a
tuple of camera configs, so a wrist or side camera can be added without changing
the recorder.

## Why export in a separate process

Isaac Sim 6.0.1 pins NumPy 2.3.1. The official LeRobot 0.6.0 package produces
LeRobot dataset format v3.0 but requires NumPy below 2.3. Installing it into the
Isaac/CuRobo Python environment would therefore replace a core simulation
dependency.

Collection is split into two explicit stages:

```text
Isaac Sim + CuRobo
    -> logs/data_collection/pnp_raw (staging v1)
    -> isolated LeRobot exporter
    -> LeRobot v3.0 (Parquet + MP4 + metadata)
```

The staging data preserves simulation timestamps and PnP state names for
diagnostics. LeRobot timestamps are intentionally regenerated as
`frame_index / fps`, as required by its fixed-rate dataset contract.

## Collected features

- `observation.state`: Franka's nine joint positions in `franka.dof_names`
  order.
- `action`: the nine-dimensional position target read from
  `articulation_controller.get_applied_action()` after all task and gripper
  control calls. This is deliberately not the seven-dimensional CuRobo action
  returned by some PnP states.
- `observation.images.overview`: RGB `uint8`, HWC.
- `next.done`: true on the last frame of every exported episode.
- `next.success`: true on the last frame of a successful episode.
- `task`: the natural-language task instruction used by LeRobot's task index.

By default, only successful episodes are exported. Failed or interrupted
episodes remain useful diagnostics and can be included explicitly.

## Collect staging episodes

Launch through the usual Isaac Sim Python entry point and add `--record`:

```bash
python.sh src/pnp.py \
  --record \
  --record-root logs/data_collection/pnp_raw \
  --record-fps 60
```

The default resolution is 320 by 240. Override it with
`--camera-width` and `--camera-height`. A clean PnP completion publishes an
episode. Stop/start resets discard an incomplete episode unless
`--save-failed-episodes` is set. With that flag, application shutdown also
publishes a non-empty active episode with `next.success=false`.

The camera is created after the first world reset, warmed up before controller
execution, and then kept alive across ordinary world resets. RGB is copied from
the Warp annotator buffer immediately so later renders cannot mutate a queued
frame. If RGB becomes unavailable after warm-up, recording fails explicitly
instead of silently creating a variable-rate episode.

The first implementation requires `--record-fps` to equal the physics/control
rate. Recording at 30 Hz while the task controller emits different commands at
60 Hz would silently discard intermediate actions. Lower-rate datasets should
later use an explicit action-chunk representation instead of frame decimation.

## Export LeRobot v3.0

Create an isolated Python 3.12 environment outside Isaac Sim:

```bash
uv venv --python /usr/bin/python3.12 .venv-lerobot
uv pip install \
  --python .venv-lerobot/bin/python \
  -r requirements/lerobot-v30.txt
.venv-lerobot/bin/python tools/export_lerobot_v30.py \
  --input-root logs/data_collection/pnp_raw \
  --output-root logs/data_collection/pnp_lerobot_v30 \
  --repo-id local/isaac-pnp
```

The exporter refuses to overwrite an existing output directory. It builds in a
temporary sibling directory, finalizes and reloads the result with
`LeRobotDataset`, decodes the first and last camera frame of every episode, and
only then atomically publishes the requested output directory. A failed export
is removed without disguising the original error as a cleanup error.

Expected output:

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.overview/chunk-000/file-000.mp4
```

Use `--images` only for debugging when embedding images in Parquet is preferred
over MP4. Use `--include-failed` to retain explicitly saved failed episodes.
Only one process may write a staging root at once. Assign a distinct
`--record-root` to every parallel collection worker; a writer lock rejects
accidental sharing.

## Merge contract with other runtime features

Naughty Ghost and other future systems may mutate the world before the task
controller runs. Data collection should remain last in the control-decision
portion of the loop: determine/apply all commands first, read the full applied
robot action, record the current observation/action pair, and finally call
`world.step(render=True)`. It may read a public task state for staging metadata,
but it must not write task state or own recovery behavior.

The intended merge order in the application loop is:

```text
reset: recorder finish/abort -> world reset/camera settle
       -> task controller reset -> ghost reset -> recorder begin
tick:  ghost step -> task controller forward/apply
       -> recorder sample -> world step(render=True)
```

The current `_ENABLE_RECOVERY_TEST_PERTURBATION` block is only an earlier manual
recovery probe. Remove it when the Naughty Ghost branch is integrated instead
of enabling both perturbation paths.

`CameraSensor` currently exposes teardown through object destruction rather
than a public `destroy()` method, so the camera rig drops its last sensor
references during shutdown. This lifecycle should be regression-tested in the
actual Isaac runtime when repeated in-process task creation is introduced.
