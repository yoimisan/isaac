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

The default rig contains three RGB cameras. `wrist` is parented to
`/World/Franka/panda_hand` and therefore follows the gripper. The two fixed
external cameras observe the same workspace center from symmetric `+X/+Y` and
`+X/-Y` positions. Adding or removing camera configs does not change the
recorder.

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

Replay operates on staging episodes before export. The LeRobot exporter is a
deterministic format conversion and already reloads and decodes its output as
its own validation step.

## Collected features

- `observation.state`: Franka's nine joint positions in `franka.dof_names`
  order.
- `action`: the nine-dimensional position target read from
  `articulation_controller.get_applied_action()` after all task and gripper
  control calls. This is deliberately not the seven-dimensional CuRobo action
  returned by some PnP states.
- `observation.images.wrist`: moving wrist RGB, `uint8`, HWC.
- `observation.images.external_pos_y`: fixed `+X/+Y` RGB, `uint8`, HWC.
- `observation.images.external_neg_y`: fixed `+X/-Y` RGB, `uint8`, HWC.
- `next.done`: true on the last frame of every exported episode.
- `next.success`: true on the last frame of a successful episode.
- `task`: the natural-language task instruction used by LeRobot's task index.

Staging also stores `task_state` and, for newly collected episodes,
`scene_pose`. The latter is a `[frame, object, 7]` array containing world-frame
`[x, y, z, qw, qx, qy, qz]` poses for the objects named by
`dataset.json.replay_objects`. These fields are replay diagnostics and are not
currently exported as LeRobot training features.

By default, only successful episodes are exported. Failed or interrupted
episodes remain useful diagnostics and can be included explicitly.

## Collect staging episodes

Launch through the usual Isaac Sim Python entry point and add `--record`:

```bash
python.sh src/pnp.py \
  --record \
  --record-root logs/data_collection/pnp_raw \
  --record-episodes 10 \
  --record-fps 60
```

Clean collection is the default. To collect state-aware perturbed rollouts,
enable the Naughty Ghost explicitly and use a separate staging root:

```bash
python.sh src/pnp.py \
  --record \
  --perturb \
  --perturb-seed 0 \
  --perturb-min-attacks 0 \
  --perturb-max-attacks 5 \
  --record-root logs/data_collection/pnp_perturbed_raw \
  --record-episodes 10
```

The staging schema records `collection_mode` and the perturbation seed and
attack-count range. It rejects attempts to append clean and perturbed episodes
to the same root, preventing accidental contamination of the clean dataset.
The current metadata describes the configured policy; individual disturbance
events remain visible in the Isaac log but are not yet stored frame-by-frame.

The default resolution is 640 by 480. Override it with
`--camera-width` and `--camera-height`. An episode ends only after `ReturnState`
confirms that the robot has reached the joint pose captured at reset.
Successful completion saves the episode and
automatically resets the world for the next rollout. Collection exits after
`--record-episodes` successful episodes; its default is one. This completion
condition deliberately does not depend on an additional `task.is_done()` check.

Manual stop/start resets discard an incomplete episode unless
`--save-failed-episodes` is set. With that flag, application shutdown also
publishes a non-empty active episode with `next.success=false`.

The camera is created after the first world reset, warmed up before controller
execution, and then kept alive across ordinary world resets. RGB is copied from
the Warp annotator buffer immediately so later renders cannot mutate a queued
frame. If RGB becomes unavailable after warm-up, recording fails explicitly
instead of silently creating a variable-rate episode.

The default camera poses are:

| Feature suffix | Pose | Optical setup |
|---|---|---|
| `wrist` | parent-relative `(0.06, 0.0, 0.035)` on `panda_hand` | 18 mm, looks along hand `+Z` |
| `external_pos_y` | world `(0.95, 0.55, 0.75)` | 28 mm, looks at `(0.48, 0.0, 0.08)` |
| `external_neg_y` | world `(0.95, -0.55, 0.75)` | 28 mm, looks at `(0.48, 0.0, 0.08)` |

Optical lengths are stored in meters because the application creates a
meter-unit stage: 18 mm is authored as `0.018` and 28 mm as `0.028`. The
horizontal aperture is 36 mm (`0.036`), yielding approximately 90 degrees of
horizontal field of view for the wrist camera and 65 degrees externally.

Camera definitions are part of the staging schema. Use a fresh
`--record-root` after changing this rig rather than appending three-camera
episodes to an older one-camera staging dataset.

Collection forces DLSS Quality mode (`rtx/post/dlss/execMode = 2`), matching
Isaac Sim's SDG examples. At 640 by 480 this keeps DLSS's internal render input
above its minimum dimension and avoids the low-resolution warning produced by
the previous 320 by 240 Performance-mode setup. Four PNG workers and a bounded
128-write queue absorb the additional three-camera encoding load.

Both collection and replay use explicit RayTracedLighting, a 100-intensity dome
light, a 500-intensity distant light, and ACES tone mapping with ISO 100. Keeping
these settings in the shared task/camera setup prevents replay comparisons from
depending on implicit Kit lighting defaults.

The first implementation requires `--record-fps` to equal the physics/control
rate. Recording at 30 Hz while the task controller emits different commands at
60 Hz would silently discard intermediate actions. Lower-rate datasets should
later use an explicit action-chunk representation instead of frame decimation.

## Validate and replay an episode

Run the integrity pass without launching Isaac Sim:

```bash
python.sh src/replay.py \
  --dataset-root logs/data_collection/pnp_raw \
  --episode 0 \
  --validate-only
```

It checks the declared frame count, array shapes, finite values, 60 Hz timestamp
spacing, scene quaternion normalization, every expected camera filename, PNG
decodability and resolution, and representative non-black RGB values. The JSON
report is written beside the episode as `replay-report-action.json` unless
`--report` is supplied.

There are two simulator replay modes:

- `--mode state` directly writes every recorded joint and object pose. Use it
  for visual inspection and synchronized RGB comparison. It does not test
  whether the recorded actions caused the motion.
- `--mode action` sets frame zero, then applies `action[i]` before the physics
  transition to `observation_state[i + 1]`. The report contains joint RMSE and
  maximum absolute error, making this the control-semantic validation mode.

Launch a real-time visual state replay:

```bash
python.sh src/replay.py \
  --dataset-root logs/data_collection/pnp_raw \
  --episode 0 \
  --mode state
```

Run action validation headlessly and as fast as possible:

```bash
python.sh src/replay.py \
  --dataset-root logs/data_collection/pnp_raw \
  --episode 0 \
  --headless \
  --mode action \
  --scene-mode initial \
  --playback-speed 0
```

`--scene-mode initial` sets replay objects only at frame zero and then lets
physics evolve, which is the strict choice for clean trajectories.
`--scene-mode trace` corrects replay objects to their recorded poses every
frame. Trace mode reproduces exogenous object motion in perturbed episodes and
supports visual validation, but the report explicitly notes that this masks
object-physics drift. RGB MAE, RMSE, PSNR, and the worst frame index are
reported per camera; use `--image-stride` to compare a subset or
`--no-image-compare` for joint-only validation.

Episodes collected before scene-pose recording remain readable. They receive a
warning and can validate robot actions, timing, arrays and existing RGB files,
but cannot reproduce the original randomized cube/target poses. Collect new
episodes under a fresh `--record-root`; the staging schema deliberately rejects
mixing old robot-only episodes with replay-complete episodes.

The shared runtime in `data_collection.replay_runtime` receives generic world,
robot and object handles. PnP-specific scene construction lives in
`pick_place.replay`; a future task adds its own adapter rather than changing the
replay engine or task controller.

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
videos/observation.images.wrist/chunk-000/file-000.mp4
videos/observation.images.external_pos_y/chunk-000/file-000.mp4
videos/observation.images.external_neg_y/chunk-000/file-000.mp4
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

The Naughty Ghost is disabled unless `--perturb` is supplied, so the same
runtime can collect clean and perturbed rollouts without maintaining two task
controller implementations.

`CameraSensor` currently exposes teardown through object destruction rather
than a public `destroy()` method, so the camera rig drops its last sensor
references during shutdown. This lifecycle should be regression-tested in the
actual Isaac runtime when repeated in-process task creation is introduced.
