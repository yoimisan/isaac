"""Isaac-side episode staging independent of the LeRobot Python stack."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from data_collection.config import DataCollectionConfig

if TYPE_CHECKING:
    from data_collection.camera import RgbCameraRig


_LOGGER = logging.getLogger(__name__)
_FORMAT_VERSION = "isaac-scenes-staging-v1"


class ArticulationFrameSource:
    """Read fixed-order joint state and the full applied position target."""

    def __init__(self, robot: Any, articulation_controller: Any) -> None:
        self._robot = robot
        self._articulation_controller = articulation_controller
        self._dof_names = tuple(robot.dof_names)
        if not self._dof_names:
            raise RuntimeError("The articulation has no resolved DOFs.")

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    @property
    def features(self) -> dict[str, dict]:
        names = {"motors": list(self._dof_names)}
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(self._dof_names)],
                "names": names,
            },
            "action": {
                "dtype": "float32",
                "shape": [len(self._dof_names)],
                "names": names,
            },
        }

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        joint_state = self._robot.get_joints_state()
        if joint_state is None or joint_state.positions is None:
            raise RuntimeError("Failed to read articulation joint positions.")
        positions = np.asarray(joint_state.positions, dtype=np.float32).copy()
        if positions.shape != (len(self._dof_names),):
            raise RuntimeError(
                f"Joint state has shape {positions.shape}; "
                f"expected {(len(self._dof_names),)}."
            )

        applied_action = self._articulation_controller.get_applied_action()
        targets = None if applied_action is None else applied_action.joint_positions
        if targets is None:
            targets_array = positions.copy()
        else:
            targets_array = np.asarray(targets, dtype=np.float32).copy()
            if targets_array.shape != positions.shape:
                raise RuntimeError(
                    f"Applied joint target has shape {targets_array.shape}; "
                    f"expected {positions.shape}."
                )
            # Isaac uses NaN for an uncontrolled component in some action modes.
            targets_array = np.where(
                np.isfinite(targets_array),
                targets_array,
                positions,
            ).astype(np.float32, copy=False)
        return positions, targets_array


class StagingEpisodeRecorder:
    """Record fixed-rate episodes without importing LeRobot inside Isaac Sim."""

    def __init__(
        self,
        config: DataCollectionConfig,
        articulation: ArticulationFrameSource,
        cameras: RgbCameraRig,
    ) -> None:
        self._config = config
        self._articulation = articulation
        self._cameras = cameras
        self._root = config.root.expanduser().resolve()
        self._episodes_root = self._root / "episodes"
        self._writer_lock_path = self._root / ".writer.lock"
        self._writer_lock_token = f"{os.getpid()}:{uuid.uuid4().hex}"
        self._pending_images: list[Future[None]] = []
        self._active_dir: Path | None = None
        self._active_index: int | None = None
        self._next_capture_time: float | None = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._simulation_times: list[float] = []
        self._task_states: list[str] = []
        self._closed = False

        self._root.mkdir(parents=True, exist_ok=True)
        self._acquire_writer_lock()
        try:
            self._initialize_root()
            self._executor = ThreadPoolExecutor(
                max_workers=config.image_writer_threads,
                thread_name_prefix="rgb-writer",
            )
        except Exception:
            self._release_writer_lock()
            raise

    @property
    def is_episode_active(self) -> bool:
        return self._active_dir is not None

    @property
    def num_pending_frames(self) -> int:
        return len(self._states)

    @property
    def root(self) -> Path:
        return self._root

    def begin_episode(self, simulation_time: float) -> None:
        """Open a new temporary episode and reset the fixed-rate clock."""
        self._require_open()
        if self.is_episode_active:
            raise RuntimeError("Cannot begin an episode while another is active.")
        self._active_index = self._next_episode_index()
        token = uuid.uuid4().hex[:8]
        self._active_dir = self._episodes_root / (
            f".episode-{self._active_index:06d}.in-progress-{token}"
        )
        for camera in self._config.cameras:
            (self._active_dir / "images" / camera.name).mkdir(
                parents=True,
                exist_ok=False,
            )
        self._next_capture_time = float(simulation_time)
        self._states = []
        self._actions = []
        self._simulation_times = []
        self._task_states = []
        self._pending_images = []

    def record_frame(self, simulation_time: float, task_state: str) -> bool:
        """Record the current observation/action pair when its sample is due."""
        self._require_open()
        if not self.is_episode_active:
            return False
        if self._next_capture_time is None:
            raise RuntimeError("Active episode has no capture clock.")

        period = 1.0 / self._config.fps
        tolerance = period * 1e-4
        if float(simulation_time) + tolerance < self._next_capture_time:
            return False
        while self._next_capture_time <= float(simulation_time) + tolerance:
            self._next_capture_time += period

        images = self._cameras.capture()
        if images is None:
            raise RuntimeError(
                "RGB cameras became unavailable during an active episode. "
                "Refusing to create a variable-rate dataset."
            )

        positions, targets = self._articulation.capture()
        frame_index = len(self._states)
        for camera in self._config.cameras:
            image_path = (
                self._active_dir
                / "images"
                / camera.name
                / f"frame-{frame_index:06d}.png"
            )
            self._pending_images.append(
                self._executor.submit(
                    _write_rgb_png,
                    image_path,
                    images[camera.feature_key],
                )
            )
            if (
                len(self._pending_images)
                >= self._config.max_pending_image_writes
            ):
                self._pending_images.pop(0).result()

        self._states.append(positions)
        self._actions.append(targets)
        self._simulation_times.append(float(simulation_time))
        self._task_states.append(str(task_state))
        return True

    def finish_episode(self, *, success: bool, end_reason: str) -> Path | None:
        """Flush and atomically publish the current episode."""
        self._require_open()
        if not self.is_episode_active:
            return None
        if not self._states:
            self.abort_episode()
            return None

        self._wait_for_images()
        np.savez(
            self._active_dir / "data.npz",
            observation_state=np.stack(self._states).astype(np.float32),
            action=np.stack(self._actions).astype(np.float32),
            simulation_time=np.asarray(self._simulation_times, dtype=np.float64),
            task_state=np.asarray(self._task_states, dtype=np.str_),
        )
        _write_json(
            self._active_dir / "episode.json",
            {
                "episode_index": self._active_index,
                "num_frames": len(self._states),
                "task": self._config.task,
                "success": bool(success),
                "end_reason": str(end_reason),
            },
        )
        final_dir = self._episodes_root / f"episode-{self._active_index:06d}"
        self._active_dir.rename(final_dir)
        self._reset_episode_state()
        return final_dir

    def abort_episode(self) -> None:
        """Discard an interrupted episode without touching completed episodes."""
        self._require_open()
        if not self.is_episode_active:
            return
        active_dir = self._active_dir
        first_error: Exception | None = None
        try:
            self._wait_for_images()
        except Exception as error:
            first_error = error
        try:
            shutil.rmtree(active_dir)
        except Exception as error:
            if first_error is None:
                first_error = error
            else:
                _LOGGER.exception(
                    "Failed to remove interrupted episode %s.",
                    active_dir,
                )
        finally:
            self._reset_episode_state()
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Discard any incomplete episode and stop image workers."""
        if self._closed:
            return
        first_error: Exception | None = None
        try:
            if self.is_episode_active:
                self.abort_episode()
        except Exception as error:
            first_error = error
        try:
            self._executor.shutdown(wait=True)
        except Exception as error:
            if first_error is None:
                first_error = error
        finally:
            self._release_writer_lock()
            self._closed = True
        if first_error is not None:
            raise first_error

    def _initialize_root(self) -> None:
        metadata_path = self._root / "dataset.json"
        metadata = {
            "format": _FORMAT_VERSION,
            "fps": self._config.fps,
            "dlss_exec_mode": self._config.dlss_exec_mode,
            "robot_type": self._config.robot_type,
            "dof_names": list(self._articulation.dof_names),
            "features": {
                **self._articulation.features,
                **self._cameras.features,
            },
            "cameras": [
                {
                    "name": camera.name,
                    "feature_key": camera.feature_key,
                    "resolution": list(camera.resolution),
                    "prim_path": camera.prim_path,
                    "pose_frame": camera.pose_frame,
                    "position": (
                        None if camera.position is None else list(camera.position)
                    ),
                    "look_at": (
                        None if camera.look_at is None else list(camera.look_at)
                    ),
                    "translation": (
                        None
                        if camera.translation is None
                        else list(camera.translation)
                    ),
                    "orientation": (
                        None
                        if camera.orientation is None
                        else list(camera.orientation)
                    ),
                    "focal_length_m": camera.focal_length_m,
                    "horizontal_aperture_m": camera.horizontal_aperture_m,
                    "clipping_range": list(camera.clipping_range),
                }
                for camera in self._config.cameras
            ],
        }
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != metadata:
                raise RuntimeError(
                    f"Staging schema at {metadata_path} does not match this run. "
                    "Choose another --record-root."
                )
        else:
            unexpected_entries = [
                path
                for path in self._root.iterdir()
                if path != self._writer_lock_path
            ]
            if unexpected_entries:
                raise RuntimeError(
                    f"Staging root {self._root} is non-empty but has no dataset.json."
                )
            _write_json(metadata_path, metadata)
        self._episodes_root.mkdir(parents=True, exist_ok=True)

    def _next_episode_index(self) -> int:
        indices = []
        for path in self._episodes_root.glob("episode-[0-9][0-9][0-9][0-9][0-9][0-9]"):
            indices.append(int(path.name.removeprefix("episode-")))
        return max(indices, default=-1) + 1

    def _wait_for_images(self) -> None:
        pending_images = self._pending_images
        self._pending_images = []
        first_error: Exception | None = None
        for future in pending_images:
            try:
                future.result()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _acquire_writer_lock(self) -> None:
        try:
            descriptor = os.open(
                self._writer_lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as error:
            owner = "unknown"
            try:
                owner = self._writer_lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise RuntimeError(
                f"Staging root {self._root} already has an active writer "
                f"(lock owner {owner!r}). Use one root per collection worker."
            ) from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(self._writer_lock_token + "\n")
        except Exception:
            self._writer_lock_path.unlink(missing_ok=True)
            raise

    def _release_writer_lock(self) -> None:
        try:
            if not self._writer_lock_path.exists():
                return
            owner = self._writer_lock_path.read_text(encoding="utf-8").strip()
            if owner == self._writer_lock_token:
                self._writer_lock_path.unlink()
        except OSError:
            _LOGGER.exception(
                "Failed to release staging writer lock %s.",
                self._writer_lock_path,
            )

    def _reset_episode_state(self) -> None:
        self._active_dir = None
        self._active_index = None
        self._next_capture_time = None
        self._states = []
        self._actions = []
        self._simulation_times = []
        self._task_states = []
        self._pending_images = []

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The staging recorder is already closed.")


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, format="PNG")


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
