"""Isaac Sim runtime for deterministic staging-episode replay."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction

from data_collection.camera import RgbCameraRig
from data_collection.replay import (
    StagingEpisode,
    camera_configs_from_metadata,
)


@dataclass(frozen=True)
class ReplaySceneHandles:
    """Simulator handles supplied by one task-specific replay adapter."""

    world: World
    robot: Any
    articulation_controller: Any
    objects: Mapping[str, Any]


@dataclass(frozen=True)
class ReplayOptions:
    """Control physics-vs-state replay and validation strictness."""

    mode: str = "action"
    scene_mode: str = "trace"
    compare_images: bool = True
    image_stride: int = 1
    playback_speed: float = 0.0
    max_frames: int | None = None
    joint_error_threshold: float = 0.05
    image_mae_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"action", "state"}:
            raise ValueError("Replay mode must be 'action' or 'state'.")
        if self.scene_mode not in {"trace", "initial"}:
            raise ValueError("scene_mode must be 'trace' or 'initial'.")
        if self.image_stride <= 0:
            raise ValueError("image_stride must be positive.")
        if self.playback_speed < 0.0:
            raise ValueError("playback_speed cannot be negative.")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when supplied.")
        if self.joint_error_threshold <= 0.0:
            raise ValueError("joint_error_threshold must be positive.")
        if self.image_mae_threshold is not None and self.image_mae_threshold < 0:
            raise ValueError("image_mae_threshold cannot be negative.")


class IsaacEpisodeReplayer:
    """Replay recorded robot targets or directly visualize recorded states."""

    def __init__(
        self,
        simulation_app: SimulationApp,
        episode: StagingEpisode,
        options: ReplayOptions,
        scene: ReplaySceneHandles,
    ) -> None:
        self._simulation_app = simulation_app
        self._episode = episode
        self._options = options
        self._world = scene.world
        self._robot = scene.robot
        self._articulation_controller = scene.articulation_controller
        actual_dof_names = tuple(self._robot.dof_names)
        if actual_dof_names != episode.dof_names:
            raise RuntimeError(
                "Replay robot DOF order differs from the dataset: "
                f"runtime={actual_dof_names}, dataset={episode.dof_names}."
            )

        self._scene_objects = {
            name: scene.objects.get(name)
            for name in episode.replay_object_names
        }
        missing_objects = [
            name for name, value in self._scene_objects.items() if value is None
        ]
        if missing_objects:
            raise RuntimeError(
                f"Replay scene is missing objects: {missing_objects}."
            )

        self._camera_configs = ()
        self._cameras = None
        if options.compare_images:
            rendering = episode.dataset_metadata.get("rendering", {})
            self._camera_configs = camera_configs_from_metadata(
                episode.dataset_metadata
            )
            self._cameras = RgbCameraRig(
                self._camera_configs,
                dlss_exec_mode=int(
                    rendering.get(
                        "dlss_exec_mode",
                        episode.dataset_metadata.get("dlss_exec_mode", 2),
                    )
                ),
                tonemap_op=int(rendering.get("tonemap_op", 4)),
                film_iso=float(rendering.get("film_iso", 100.0)),
            )
            self._warm_up_cameras()

    def replay(self) -> dict[str, Any]:
        """Run the selected replay and return quantitative error metrics."""
        frame_count = self._episode.num_frames
        if self._options.max_frames is not None:
            frame_count = min(frame_count, self._options.max_frames)

        joint_errors = _VectorErrorAccumulator()
        scene_errors = {
            name: _PoseErrorAccumulator()
            for name in self._episode.replay_object_names
        }
        image_errors = {
            config.name: _ImageErrorAccumulator()
            for config in self._camera_configs
        }
        warnings: list[str] = []
        if self._episode.scene_pose is None:
            warnings.append(
                "No scene_pose trace is available; cube and target replay use "
                "the newly created task scene."
            )
        if (
            self._options.mode == "action"
            and self._options.scene_mode == "trace"
            and self._episode.scene_pose is not None
        ):
            warnings.append(
                "scene_mode='trace' corrects replay objects to recorded poses "
                "each frame. Use scene_mode='initial' to measure uncorrected "
                "clean-scene physics reproducibility."
            )

        self._set_robot_state(self._episode.observation_state[0])
        if self._episode.scene_pose is not None:
            self._set_scene_pose(0)
        self._world.render()

        start_time = time.monotonic()
        for frame_index in range(frame_count):
            if not self._simulation_app.is_running():
                warnings.append("Simulation stopped before replay completed.")
                break
            if self._options.mode == "state":
                self._set_robot_state(
                    self._episode.observation_state[frame_index]
                )
                if self._episode.scene_pose is not None:
                    self._set_scene_pose(frame_index)
                self._world.render()
            else:
                self._measure_scene_error(frame_index, scene_errors)
                if (
                    frame_index > 0
                    and self._options.scene_mode == "trace"
                    and self._episode.scene_pose is not None
                ):
                    self._set_scene_pose(frame_index)
                    self._world.render()

            actual_positions = self._read_robot_positions()
            joint_errors.add(
                actual_positions
                - self._episode.observation_state[frame_index],
                frame_index,
            )
            if self._options.mode == "state":
                self._measure_scene_error(frame_index, scene_errors)

            if (
                self._options.compare_images
                and frame_index % self._options.image_stride == 0
            ):
                self._compare_images(frame_index, image_errors)

            if self._options.mode == "action":
                self._articulation_controller.apply_action(
                    ArticulationAction(
                        joint_positions=self._episode.action[frame_index]
                    )
                )
                self._world.step(render=True)

            self._pace(frame_index, start_time)

        replayed_frames = joint_errors.count
        joint_metrics = joint_errors.metrics()
        image_metrics = {
            name: accumulator.metrics()
            for name, accumulator in image_errors.items()
        }
        scene_metrics = {
            name: accumulator.metrics()
            for name, accumulator in scene_errors.items()
        }
        errors = []
        if joint_metrics["max_abs"] > self._options.joint_error_threshold:
            errors.append(
                "Maximum replay joint error "
                f"{joint_metrics['max_abs']:.6g} exceeds threshold "
                f"{self._options.joint_error_threshold:.6g}."
            )
        if self._options.image_mae_threshold is not None:
            for name, metrics in image_metrics.items():
                if metrics["frames"] and (
                    metrics["mae"] > self._options.image_mae_threshold
                ):
                    errors.append(
                        f"Camera {name!r} replay MAE {metrics['mae']:.6g} "
                        "exceeds threshold "
                        f"{self._options.image_mae_threshold:.6g}."
                    )

        return {
            "ok": not errors and replayed_frames == frame_count,
            "mode": self._options.mode,
            "scene_mode": self._effective_scene_mode(),
            "requested_frames": frame_count,
            "replayed_frames": replayed_frames,
            "errors": errors,
            "warnings": warnings,
            "joint_error": joint_metrics,
            "scene_pose_error_before_correction": scene_metrics,
            "image_error": image_metrics,
        }

    def close(self) -> None:
        if self._cameras is not None:
            self._cameras.close()

    def _warm_up_cameras(self) -> None:
        if self._cameras is None:
            return
        for _ in range(10):
            self._world.render()
            if self._cameras.capture() is not None:
                return
        raise RuntimeError("Replay cameras produced no RGB after 10 renders.")

    def _read_robot_positions(self) -> np.ndarray:
        state = self._robot.get_joints_state()
        if state is None or state.positions is None:
            raise RuntimeError("Failed to read replay robot joint positions.")
        return np.asarray(state.positions, dtype=np.float64)

    def _set_robot_state(self, positions: np.ndarray) -> None:
        values = np.asarray(positions, dtype=np.float64)
        self._robot.set_joint_positions(values)
        self._robot.set_joint_velocities(np.zeros_like(values))

    def _set_scene_pose(self, frame_index: int) -> None:
        scene_pose = self._episode.scene_pose
        if scene_pose is None:
            return
        for object_index, name in enumerate(self._episode.replay_object_names):
            pose = scene_pose[frame_index, object_index]
            scene_object = self._scene_objects[name]
            scene_object.set_world_pose(
                position=np.asarray(pose[:3], dtype=np.float64),
                orientation=np.asarray(pose[3:], dtype=np.float64),
            )
            if hasattr(scene_object, "set_linear_velocity"):
                scene_object.set_linear_velocity(np.zeros(3))
            if hasattr(scene_object, "set_angular_velocity"):
                scene_object.set_angular_velocity(np.zeros(3))

    def _measure_scene_error(
        self,
        frame_index: int,
        accumulators: dict[str, _PoseErrorAccumulator],
    ) -> None:
        scene_pose = self._episode.scene_pose
        if scene_pose is None:
            return
        for object_index, name in enumerate(self._episode.replay_object_names):
            position, orientation = self._scene_objects[name].get_world_pose()
            expected = scene_pose[frame_index, object_index]
            accumulators[name].add(
                np.asarray(position, dtype=np.float64) - expected[:3],
                np.asarray(orientation, dtype=np.float64),
                np.asarray(expected[3:], dtype=np.float64),
                frame_index,
            )

    def _compare_images(
        self,
        frame_index: int,
        accumulators: dict[str, _ImageErrorAccumulator],
    ) -> None:
        if self._cameras is None:
            raise RuntimeError("Image comparison was not initialized.")
        replay_images = self._cameras.capture()
        if replay_images is None:
            raise RuntimeError(
                f"Replay cameras returned no RGB at frame {frame_index}."
            )
        for config in self._camera_configs:
            reference_path = self._episode.image_path(
                config.name,
                frame_index,
            )
            with Image.open(reference_path) as image:
                reference = np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                )
            replayed = np.asarray(
                replay_images[config.feature_key],
                dtype=np.uint8,
            )
            if replayed.shape != reference.shape:
                raise RuntimeError(
                    f"Replay camera {config.name!r} produced {replayed.shape}; "
                    f"recorded frame has {reference.shape}."
                )
            accumulators[config.name].add(
                replayed,
                reference,
                frame_index,
            )

    def _pace(self, frame_index: int, start_time: float) -> None:
        if self._options.playback_speed == 0.0:
            return
        fps = float(self._episode.dataset_metadata["fps"])
        target_elapsed = (frame_index + 1) / (
            fps * self._options.playback_speed
        )
        remaining = target_elapsed - (time.monotonic() - start_time)
        if remaining > 0.0:
            time.sleep(remaining)

    def _effective_scene_mode(self) -> str:
        if self._episode.scene_pose is None:
            return "unavailable"
        if self._options.mode == "state":
            return "trace"
        return self._options.scene_mode


class _VectorErrorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self._element_count = 0
        self._sum_squared = 0.0
        self._max_abs = 0.0
        self._max_abs_frame: int | None = None

    def add(self, error: np.ndarray, frame_index: int) -> None:
        values = np.asarray(error, dtype=np.float64)
        self.count += 1
        self._element_count += values.size
        self._sum_squared += float(np.sum(values * values))
        maximum = float(np.max(np.abs(values)))
        if maximum > self._max_abs:
            self._max_abs = maximum
            self._max_abs_frame = frame_index

    def metrics(self) -> dict[str, float | int | None]:
        rmse = (
            math.sqrt(self._sum_squared / self._element_count)
            if self._element_count
            else 0.0
        )
        return {
            "frames": self.count,
            "rmse": rmse,
            "max_abs": self._max_abs,
            "max_abs_frame": self._max_abs_frame,
        }


class _PoseErrorAccumulator:
    def __init__(self) -> None:
        self._positions = _VectorErrorAccumulator()
        self._orientation_count = 0
        self._orientation_sum_squared = 0.0
        self._orientation_max = 0.0
        self._orientation_max_frame: int | None = None

    def add(
        self,
        position_error: np.ndarray,
        actual_orientation: np.ndarray,
        expected_orientation: np.ndarray,
        frame_index: int,
    ) -> None:
        self._positions.add(position_error, frame_index)
        dot = float(
            np.dot(actual_orientation, expected_orientation)
            / (
                np.linalg.norm(actual_orientation)
                * np.linalg.norm(expected_orientation)
            )
        )
        angle = 2.0 * math.acos(min(1.0, max(-1.0, abs(dot))))
        self._orientation_count += 1
        self._orientation_sum_squared += angle * angle
        if angle > self._orientation_max:
            self._orientation_max = angle
            self._orientation_max_frame = frame_index

    def metrics(self) -> dict[str, Any]:
        orientation_rmse = (
            math.sqrt(
                self._orientation_sum_squared / self._orientation_count
            )
            if self._orientation_count
            else 0.0
        )
        return {
            "position": self._positions.metrics(),
            "orientation_radians": {
                "frames": self._orientation_count,
                "rmse": orientation_rmse,
                "max_abs": self._orientation_max,
                "max_abs_frame": self._orientation_max_frame,
            },
        }


class _ImageErrorAccumulator:
    def __init__(self) -> None:
        self._frames = 0
        self._elements = 0
        self._sum_abs = 0.0
        self._sum_squared = 0.0
        self._max_abs = 0.0
        self._replayed_sum = 0.0
        self._reference_sum = 0.0
        self._worst_frame: int | None = None
        self._worst_frame_mae = 0.0

    def add(
        self,
        replayed: np.ndarray,
        reference: np.ndarray,
        frame_index: int,
    ) -> None:
        difference = replayed.astype(np.float32) - reference.astype(np.float32)
        absolute = np.abs(difference)
        self._frames += 1
        self._elements += difference.size
        self._sum_abs += float(np.sum(absolute))
        self._sum_squared += float(np.sum(difference * difference))
        self._replayed_sum += float(np.sum(replayed))
        self._reference_sum += float(np.sum(reference))
        self._max_abs = max(self._max_abs, float(np.max(absolute)))
        frame_mae = float(np.mean(absolute))
        if frame_mae > self._worst_frame_mae:
            self._worst_frame_mae = frame_mae
            self._worst_frame = frame_index

    def metrics(self) -> dict[str, float | int | None]:
        if not self._elements:
            return {
                "frames": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "psnr_db": None,
                "max_abs": 0.0,
                "mean_replayed_rgb": 0.0,
                "mean_reference_rgb": 0.0,
                "worst_frame": None,
                "worst_frame_mae": 0.0,
            }
        mae = self._sum_abs / self._elements
        rmse = math.sqrt(self._sum_squared / self._elements)
        psnr = None if rmse == 0.0 else 20.0 * math.log10(255.0 / rmse)
        return {
            "frames": self._frames,
            "mae": mae,
            "rmse": rmse,
            "psnr_db": psnr,
            "max_abs": self._max_abs,
            "mean_replayed_rgb": self._replayed_sum / self._elements,
            "mean_reference_rgb": self._reference_sum / self._elements,
            "worst_frame": self._worst_frame,
            "worst_frame_mae": self._worst_frame_mae,
        }
