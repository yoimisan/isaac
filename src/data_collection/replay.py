"""Load and validate staging episodes before or during Isaac replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_collection.config import CameraConfig


_SUPPORTED_FORMATS = frozenset({"isaac-scenes-staging-v1"})


@dataclass(frozen=True)
class StagingEpisode:
    """In-memory trajectory plus paths to its recorded RGB frames."""

    dataset_root: Path
    episode_path: Path
    dataset_metadata: dict[str, Any]
    episode_metadata: dict[str, Any]
    observation_state: np.ndarray
    action: np.ndarray
    simulation_time: np.ndarray
    task_state: np.ndarray
    scene_pose: np.ndarray | None

    @classmethod
    def load(cls, dataset_root: Path, episode_index: int) -> StagingEpisode:
        """Load one published episode without importing Isaac Sim."""
        if episode_index < 0:
            raise ValueError("episode_index must be nonnegative.")
        root = dataset_root.expanduser().resolve()
        metadata_path = root / "dataset.json"
        episode_path = root / "episodes" / f"episode-{episode_index:06d}"
        episode_metadata_path = episode_path / "episode.json"
        arrays_path = episode_path / "data.npz"
        for path in (metadata_path, episode_metadata_path, arrays_path):
            if not path.is_file():
                raise FileNotFoundError(f"Replay input is missing {path}.")

        dataset_metadata = _read_json(metadata_path)
        episode_metadata = _read_json(episode_metadata_path)
        missing_metadata = sorted(
            {"format", "fps", "dof_names", "cameras"}.difference(
                dataset_metadata
            )
        )
        if missing_metadata:
            raise ValueError(
                f"Dataset metadata is missing required fields: {missing_metadata}."
            )
        if "episode_index" not in episode_metadata:
            raise ValueError("Episode metadata is missing episode_index.")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = {
                "observation_state",
                "action",
                "simulation_time",
                "task_state",
            }
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise ValueError(
                    f"Episode arrays are missing required fields: {missing}."
                )
            scene_pose = (
                np.asarray(arrays["scene_pose"]).copy()
                if "scene_pose" in arrays.files
                else None
            )
            return cls(
                dataset_root=root,
                episode_path=episode_path,
                dataset_metadata=dataset_metadata,
                episode_metadata=episode_metadata,
                observation_state=np.asarray(
                    arrays["observation_state"]
                ).copy(),
                action=np.asarray(arrays["action"]).copy(),
                simulation_time=np.asarray(arrays["simulation_time"]).copy(),
                task_state=np.asarray(arrays["task_state"]).copy(),
                scene_pose=scene_pose,
            )

    @property
    def episode_index(self) -> int:
        return int(self.episode_metadata["episode_index"])

    @property
    def num_frames(self) -> int:
        return int(self.observation_state.shape[0])

    @property
    def dof_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.dataset_metadata["dof_names"])

    @property
    def replay_object_names(self) -> tuple[str, ...]:
        return tuple(
            str(name)
            for name in self.dataset_metadata.get("replay_objects", [])
        )

    @property
    def camera_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.dataset_metadata.get("cameras", []))

    def image_path(self, camera_name: str, frame_index: int) -> Path:
        return (
            self.episode_path
            / "images"
            / camera_name
            / f"frame-{frame_index:06d}.png"
        )


@dataclass(frozen=True)
class EpisodeValidationReport:
    """Machine-readable integrity result for one staging episode."""

    dataset_root: str
    episode_index: int
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "episode_index": self.episode_index,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }


def validate_episode(
    episode: StagingEpisode,
    *,
    verify_images: bool = True,
) -> EpisodeValidationReport:
    """Check array, timing, scene-pose, and RGB-file consistency."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    metadata = episode.dataset_metadata
    format_name = metadata.get("format")
    if format_name not in _SUPPORTED_FORMATS:
        errors.append(f"Unsupported staging format: {format_name!r}.")
    if "rendering" not in metadata:
        warnings.append(
            "Dataset predates explicit rendering metadata; replay uses the "
            "current compatibility defaults."
        )

    num_frames = episode.num_frames
    expected_frames = int(episode.episode_metadata.get("num_frames", -1))
    metrics["num_frames"] = num_frames
    if num_frames <= 0:
        errors.append("Episode has no frames.")
    if expected_frames != num_frames:
        errors.append(
            f"episode.json declares {expected_frames} frames, found {num_frames}."
        )

    num_dofs = len(episode.dof_names)
    expected_matrix_shape = (num_frames, num_dofs)
    for name, array in (
        ("observation_state", episode.observation_state),
        ("action", episode.action),
    ):
        if array.shape != expected_matrix_shape:
            errors.append(
                f"{name} has shape {array.shape}; expected "
                f"{expected_matrix_shape}."
            )
        elif not np.all(np.isfinite(array)):
            errors.append(f"{name} contains NaN or infinity.")

    for name, array in (
        ("simulation_time", episode.simulation_time),
        ("task_state", episode.task_state),
    ):
        if array.shape != (num_frames,):
            errors.append(
                f"{name} has shape {array.shape}; expected {(num_frames,)}."
            )

    if episode.simulation_time.shape == (num_frames,):
        _validate_timing(episode, errors, metrics)
    if episode.task_state.shape == (num_frames,):
        empty_states = int(
            np.count_nonzero(
                np.asarray([not str(value) for value in episode.task_state])
            )
        )
        if empty_states:
            errors.append(f"task_state contains {empty_states} empty values.")
        transitions = []
        previous = None
        for value in episode.task_state:
            current = str(value)
            if current != previous:
                transitions.append(current)
                previous = current
        metrics["task_state_transitions"] = transitions

    if episode.observation_state.shape == expected_matrix_shape:
        state_delta = np.diff(episode.observation_state, axis=0)
        metrics["max_joint_step"] = (
            0.0 if not state_delta.size else float(np.max(np.abs(state_delta)))
        )
    if episode.action.shape == expected_matrix_shape and episode.action.size:
        metrics["action_min"] = float(np.min(episode.action))
        metrics["action_max"] = float(np.max(episode.action))

    _validate_scene_trace(episode, errors, warnings, metrics)
    _validate_images(
        episode,
        errors,
        warnings,
        metrics,
        verify_images=verify_images,
    )
    return EpisodeValidationReport(
        dataset_root=str(episode.dataset_root),
        episode_index=episode.episode_index,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def camera_configs_from_metadata(
    metadata: dict[str, Any],
) -> tuple[CameraConfig, ...]:
    """Recreate the recorded camera rig instead of using current defaults."""
    configs = []
    for camera in metadata.get("cameras", []):
        configs.append(
            CameraConfig(
                name=str(camera["name"]),
                prim_path=str(camera["prim_path"]),
                position=_optional_tuple(camera.get("position")),
                look_at=_optional_tuple(camera.get("look_at")),
                translation=_optional_tuple(camera.get("translation")),
                orientation=_optional_tuple(camera.get("orientation")),
                resolution=tuple(int(value) for value in camera["resolution"]),
                focal_length_m=float(camera["focal_length_m"]),
                horizontal_aperture_m=float(camera["horizontal_aperture_m"]),
                clipping_range=tuple(
                    float(value) for value in camera["clipping_range"]
                ),
            )
        )
    if not configs:
        raise ValueError("Dataset metadata contains no cameras.")
    return tuple(configs)


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a JSON report atomically enough for local replay diagnostics."""
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _validate_timing(
    episode: StagingEpisode,
    errors: list[str],
    metrics: dict[str, Any],
) -> None:
    times = np.asarray(episode.simulation_time, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        errors.append("simulation_time contains NaN or infinity.")
        return
    if len(times) < 2:
        metrics["duration_seconds"] = 0.0
        return
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        errors.append("simulation_time must be strictly increasing.")
        return
    fps = float(episode.dataset_metadata["fps"])
    expected_period = 1.0 / fps
    maximum_error = float(np.max(np.abs(deltas - expected_period)))
    metrics.update(
        {
            "duration_seconds": float(times[-1] - times[0]),
            "mean_frame_period": float(np.mean(deltas)),
            "max_frame_period_error": maximum_error,
        }
    )
    if maximum_error > expected_period * 1e-3:
        errors.append(
            "simulation_time is not uniformly sampled at the declared fps: "
            f"maximum period error is {maximum_error:.6g}s."
        )


def _validate_scene_trace(
    episode: StagingEpisode,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    object_names = episode.replay_object_names
    scene_pose = episode.scene_pose
    if scene_pose is None:
        if object_names:
            errors.append("dataset.json declares replay_objects but scene_pose is missing.")
        else:
            warnings.append(
                "Episode predates scene-pose recording; replay is limited to "
                "the robot trajectory and cannot reproduce cube/target motion."
            )
        metrics["replay_objects"] = []
        return
    if not object_names:
        errors.append("scene_pose exists but dataset.json has no replay_objects.")
        return
    expected_shape = (episode.num_frames, len(object_names), 7)
    if scene_pose.shape != expected_shape:
        errors.append(
            f"scene_pose has shape {scene_pose.shape}; expected {expected_shape}."
        )
        return
    if not np.all(np.isfinite(scene_pose)):
        errors.append("scene_pose contains NaN or infinity.")
        return
    quaternion_norms = np.linalg.norm(scene_pose[:, :, 3:], axis=2)
    if not quaternion_norms.size:
        return
    maximum_norm_error = float(np.max(np.abs(quaternion_norms - 1.0)))
    metrics["replay_objects"] = list(object_names)
    metrics["max_scene_quaternion_norm_error"] = maximum_norm_error
    if maximum_norm_error > 1e-3:
        errors.append(
            "scene_pose contains non-normalized quaternions: maximum norm "
            f"error is {maximum_norm_error:.6g}."
        )


def _validate_images(
    episode: StagingEpisode,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
    *,
    verify_images: bool,
) -> None:
    camera_metrics: dict[str, Any] = {}
    if not episode.camera_metadata:
        errors.append("dataset.json contains no camera metadata.")
        metrics["cameras"] = camera_metrics
        return
    for camera in episode.camera_metadata:
        name = str(camera["name"])
        resolution = tuple(int(value) for value in camera["resolution"])
        expected_size = (resolution[1], resolution[0])
        directory = episode.episode_path / "images" / name
        expected_paths = [
            episode.image_path(name, index)
            for index in range(episode.num_frames)
        ]
        missing = [path.name for path in expected_paths if not path.is_file()]
        actual_paths = set(directory.glob("frame-*.png")) if directory.is_dir() else set()
        unexpected_count = len(actual_paths.difference(expected_paths))
        if missing:
            errors.append(
                f"Camera {name!r} is missing {len(missing)} RGB frames; "
                f"first missing file is {missing[0]}."
            )
        if unexpected_count:
            errors.append(
                f"Camera {name!r} contains {unexpected_count} unexpected frames."
            )

        invalid_images = 0
        if verify_images:
            for path in expected_paths:
                if not path.is_file():
                    continue
                try:
                    with Image.open(path) as image:
                        if image.size != expected_size or image.mode != "RGB":
                            invalid_images += 1
                        image.verify()
                except (OSError, SyntaxError, ValueError):
                    invalid_images += 1
            if invalid_images:
                errors.append(
                    f"Camera {name!r} has {invalid_images} corrupt or malformed "
                    "RGB frames."
                )

        sample_means = []
        for index in sorted({0, episode.num_frames // 2, episode.num_frames - 1}):
            if index < 0:
                continue
            path = episode.image_path(name, index)
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    sample_means.append(
                        float(np.asarray(image.convert("RGB"), dtype=np.float32).mean())
                    )
            except (OSError, SyntaxError, ValueError):
                continue
        minimum_sample_mean = min(sample_means, default=0.0)
        if sample_means and minimum_sample_mean <= 5.0:
            warnings.append(
                f"Camera {name!r} has a nearly black sampled frame "
                f"(mean={minimum_sample_mean:.3f})."
            )
        camera_metrics[name] = {
            "expected_frames": episode.num_frames,
            "present_frames": len(actual_paths),
            "invalid_frames": invalid_images,
            "sample_mean_rgb": sample_means,
        }
    metrics["cameras"] = camera_metrics


def _optional_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value
