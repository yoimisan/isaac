#!/usr/bin/env python3
"""Convert Isaac Scenes staging episodes into a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/isaac-pnp")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Export interrupted/failed episodes in addition to successful ones.",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Store visual features in Parquet instead of MP4 videos.",
    )
    return parser.parse_args()


def _require_supported_lerobot() -> None:
    try:
        installed = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "LeRobot is not installed. Use the isolated environment described in "
            "docs/data_collection.md."
        ) from error
    if installed != "0.6.0":
        raise RuntimeError(
            f"This exporter is validated against lerobot==0.6.0; found {installed}."
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metadata(input_root: Path) -> dict:
    metadata_path = input_root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing staging metadata: {metadata_path}")
    metadata = _load_json(metadata_path)
    if metadata.get("format") != "isaac-scenes-staging-v1":
        raise RuntimeError(
            f"Unsupported staging format {metadata.get('format')!r} in {metadata_path}."
        )
    return metadata


def _episode_paths(input_root: Path, include_failed: bool) -> list[Path]:
    episodes = []
    for path in sorted((input_root / "episodes").glob("episode-[0-9]*")):
        manifest_path = path / "episode.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        if include_failed or manifest.get("success") is True:
            episodes.append(path)
    if not episodes:
        qualifier = "completed" if include_failed else "successful"
        raise RuntimeError(f"No {qualifier} staging episodes found in {input_root}.")
    return episodes


def _features(metadata: dict, use_videos: bool) -> dict:
    features = json.loads(json.dumps(metadata["features"]))
    for camera in metadata["cameras"]:
        features[camera["feature_key"]]["dtype"] = (
            "video" if use_videos else "image"
        )
    features["next.done"] = {
        "dtype": "bool",
        "shape": [1],
        "names": None,
    }
    features["next.success"] = {
        "dtype": "bool",
        "shape": [1],
        "names": None,
    }
    return features


def _validate_episode(
    episode_path: Path,
    metadata: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = _load_json(episode_path / "episode.json")
    with np.load(episode_path / "data.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_frames = int(manifest["num_frames"])
    for key in ("observation_state", "action", "simulation_time", "task_state"):
        if key not in arrays:
            raise RuntimeError(f"{episode_path} is missing array {key!r}.")
        if len(arrays[key]) != expected_frames:
            raise RuntimeError(
                f"{episode_path}/{key} has {len(arrays[key])} frames; "
                f"expected {expected_frames}."
            )
    for camera in metadata["cameras"]:
        image_dir = episode_path / "images" / camera["name"]
        if len(list(image_dir.glob("frame-*.png"))) != expected_frames:
            raise RuntimeError(
                f"Camera {camera['name']!r} in {episode_path} does not contain "
                f"exactly {expected_frames} PNG frames."
            )
    return manifest, arrays


def export_dataset(args: argparse.Namespace) -> None:
    _require_supported_lerobot()
    from lerobot.datasets import LeRobotDataset

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}. "
            "Use a new path so an existing dataset is never overwritten."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_root = output_root.parent / (
        f".{output_root.name}.in-progress-{uuid.uuid4().hex[:8]}"
    )

    metadata = _load_metadata(input_root)
    episode_paths = _episode_paths(input_root, args.include_failed)
    features = _features(metadata, use_videos=not args.images)
    expected_frames = 0
    episode_boundaries: list[tuple[int, int]] = []
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=temporary_output_root,
            fps=int(metadata["fps"]),
            robot_type=metadata["robot_type"],
            features=features,
            use_videos=not args.images,
            image_writer_threads=4,
            streaming_encoding=False,
        )
        for episode_path in episode_paths:
            manifest, arrays = _validate_episode(episode_path, metadata)
            episode_length = int(manifest["num_frames"])
            start_index = expected_frames
            for frame_index in range(episode_length):
                frame = {
                    "observation.state": arrays["observation_state"][
                        frame_index
                    ].astype(np.float32, copy=False),
                    "action": arrays["action"][frame_index].astype(
                        np.float32,
                        copy=False,
                    ),
                    "next.done": np.asarray(
                        [frame_index == episode_length - 1],
                        dtype=np.bool_,
                    ),
                    "next.success": np.asarray(
                        [
                            bool(manifest["success"])
                            and frame_index == episode_length - 1
                        ],
                        dtype=np.bool_,
                    ),
                    "task": manifest["task"],
                }
                for camera in metadata["cameras"]:
                    image_path = (
                        episode_path
                        / "images"
                        / camera["name"]
                        / f"frame-{frame_index:06d}.png"
                    )
                    with Image.open(image_path) as image:
                        frame[camera["feature_key"]] = np.asarray(
                            image.convert("RGB"),
                            dtype=np.uint8,
                        ).copy()
                dataset.add_frame(frame)
            dataset.save_episode()
            expected_frames += episode_length
            episode_boundaries.append((start_index, expected_frames - 1))
        dataset.finalize()

        check = LeRobotDataset(
            repo_id=args.repo_id,
            root=temporary_output_root,
            return_uint8=True,
        )
        if check.num_episodes != len(episode_boundaries):
            raise RuntimeError(
                f"Reloaded dataset has {check.num_episodes} episodes; "
                f"expected {len(episode_boundaries)}."
            )
        if len(check) != expected_frames:
            raise RuntimeError(
                f"Reloaded dataset has {len(check)} frames; "
                f"expected {expected_frames}."
            )
        image_keys = [camera["feature_key"] for camera in metadata["cameras"]]
        for start_index, end_index in episode_boundaries:
            for sample_index in {start_index, end_index}:
                sample = check[sample_index]
                for image_key in image_keys:
                    if image_key not in sample:
                        raise RuntimeError(
                            f"Decoded sample {sample_index} is missing {image_key}."
                        )
        num_episodes = check.num_episodes
        num_frames = len(check)
        del check
        temporary_output_root.rename(output_root)
    except Exception:
        if dataset is not None:
            # Preserve the original export/validation exception even if LeRobot's
            # worker cleanup also fails.
            with contextlib.suppress(Exception):
                dataset.finalize()
        shutil.rmtree(temporary_output_root, ignore_errors=True)
        raise

    print(
        f"Exported and reloaded LeRobot v3.0 dataset at {output_root}: "
        f"{num_episodes} episodes, {num_frames} frames."
    )


if __name__ == "__main__":
    export_dataset(_parse_args())
