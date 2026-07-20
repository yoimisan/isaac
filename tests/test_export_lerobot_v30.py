"""Tests for atomic publication by the isolated LeRobot exporter."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools import export_lerobot_v30


class _FakeLeRobotDataset:
    datasets: dict[Path, dict] = {}
    fail_on_add = False

    def __init__(self, repo_id: str, root: Path, return_uint8: bool) -> None:
        del repo_id, return_uint8
        stored = self.datasets[Path(root)]
        self._frames = stored["frames"]
        self.num_episodes = stored["num_episodes"]

    @classmethod
    def create(cls, *, root: Path, **kwargs) -> _FakeLeRobotDataset:
        del kwargs
        root = Path(root)
        root.mkdir(parents=True)
        instance = object.__new__(cls)
        instance._root = root
        instance._frames = []
        instance.num_episodes = 0
        cls.datasets[root] = {
            "frames": instance._frames,
            "num_episodes": 0,
        }
        return instance

    def add_frame(self, frame: dict) -> None:
        if self.fail_on_add:
            raise OSError("synthetic export failure")
        self._frames.append(frame)

    def save_episode(self) -> None:
        self.num_episodes += 1
        self.datasets[self._root]["num_episodes"] = self.num_episodes

    def finalize(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> dict:
        return self._frames[index]


class ExportLeRobotV30Test(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.input_root = self.root / "raw"
        self.output_root = self.root / "lerobot"
        self._write_staging_episode()
        self.args = argparse.Namespace(
            input_root=self.input_root,
            output_root=self.output_root,
            repo_id="local/test",
            include_failed=False,
            images=True,
        )
        _FakeLeRobotDataset.datasets = {}
        _FakeLeRobotDataset.fail_on_add = False

        package = types.ModuleType("lerobot")
        datasets_module = types.ModuleType("lerobot.datasets")
        datasets_module.LeRobotDataset = _FakeLeRobotDataset
        self._module_patch = patch.dict(
            sys.modules,
            {"lerobot": package, "lerobot.datasets": datasets_module},
        )
        self._version_patch = patch.object(
            export_lerobot_v30,
            "_require_supported_lerobot",
        )
        self._module_patch.start()
        self._version_patch.start()

    def tearDown(self) -> None:
        self._version_patch.stop()
        self._module_patch.stop()
        self._temporary_directory.cleanup()

    def test_validated_export_is_published_atomically(self) -> None:
        export_lerobot_v30.export_dataset(self.args)

        self.assertTrue(self.output_root.is_dir())
        self.assertFalse(list(self.root.glob(".lerobot.in-progress-*")))

    def test_failed_export_removes_temporary_output(self) -> None:
        _FakeLeRobotDataset.fail_on_add = True

        with self.assertRaisesRegex(OSError, "synthetic export failure"):
            export_lerobot_v30.export_dataset(self.args)

        self.assertFalse(self.output_root.exists())
        self.assertFalse(list(self.root.glob(".lerobot.in-progress-*")))

    def _write_staging_episode(self) -> None:
        episode = self.input_root / "episodes" / "episode-000000"
        image_dir = episode / "images" / "overview"
        image_dir.mkdir(parents=True)
        metadata = {
            "format": "isaac-scenes-staging-v1",
            "fps": 60,
            "robot_type": "franka",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {"motors": ["joint_0", "joint_1"]},
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {"motors": ["joint_0", "joint_1"]},
                },
                "observation.images.overview": {
                    "dtype": "video",
                    "shape": [4, 5, 3],
                    "names": ["height", "width", "channels"],
                },
            },
            "cameras": [
                {
                    "name": "overview",
                    "feature_key": "observation.images.overview",
                }
            ],
        }
        (self.input_root / "dataset.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        (episode / "episode.json").write_text(
            json.dumps(
                {
                    "num_frames": 1,
                    "success": True,
                    "task": "test task",
                }
            ),
            encoding="utf-8",
        )
        np.savez(
            episode / "data.npz",
            observation_state=np.zeros((1, 2), dtype=np.float32),
            action=np.ones((1, 2), dtype=np.float32),
            simulation_time=np.zeros(1, dtype=np.float64),
            task_state=np.asarray(["APPROACH"]),
        )
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
            image_dir / "frame-000000.png"
        )


if __name__ == "__main__":
    unittest.main()
