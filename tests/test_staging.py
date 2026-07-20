"""Tests for the Isaac-independent staging layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_collection.config import CameraConfig, DataCollectionConfig
from data_collection.staging import ArticulationFrameSource, StagingEpisodeRecorder


class _FakeArticulationSource:
    dof_names = ("joint_0", "joint_1")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": {"motors": list(dof_names)},
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": {"motors": list(dof_names)},
        },
    }

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0, 4.0], dtype=np.float32),
        )


class _FakeCameraRig:
    def __init__(self, camera: CameraConfig) -> None:
        self._camera = camera
        self.features = {
            camera.feature_key: {
                "dtype": "video",
                "shape": [*camera.resolution, 3],
                "names": ["height", "width", "channels"],
            }
        }

    def capture(self) -> dict[str, np.ndarray]:
        return {
            self._camera.feature_key: np.zeros(
                (*self._camera.resolution, 3),
                dtype=np.uint8,
            )
        }


class StagingEpisodeRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name) / "dataset"
        self.camera = CameraConfig(
            name="overview",
            prim_path="/World/Camera",
            position=(1.0, 1.0, 1.0),
            look_at=(0.0, 0.0, 0.0),
            resolution=(4, 5),
        )
        self.config = DataCollectionConfig(
            enabled=True,
            root=self.root,
            fps=60,
            cameras=(self.camera,),
            max_pending_image_writes=2,
        )

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _make_recorder(self) -> StagingEpisodeRecorder:
        return StagingEpisodeRecorder(
            self.config,
            articulation=_FakeArticulationSource(),
            cameras=_FakeCameraRig(self.camera),
        )

    def test_fixed_rate_publish_abort_and_resume(self) -> None:
        recorder = self._make_recorder()
        recorder.begin_episode(0.0)
        for step in range(7):
            recorder.record_frame(step / 60.0, "APPROACH")
        episode = recorder.finish_episode(success=True, end_reason="test")
        recorder.close()

        self.assertEqual(episode.name, "episode-000000")
        arrays = np.load(episode / "data.npz", allow_pickle=False)
        self.assertEqual(arrays["observation_state"].shape, (7, 2))
        self.assertEqual(len(list((episode / "images" / "overview").glob("*.png"))), 7)

        resumed = self._make_recorder()
        resumed.begin_episode(0.0)
        resumed.record_frame(0.0, "DESCEND")
        second_episode = resumed.finish_episode(
            success=False,
            end_reason="test_reset",
        )
        resumed.begin_episode(0.0)
        resumed.record_frame(0.0, "APPROACH")
        resumed.abort_episode()
        resumed.close()

        self.assertEqual(second_episode.name, "episode-000001")
        self.assertFalse(any((self.root / "episodes").glob(".episode-*.in-progress-*")))

    def test_only_one_writer_can_own_a_staging_root(self) -> None:
        recorder = self._make_recorder()
        try:
            with self.assertRaisesRegex(RuntimeError, "active writer"):
                self._make_recorder()
        finally:
            recorder.close()

        reopened = self._make_recorder()
        reopened.close()

    def test_async_image_failure_is_cleaned_up_and_releases_lock(self) -> None:
        recorder = self._make_recorder()
        recorder.begin_episode(0.0)
        with patch(
            "data_collection.staging._write_rgb_png",
            side_effect=OSError("disk full"),
        ):
            recorder.record_frame(0.0, "APPROACH")
            with self.assertRaisesRegex(OSError, "disk full"):
                recorder.finish_episode(success=True, end_reason="test")
        recorder.close()

        self.assertFalse((self.root / ".writer.lock").exists())
        self.assertFalse(
            any((self.root / "episodes").glob(".episode-*.in-progress-*"))
        )
        reopened = self._make_recorder()
        reopened.close()

    def test_unavailable_camera_fails_instead_of_dropping_a_frame(self) -> None:
        camera_rig = _FakeCameraRig(self.camera)
        camera_rig.capture = lambda: None
        recorder = StagingEpisodeRecorder(
            self.config,
            articulation=_FakeArticulationSource(),
            cameras=camera_rig,
        )
        recorder.begin_episode(0.0)

        with self.assertRaisesRegex(RuntimeError, "variable-rate dataset"):
            recorder.record_frame(0.0, "APPROACH")
        recorder.close()


class ArticulationFrameSourceTest(unittest.TestCase):
    def test_nan_targets_hold_the_observed_joint_position(self) -> None:
        robot = SimpleNamespace(
            dof_names=["joint_0", "joint_1", "joint_2"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.asarray([1.0, 2.0, 3.0]),
            ),
        )
        controller = SimpleNamespace(
            get_applied_action=lambda: SimpleNamespace(
                joint_positions=np.asarray([4.0, np.nan, 6.0]),
            ),
        )

        state, action = ArticulationFrameSource(robot, controller).capture()

        np.testing.assert_array_equal(state, np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(action, np.asarray([4.0, 2.0, 6.0]))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(action.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
