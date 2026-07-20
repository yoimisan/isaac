"""Tests for the Isaac-independent staging layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_collection.config import (
    CameraConfig,
    DataCollectionConfig,
    default_cameras,
)
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
    def __init__(
        self,
        cameras: CameraConfig | tuple[CameraConfig, ...],
    ) -> None:
        self._cameras = cameras if isinstance(cameras, tuple) else (cameras,)
        self.features = {
            camera.feature_key: {
                "dtype": "video",
                "shape": [*camera.resolution, 3],
                "names": ["height", "width", "channels"],
            }
            for camera in self._cameras
        }

    def capture(self) -> dict[str, np.ndarray]:
        return {
            camera.feature_key: np.zeros(
                (*camera.resolution, 3),
                dtype=np.uint8,
            )
            for camera in self._cameras
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

    def test_metadata_preserves_world_and_parent_relative_poses(self) -> None:
        cameras = default_cameras(resolution=(4, 5))
        config = DataCollectionConfig(root=self.root, cameras=cameras)
        recorder = StagingEpisodeRecorder(
            config,
            articulation=_FakeArticulationSource(),
            cameras=_FakeCameraRig(cameras),
        )
        recorder.close()

        metadata = json.loads(
            (self.root / "dataset.json").read_text(encoding="utf-8")
        )
        wrist, external_pos_y, external_neg_y = metadata["cameras"]
        self.assertEqual(wrist["pose_frame"], "parent")
        self.assertIsNone(wrist["position"])
        self.assertEqual(wrist["translation"], [0.06, 0.0, 0.035])
        self.assertEqual(external_pos_y["pose_frame"], "world")
        self.assertIsNone(external_pos_y["translation"])
        self.assertEqual(
            external_pos_y["position"][1],
            -external_neg_y["position"][1],
        )
        self.assertEqual(metadata["collection_mode"], "clean")
        self.assertIsNone(metadata["perturbation"])

    def test_metadata_records_perturbation_configuration(self) -> None:
        config = DataCollectionConfig(
            root=self.root,
            cameras=(self.camera,),
            collection_mode="perturbed",
            perturbation_seed=7,
            perturbation_attack_count_range=(1, 3),
        )
        recorder = StagingEpisodeRecorder(
            config,
            articulation=_FakeArticulationSource(),
            cameras=_FakeCameraRig(self.camera),
        )
        recorder.close()

        metadata = json.loads(
            (self.root / "dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["collection_mode"], "perturbed")
        self.assertEqual(
            metadata["perturbation"],
            {"seed": 7, "attack_count_range": [1, 3]},
        )


class DataCollectionConfigTest(unittest.TestCase):
    def test_default_rig_has_wrist_and_symmetric_external_cameras(self) -> None:
        config = DataCollectionConfig()
        cameras = config.cameras

        self.assertEqual(config.dlss_exec_mode, 2)
        self.assertTrue(all(camera.resolution == (480, 640) for camera in cameras))
        self.assertEqual(
            [camera.name for camera in cameras],
            ["wrist", "external_pos_y", "external_neg_y"],
        )
        self.assertEqual(cameras[0].pose_frame, "parent")
        self.assertEqual(cameras[1].pose_frame, "world")
        self.assertEqual(cameras[1].position[0], cameras[2].position[0])
        self.assertEqual(cameras[1].position[1], -cameras[2].position[1])
        self.assertEqual(cameras[1].focal_length_m, 0.028)
        self.assertEqual(cameras[2].focal_length_m, 0.028)

    def test_camera_pose_must_be_world_or_parent_relative(self) -> None:
        with self.assertRaisesRegex(ValueError, "Camera pose is missing"):
            CameraConfig(name="missing", prim_path="/World/Missing")
        with self.assertRaisesRegex(ValueError, "either position/look_at"):
            CameraConfig(
                name="mixed",
                prim_path="/World/Mixed",
                position=(1.0, 0.0, 1.0),
                look_at=(0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

    def test_episode_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_episodes must be positive"):
            DataCollectionConfig(num_episodes=0)

    def test_perturbed_collection_requires_valid_attack_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires perturbation_attack_count_range",
        ):
            DataCollectionConfig(collection_mode="perturbed")

        config = DataCollectionConfig(
            collection_mode="perturbed",
            perturbation_seed=7,
            perturbation_attack_count_range=(1, 3),
        )
        self.assertEqual(config.perturbation_attack_count_range, (1, 3))

    def test_dlss_mode_must_be_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "dlss_exec_mode must be"):
            DataCollectionConfig(dlss_exec_mode=4)


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
