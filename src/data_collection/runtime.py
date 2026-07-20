"""Application-loop adapter for sidecar data collection."""

from __future__ import annotations

from typing import Any

import carb
import numpy as np
from isaacsim.core.api import World

from data_collection.camera import RgbCameraRig
from data_collection.config import DataCollectionConfig
from data_collection.staging import ArticulationFrameSource, StagingEpisodeRecorder


class DataCollectionRuntime:
    """Translate app lifecycle events into recorder lifecycle events."""

    def __init__(
        self,
        config: DataCollectionConfig,
        world: World,
        robot: Any,
        articulation_controller: Any,
    ) -> None:
        self._config = config
        self._validate_recording_rate(world)
        self._cameras = RgbCameraRig(config.cameras)
        try:
            self._recorder = StagingEpisodeRecorder(
                config,
                articulation=ArticulationFrameSource(
                    robot,
                    articulation_controller,
                ),
                cameras=self._cameras,
            )
            self._warm_up_cameras(world)
        except Exception:
            if hasattr(self, "_recorder"):
                self._recorder.close()
            self._cameras.close()
            raise
        carb.log_info(f"Recording staging episodes under {self._recorder.root}.")

    @property
    def num_episodes(self) -> int:
        """Return the requested number of successful automatic rollouts."""
        return self._config.num_episodes

    def begin_episode(self, simulation_time: float) -> None:
        self._recorder.begin_episode(simulation_time)

    def before_world_reset(self) -> None:
        """Publish or discard the interrupted episode before a reset."""
        if not self._recorder.is_episode_active:
            return
        if self._config.save_failed_episodes:
            self._recorder.finish_episode(
                success=False,
                end_reason="simulation_reset",
            )
        else:
            self._recorder.abort_episode()

    def after_world_reset(self, world: World) -> None:
        """Discard the first few rendered frames after a world reset."""
        self._settle_cameras(world)

    def record_frame(self, simulation_time: float, task_state: str) -> bool:
        return self._recorder.record_frame(simulation_time, task_state)

    def finish_successful_episode(self) -> None:
        if not self._recorder.is_episode_active:
            return
        episode_path = self._recorder.finish_episode(
            success=True,
            end_reason="task_success",
        )
        carb.log_info(f"Saved successful staging episode to {episode_path}.")

    def close(self) -> None:
        try:
            if (
                self._config.save_failed_episodes
                and self._recorder.is_episode_active
            ):
                self._recorder.finish_episode(
                    success=False,
                    end_reason="application_shutdown",
                )
        finally:
            try:
                self._recorder.close()
            finally:
                self._cameras.close()

    def _warm_up_cameras(self, world: World) -> None:
        for _ in range(self._config.camera_warmup_max_steps):
            world.step(render=True)
            if self._cameras.capture() is not None:
                self._settle_cameras(world)
                return
        raise RuntimeError(
            "Data-collection cameras produced no RGB after "
            f"{self._config.camera_warmup_max_steps} rendered steps."
        )

    def _settle_cameras(self, world: World) -> None:
        for _ in range(self._config.camera_warmup_settle_steps):
            world.step(render=True)

    def _validate_recording_rate(self, world: World) -> None:
        simulation_fps = 1.0 / world.get_physics_dt()
        if not np.isclose(
            self._config.fps,
            simulation_fps,
            atol=1e-6,
        ):
            raise ValueError(
                f"Recording fps ({self._config.fps}) must equal the physics/control "
                f"rate ({simulation_fps:g} Hz). Simple frame decimation would lose "
                "intermediate actions."
            )
