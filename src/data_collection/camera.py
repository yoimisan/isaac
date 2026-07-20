"""RGB camera sources driven by the application's existing render loop."""

from __future__ import annotations

from collections.abc import Mapping

import carb
import numpy as np
import isaacsim.core.experimental.utils.prim as prim_utils
from isaacsim.core.experimental.utils.transform import look_at_quaternion
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

from data_collection.config import CameraConfig


class RgbCameraRig:
    """Own world-fixed and robot-mounted RTX RGB cameras."""

    def __init__(
        self,
        camera_configs: tuple[CameraConfig, ...],
        dlss_exec_mode: int,
    ) -> None:
        self._configs = camera_configs
        self._sensors: dict[str, CameraSensor] = {}
        # Quality mode is the Isaac Sim SDG recommendation and prevents DLSS
        # from rendering 640x480 products below its minimum input dimensions.
        carb.settings.get_settings().set(
            "rtx/post/dlss/execMode",
            dlss_exec_mode,
        )
        for config in camera_configs:
            pose_arguments = self._pose_arguments(config)
            camera = RtxCamera(
                config.prim_path,
                # Autotrigger on every rendered update. Dataset sampling is
                # decimated separately, avoiding camera/control phase drift.
                tick_rate=0.0,
                **pose_arguments,
            )
            # Experimental Camera accepts optical lengths in scene units. This
            # application uses a meter stage, so 24 mm is configured as 0.024.
            camera.camera.set_focal_lengths(config.focal_length_m)
            camera.camera.set_apertures(
                horizontal_apertures=config.horizontal_aperture_m
            )
            camera.camera.set_clipping_ranges(*config.clipping_range)
            self._sensors[config.name] = CameraSensor(
                camera,
                resolution=config.resolution,
                annotators=["rgb"],
            )

    @staticmethod
    def _pose_arguments(config: CameraConfig) -> dict[str, np.ndarray]:
        if config.pose_frame == "world":
            position = np.asarray(config.position, dtype=np.float32)
            orientation = look_at_quaternion(
                eye=position,
                target=np.asarray(config.look_at, dtype=np.float32),
                device="cpu",
            ).numpy()
            return {
                "positions": position,
                "orientations": np.asarray(orientation, dtype=np.float32),
            }

        parent_path = config.prim_path.rsplit("/", maxsplit=1)[0]
        if not prim_utils.get_prim_at_path(parent_path).IsValid():
            raise RuntimeError(
                f"Camera {config.name!r} parent prim does not exist: {parent_path}."
            )
        return {
            "translations": np.asarray(config.translation, dtype=np.float32),
            "orientations": np.asarray(config.orientation, dtype=np.float32),
        }

    @property
    def configs(self) -> tuple[CameraConfig, ...]:
        return self._configs

    @property
    def features(self) -> dict[str, dict]:
        """Return the visual feature schema used by the offline exporter."""
        return {
            config.feature_key: {
                "dtype": "video",
                "shape": [*config.resolution, 3],
                "names": ["height", "width", "channels"],
            }
            for config in self._configs
        }

    def capture(self) -> Mapping[str, np.ndarray] | None:
        """Return one synchronized RGB sample, or None during sensor warm-up."""
        images: dict[str, np.ndarray] = {}
        for config in self._configs:
            data, _ = self._sensors[config.name].get_data("rgb")
            if data is None:
                return None
            image = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
            image = np.ascontiguousarray(image, dtype=np.uint8).copy()
            expected_shape = (*config.resolution, 3)
            if image.shape != expected_shape:
                raise RuntimeError(
                    f"Camera {config.name!r} returned {image.shape}; "
                    f"expected {expected_shape}."
                )
            images[config.feature_key] = image
        return images

    def close(self) -> None:
        """Drop references; CameraSensor teardown detaches its render product."""
        self._sensors.clear()
