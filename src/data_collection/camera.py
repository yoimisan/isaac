"""RGB camera sources driven by the application's existing render loop."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from isaacsim.core.experimental.utils.transform import look_at_quaternion
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

from data_collection.config import CameraConfig


class RgbCameraRig:
    """Own fixed RTX cameras and expose copied CPU RGB observations."""

    def __init__(self, camera_configs: tuple[CameraConfig, ...]) -> None:
        self._configs = camera_configs
        self._sensors: dict[str, CameraSensor] = {}
        for config in camera_configs:
            orientation = look_at_quaternion(
                eye=np.asarray(config.position, dtype=np.float32),
                target=np.asarray(config.look_at, dtype=np.float32),
                device="cpu",
            ).numpy()
            camera = RtxCamera(
                config.prim_path,
                # Autotrigger on every rendered update. Dataset sampling is
                # decimated separately, avoiding camera/control phase drift.
                tick_rate=0.0,
                positions=np.asarray(config.position, dtype=np.float32),
                orientations=np.asarray(orientation, dtype=np.float32),
            )
            camera.camera.set_focal_lengths(config.focal_length)
            camera.camera.set_clipping_ranges(*config.clipping_range)
            self._sensors[config.name] = CameraSensor(
                camera,
                resolution=config.resolution,
                annotators=["rgb"],
            )

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
