"""Configuration values for simulation-side data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    """Describe one fixed RGB camera without binding it to a task."""

    name: str
    prim_path: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    resolution: tuple[int, int] = (240, 320)
    focal_length: float = 24.0
    clipping_range: tuple[float, float] = (0.01, 100.0)

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                "Camera name must contain only letters, numbers, and underscores."
            )
        if not self.prim_path.startswith("/"):
            raise ValueError("Camera prim_path must be an absolute USD path.")
        if any(dimension <= 0 for dimension in self.resolution):
            raise ValueError(f"Camera resolution must be positive: {self.resolution}.")
        if self.focal_length <= 0.0:
            raise ValueError("Camera focal_length must be positive.")
        near, far = self.clipping_range
        if near <= 0.0 or far <= near:
            raise ValueError(f"Invalid camera clipping range: {self.clipping_range}.")

    @property
    def feature_key(self) -> str:
        """Return the conventional LeRobot feature name for this camera."""
        return f"observation.images.{self.name}"


def default_cameras(
    resolution: tuple[int, int] = (240, 320),
) -> tuple[CameraConfig, ...]:
    """Return the conservative first camera setup for tabletop tasks."""
    return (
        CameraConfig(
            name="overview",
            prim_path="/World/DataCollection/Cameras/Overview",
            position=(1.15, 0.75, 0.85),
            look_at=(0.48, 0.0, 0.08),
            resolution=resolution,
        ),
    )


@dataclass(frozen=True)
class DataCollectionConfig:
    """Configure the Isaac-side staging recorder."""

    enabled: bool = False
    root: Path = Path("logs/data_collection/pnp_raw")
    fps: int = 60
    task: str = "Pick up the cube and place it in the target region"
    robot_type: str = "franka"
    num_episodes: int = 1
    save_failed_episodes: bool = False
    image_writer_threads: int = 2
    max_pending_image_writes: int = 64
    camera_warmup_max_steps: int = 10
    camera_warmup_settle_steps: int = 3
    cameras: tuple[CameraConfig, ...] = field(default_factory=default_cameras)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"Data-collection fps must be positive; got {self.fps}.")
        if self.num_episodes <= 0:
            raise ValueError(
                "Data-collection num_episodes must be positive; "
                f"got {self.num_episodes}."
            )
        if not self.cameras:
            raise ValueError("At least one data-collection camera is required.")
        camera_names = [camera.name for camera in self.cameras]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError(f"Camera names must be unique; got {camera_names}.")
        if self.image_writer_threads <= 0:
            raise ValueError("image_writer_threads must be positive.")
        if self.max_pending_image_writes <= 0:
            raise ValueError("max_pending_image_writes must be positive.")
        if self.camera_warmup_max_steps <= 0:
            raise ValueError("camera_warmup_max_steps must be positive.")
        if self.camera_warmup_settle_steps < 0:
            raise ValueError("camera_warmup_settle_steps cannot be negative.")
