"""Configuration values for simulation-side data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, sqrt
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    """Describe one world-fixed or parent-relative RGB camera."""

    name: str
    prim_path: str
    position: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    translation: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None
    resolution: tuple[int, int] = (480, 640)
    focal_length_m: float = 0.024
    horizontal_aperture_m: float = 0.036
    clipping_range: tuple[float, float] = (0.01, 100.0)

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                "Camera name must contain only letters, numbers, and underscores."
            )
        if not self.prim_path.startswith("/"):
            raise ValueError("Camera prim_path must be an absolute USD path.")
        has_world_pose = self.position is not None or self.look_at is not None
        has_local_pose = (
            self.translation is not None or self.orientation is not None
        )
        if has_world_pose and has_local_pose:
            raise ValueError(
                "Camera must use either position/look_at or "
                "translation/orientation, not both."
            )
        if has_world_pose:
            if self.position is None or self.look_at is None:
                raise ValueError("World-fixed cameras require position and look_at.")
            if len(self.position) != 3 or len(self.look_at) != 3:
                raise ValueError("Camera position and look_at must be 3D vectors.")
            if self.position == self.look_at:
                raise ValueError("Camera position and look_at must differ.")
        elif has_local_pose:
            if self.translation is None or self.orientation is None:
                raise ValueError(
                    "Parent-relative cameras require translation and orientation."
                )
            if len(self.translation) != 3 or len(self.orientation) != 4:
                raise ValueError(
                    "Camera translation must be 3D and orientation must be wxyz."
                )
            quaternion_norm = sqrt(sum(value * value for value in self.orientation))
            if not isclose(quaternion_norm, 1.0, abs_tol=1e-5):
                raise ValueError("Camera orientation quaternion must be normalized.")
        else:
            raise ValueError("Camera pose is missing.")
        if any(dimension <= 0 for dimension in self.resolution):
            raise ValueError(f"Camera resolution must be positive: {self.resolution}.")
        if self.focal_length_m <= 0.0:
            raise ValueError("Camera focal_length_m must be positive.")
        if self.horizontal_aperture_m <= 0.0:
            raise ValueError("Camera horizontal_aperture_m must be positive.")
        near, far = self.clipping_range
        if near <= 0.0 or far <= near:
            raise ValueError(f"Invalid camera clipping range: {self.clipping_range}.")

    @property
    def feature_key(self) -> str:
        """Return the conventional LeRobot feature name for this camera."""
        return f"observation.images.{self.name}"

    @property
    def pose_frame(self) -> str:
        """Return whether the configured pose is in world or parent coordinates."""
        return "world" if self.position is not None else "parent"


def default_cameras(
    resolution: tuple[int, int] = (480, 640),
) -> tuple[CameraConfig, ...]:
    """Return one wrist view and two symmetric external tabletop views."""
    half_sqrt_two = sqrt(0.5)
    return (
        CameraConfig(
            name="wrist",
            prim_path="/World/Franka/panda_hand/DataCollectionWristCamera",
            translation=(0.06, 0.0, 0.035),
            # Camera -Z looks along panda_hand +Z. Image-right follows the
            # gripper's +Y opening axis and image-up follows hand +X.
            orientation=(0.0, half_sqrt_two, half_sqrt_two, 0.0),
            resolution=resolution,
            focal_length_m=0.018,
            clipping_range=(0.01, 5.0),
        ),
        CameraConfig(
            name="external_pos_y",
            prim_path="/World/DataCollection/Cameras/ExternalPosY",
            position=(0.95, 0.55, 0.75),
            look_at=(0.48, 0.0, 0.08),
            resolution=resolution,
            focal_length_m=0.028,
        ),
        CameraConfig(
            name="external_neg_y",
            prim_path="/World/DataCollection/Cameras/ExternalNegY",
            position=(0.95, -0.55, 0.75),
            look_at=(0.48, 0.0, 0.08),
            resolution=resolution,
            focal_length_m=0.028,
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
    collection_mode: str = "clean"
    perturbation_seed: int | None = None
    perturbation_attack_count_range: tuple[int, int] | None = None
    num_episodes: int = 1
    save_failed_episodes: bool = False
    dlss_exec_mode: int = 2
    image_writer_threads: int = 4
    max_pending_image_writes: int = 128
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
        if self.collection_mode not in {"clean", "perturbed"}:
            raise ValueError(
                "collection_mode must be either 'clean' or 'perturbed'."
            )
        if self.collection_mode == "clean":
            if self.perturbation_attack_count_range is not None:
                raise ValueError(
                    "Clean collection cannot define a perturbation attack range."
                )
            if self.perturbation_seed is not None:
                raise ValueError(
                    "Clean collection cannot define a perturbation seed."
                )
        elif self.perturbation_attack_count_range is None:
            raise ValueError(
                "Perturbed collection requires perturbation_attack_count_range."
            )
        if self.perturbation_attack_count_range is not None:
            if len(self.perturbation_attack_count_range) != 2:
                raise ValueError(
                    "perturbation_attack_count_range must contain two values."
                )
            minimum_attacks, maximum_attacks = (
                self.perturbation_attack_count_range
            )
            if minimum_attacks < 0 or minimum_attacks > maximum_attacks:
                raise ValueError(
                    "perturbation_attack_count_range must contain nonnegative "
                    "increasing bounds."
                )
        if not self.cameras:
            raise ValueError("At least one data-collection camera is required.")
        camera_names = [camera.name for camera in self.cameras]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError(f"Camera names must be unique; got {camera_names}.")
        if self.dlss_exec_mode not in range(4):
            raise ValueError(
                "dlss_exec_mode must be 0 (Performance), 1 (Balanced), "
                "2 (Quality), or 3 (Auto)."
            )
        if self.image_writer_threads <= 0:
            raise ValueError("image_writer_threads must be positive.")
        if self.max_pending_image_writes <= 0:
            raise ValueError("max_pending_image_writes must be positive.")
        if self.camera_warmup_max_steps <= 0:
            raise ValueError("camera_warmup_max_steps must be positive.")
        if self.camera_warmup_settle_steps < 0:
            raise ValueError("camera_warmup_settle_steps cannot be negative.")
