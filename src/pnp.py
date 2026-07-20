"""Launch the standalone Franka pick-and-place application."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "logs"
        / "data_collection"
        / "pnp_raw",
    )
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=1,
        help="Number of successful episodes to collect before exiting.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--save-failed-episodes", action="store_true")
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()
simulation_app = SimulationApp({"headless": args.headless})


def main() -> None:
    """Enable the Franka extension and run the application loop."""
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.robot.manipulators.examples")

    from data_collection.config import DataCollectionConfig, default_cameras
    from pick_place.app import run

    data_collection_config = None
    if args.record:
        resolution = (args.camera_height, args.camera_width)
        data_collection_config = DataCollectionConfig(
            enabled=True,
            root=args.record_root,
            fps=args.record_fps,
            num_episodes=args.record_episodes,
            save_failed_episodes=args.save_failed_episodes,
            cameras=default_cameras(resolution),
        )

    run(simulation_app, data_collection_config=data_collection_config)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
    finally:
        simulation_app.close()
