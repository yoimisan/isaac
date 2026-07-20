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
    parser.add_argument(
        "--perturb",
        "--perturbation",
        dest="perturb",
        action="store_true",
        help="Enable state-aware Naughty Ghost cube perturbations.",
    )
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--perturb-min-attacks", type=int, default=0)
    parser.add_argument("--perturb-max-attacks", type=int, default=5)
    args, _ = parser.parse_known_args()
    if args.perturb_min_attacks < 0:
        parser.error("--perturb-min-attacks must be nonnegative.")
    if args.perturb_max_attacks < args.perturb_min_attacks:
        parser.error(
            "--perturb-max-attacks must be greater than or equal to "
            "--perturb-min-attacks."
        )
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
            collection_mode="perturbed" if args.perturb else "clean",
            perturbation_seed=args.perturb_seed if args.perturb else None,
            perturbation_attack_count_range=(
                (args.perturb_min_attacks, args.perturb_max_attacks)
                if args.perturb
                else None
            ),
        )

    run(
        simulation_app,
        data_collection_config=data_collection_config,
        enable_naughty_ghost=args.perturb,
        naughty_seed=args.perturb_seed,
        naughty_attack_count_range=(
            args.perturb_min_attacks,
            args.perturb_max_attacks,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
    finally:
        simulation_app.close()
