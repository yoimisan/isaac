"""Validate and replay one Isaac-scenes staging episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_collection.config import DEFAULT_RENDERER
from data_collection.replay import (
    StagingEpisode,
    validate_episode,
    write_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "logs"
        / "data_collection"
        / "pnp_raw",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check arrays and RGB files without launching Isaac Sim.",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip decoding every recorded PNG during the integrity pass.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=("action", "state"), default="action")
    parser.add_argument(
        "--scene-mode",
        choices=("trace", "initial"),
        default="trace",
        help=(
            "trace corrects recorded objects each frame; initial sets only "
            "frame zero and leaves subsequent motion to physics."
        ),
    )
    parser.add_argument("--no-image-compare", action="store_true")
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=None,
        help="0 runs as fast as possible; defaults to 1 in GUI and 0 headless.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--joint-error-threshold", type=float, default=0.05)
    parser.add_argument("--image-mae-threshold", type=float)
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults inside the selected episode directory.",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = _parse_args()
    episode = StagingEpisode.load(args.dataset_root, args.episode)
    integrity = validate_episode(
        episode,
        verify_images=not args.skip_image_verification,
    )
    report_path = args.report or (
        episode.episode_path / f"replay-report-{args.mode}.json"
    )
    report: dict = {"integrity": integrity.to_dict(), "replay": None}
    if args.validate_only or not integrity.ok:
        report["ok"] = integrity.ok
        write_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Replay report: {report_path.expanduser().resolve()}")
        return 0 if integrity.ok else 1

    from isaacsim import SimulationApp

    camera_resolution = episode.camera_metadata[0]["resolution"]
    rendering = episode.dataset_metadata.get("rendering", {})
    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": rendering.get("renderer", DEFAULT_RENDERER),
            "width": int(camera_resolution[1]),
            "height": int(camera_resolution[0]),
        }
    )
    replayer = None
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.simulation_manager import SimulationManager

        enable_extension("isaacsim.robot.manipulators.examples")

        if args.mode == "state":
            SimulationManager.enable_fabric(True)

        from data_collection.replay_runtime import (
            IsaacEpisodeReplayer,
            ReplayOptions,
        )
        from pick_place.replay import build_replay_scene

        task_id = episode.dataset_metadata.get("task_id", "pick_place")
        if task_id != "pick_place":
            raise NotImplementedError(
                f"No replay scene adapter is registered for task {task_id!r}."
            )

        playback_speed = args.playback_speed
        if playback_speed is None:
            playback_speed = 0.0 if args.headless else 1.0
        replayer = IsaacEpisodeReplayer(
            simulation_app,
            episode,
            ReplayOptions(
                mode=args.mode,
                scene_mode=args.scene_mode,
                compare_images=not args.no_image_compare,
                image_stride=args.image_stride,
                playback_speed=playback_speed,
                max_frames=args.max_frames,
                joint_error_threshold=args.joint_error_threshold,
                image_mae_threshold=args.image_mae_threshold,
            ),
            scene=build_replay_scene(),
        )
        report["replay"] = replayer.replay()
        report["ok"] = bool(integrity.ok and report["replay"]["ok"])
        write_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(
            f"Replay report: {report_path.expanduser().resolve()}",
            flush=True,
        )
        return 0 if report["ok"] else 1
    except Exception as error:
        report["replay"] = {
            "ok": False,
            "errors": [f"{type(error).__name__}: {error}"],
        }
        report["ok"] = False
        write_report(report_path, report)
        raise
    finally:
        if replayer is not None:
            replayer.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
