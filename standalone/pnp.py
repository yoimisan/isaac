"""Launch the standalone Franka pick-and-place application."""

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})


def main() -> None:
    """Enable the Franka extension and run the application loop."""
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.robot.manipulators.examples")

    from pick_place.app import run

    run(simulation_app)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
    finally:
        simulation_app.close()
