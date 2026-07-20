"""CuRobo-backed state-machine controller for pick-and-place."""

from __future__ import annotations

import carb
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states import (
    ApproachState,
    CubeCollisionMode,
    DescendState,
    GraspState,
    IdleState,
    LiftState,
    PlaceState,
    PickPlacePhase,
    PnPState,
    ReleaseState,
    ReturnState,
)


class PnPController(BaseController):
    """Execute pick-and-place as a state machine backed by CuRobo plans."""

    _GRASP_TOLERANCE = 0.06
    _LIFT_OFFSET = 0.15

    def __init__(
        self,
        name: str,
        robot: Franka,
        cube: DynamicCuboid,
        world: World,
        approach_tolerance: float = 0.02,
    ) -> None:
        super().__init__(name)
        self._robot = robot
        self._cube = cube
        self._world = world
        self._approach_tolerance = approach_tolerance
        self._phase = PickPlacePhase.IDLE

        self._planner = CuroboPlanner(world.scene, robot)
        self._planner.register_dynamic_obstacle(cube)
        self._reset_phase_state()

    def reset(self) -> None:
        """Reset state-local data and begin a fresh approach."""
        super().reset()
        self._planner.reset_episode()
        self._reset_phase_state()
        self._transition_to(PickPlacePhase.APPROACH)

    def forward(self) -> ArticulationAction | None:
        """Return the next articulation action for the active phase."""
        return self._forward_state_object()

    @property
    def phase(self) -> PickPlacePhase:
        """Return the active phase for observation and validation tooling."""
        return self._phase

    def is_complete(self) -> bool:
        """Return whether the cube was released and the arm returned to its reset pose."""
        return self._return_state.is_complete

    def is_current_state_success(self) -> bool:
        """Return whether the active state's goal is currently satisfied."""
        return self._state_objects[self._phase].is_success()

    def _reset_phase_state(self) -> None:
        self._phase = PickPlacePhase.IDLE
        reset_arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ].copy()
        approach_state = ApproachState(
            world=self._world,
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            approach_tolerance=self._approach_tolerance,
        )
        descend_state = DescendState(
            cube=self._cube,
            planner=self._planner,
            approach_tolerance=self._approach_tolerance,
        )
        lift_state = LiftState(
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            lift_offset=self._LIFT_OFFSET,
            approach_tolerance=self._approach_tolerance,
            grasp_tolerance=self._GRASP_TOLERANCE,
        )
        place_state = PlaceState(
            world=self._world,
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            approach_tolerance=self._approach_tolerance,
            grasp_tolerance=self._GRASP_TOLERANCE,
        )
        self._return_state = ReturnState(
            world=self._world,
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            reset_arm_positions=reset_arm_positions,
            approach_tolerance=self._approach_tolerance,
        )
        states: tuple[PnPState, ...] = (
            IdleState(),
            approach_state,
            descend_state,
            GraspState(
                robot=self._robot,
                cube=self._cube,
                planner=self._planner,
                grasp_tolerance=self._GRASP_TOLERANCE,
            ),
            lift_state,
            place_state,
            ReleaseState(
                world=self._world,
                robot=self._robot,
                cube=self._cube,
                planner=self._planner,
                placement_tolerance=self._approach_tolerance,
                grasp_tolerance=self._GRASP_TOLERANCE,
            ),
            self._return_state,
        )
        self._state_objects = {state.phase: state for state in states}
        if set(self._state_objects) != set(PickPlacePhase):
            missing_phases = set(PickPlacePhase) - set(self._state_objects)
            raise RuntimeError(
                f"Missing pick-and-place state objects: {missing_phases}"
            )
        for state in states:
            if not isinstance(
                getattr(state, "cube_collision_mode", None),
                CubeCollisionMode,
            ):
                raise RuntimeError(
                    f"{type(state).__name__} has no valid cube collision mode."
                )

    def _forward_state_object(self) -> ArticulationAction | None:
        """Validate and advance the active state object by one tick."""
        state = self._state_objects[self._phase]
        perturbation = state.detect_perturbation()
        if perturbation is not None:
            recovery_phase = state.recovery_phase(perturbation)
            carb.log_warn(
                f"PnP state {self._phase.name} invalidated: "
                f"{perturbation.reason}; metrics={dict(perturbation.metrics)}."
            )
            self._transition_to(recovery_phase)
            return None

        step = state.update()
        if step.next_phase is not None:
            self._transition_to(step.next_phase)
        return step.action

    def _transition_to(self, next_phase: PickPlacePhase) -> None:
        """Perform state lifecycle hooks and record one state transition."""
        previous_phase = self._phase
        previous_state = self._state_objects.get(previous_phase)
        if previous_state is not None:
            previous_state.exit()

        self._phase = next_phase
        next_state = self._state_objects.get(next_phase)
        if next_state is not None:
            self._apply_cube_collision_mode(next_state.cube_collision_mode)
            next_state.enter()

        if previous_phase is not next_phase:
            carb.log_warn(
                f"PnP phase transition: {previous_phase.name} -> {next_phase.name}"
            )

    def _apply_cube_collision_mode(self, mode: CubeCollisionMode) -> None:
        """Apply one state's cube policy to the CuRobo collision world."""
        world_collision_enabled = mode is CubeCollisionMode.WORLD_OBSTACLE
        self._planner.set_obstacle_collision_enabled(
            self._cube.prim_path,
            world_collision_enabled,
        )
