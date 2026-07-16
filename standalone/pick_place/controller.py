"""CuRobo-backed state-machine controller for pick-and-place."""

from __future__ import annotations

import carb
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

from pick_place.curobo_planner import CuroboPlanner
from pick_place.states import (
    ApproachState,
    DescendState,
    GraspState,
    IdleState,
    LiftState,
    PlaceState,
    PickPlacePhase,
    PnPState,
    ReleaseState,
    ReturnState,
    WaitForStableState,
)


class PnPController(BaseController):
    """Execute pick-and-place as a state machine backed by CuRobo plans."""

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

        self._planner = CuroboPlanner(world.scene, robot, include_cube_in_collision=False)
        self._reset_phase_state()

    def reset(self) -> None:
        """Reset state-local data and begin a fresh approach."""
        super().reset()
        self._reset_phase_state()
        self._transition_to(PickPlacePhase.APPROACH)

    def forward(self) -> ArticulationAction | None:
        """Return the next articulation action for the active phase."""
        return self._forward_state_object()

    def is_complete(self) -> bool:
        """Return whether the cube was released and the arm returned to its reset pose."""
        return self._return_state.is_complete

    def _reset_phase_state(self) -> None:
        self._phase = PickPlacePhase.IDLE
        base_link = self._planner.robot_config["kinematics"]["base_link"]
        self._base_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/{base_link}",
            name="franka_base_link",
        )
        self._tool_center_prim = SingleXFormPrim(
            prim_path=f"{self._robot.prim_path}/panda_hand/tool_center",
            name="franka_tool_center",
        )
        reset_arm_positions = self._robot.get_joints_state().positions[
            self._planner.isaac_arm_joint_indices
        ].copy()
        approach_state = ApproachState(
            world=self._world,
            robot=self._robot,
            cube=self._cube,
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            approach_tolerance=self._approach_tolerance,
        )
        wait_for_stable_state = WaitForStableState(
            robot=self._robot,
            cube=self._cube,
            arm_joint_indices=self._planner.isaac_arm_joint_indices,
        )
        descend_state = DescendState(
            cube=self._cube,
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            approach_tolerance=self._approach_tolerance,
        )
        lift_state = LiftState(
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            lift_offset=self._LIFT_OFFSET,
            approach_tolerance=self._approach_tolerance,
        )
        place_state = PlaceState(
            world=self._world,
            cube=self._cube,
            planner=self._planner,
            base_prim=self._base_prim,
            tool_center_prim=self._tool_center_prim,
            approach_tolerance=self._approach_tolerance,
        )
        self._return_state = ReturnState(
            robot=self._robot,
            planner=self._planner,
            reset_arm_positions=reset_arm_positions,
            approach_tolerance=self._approach_tolerance,
        )
        states: tuple[PnPState, ...] = (
            IdleState(),
            wait_for_stable_state,
            approach_state,
            descend_state,
            GraspState(robot=self._robot, planner=self._planner),
            lift_state,
            place_state,
            ReleaseState(robot=self._robot, planner=self._planner),
            self._return_state,
        )
        self._state_objects = {state.phase: state for state in states}
        if set(self._state_objects) != set(PickPlacePhase):
            missing_phases = set(PickPlacePhase) - set(self._state_objects)
            raise RuntimeError(
                f"Missing pick-and-place state objects: {missing_phases}"
            )

    def _forward_state_object(self) -> ArticulationAction | None:
        """Validate and advance the active state object by one tick."""
        state = self._state_objects[self._phase]
        perturbation = state.detect_perturbation()
        if perturbation is not None:
            recovery_phase = state.recovery_phase(perturbation)
            carb.log_warn(
                f"PnP state {self._phase.name} invalidated: "
                f"{perturbation.reason}; metrics={dict(perturbation.metrics)}"
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
            next_state.enter()

        if previous_phase is not next_phase:
            carb.log_info(
                f"PnP phase transition: {previous_phase.name} -> {next_phase.name}"
            )
