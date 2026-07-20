"""PnP-specific naughty policy for disturbing the cube at useful phases."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from adversary.types import (
    DisturbanceChannel,
    DisturbanceCommand,
    ObjectPoseOffset,
    TaskObjectDisturbanceContext,
)


@dataclass(frozen=True)
class PnPDisturbanceScenario:
    """One phase-specific cube disturbance recipe."""

    state_name: str
    trigger_after_steps: int
    displacement: float = 0.10
    selection_weight: float = 1.0


_ATTACKABLE_STATE_NAMES = frozenset(
    {"APPROACH", "DESCEND", "GRASP", "LIFT", "PLACE", "RELEASE", "RETURN"}
)
_DEFAULT_SCENARIOS = (
    PnPDisturbanceScenario("APPROACH", trigger_after_steps=2),
    PnPDisturbanceScenario("DESCEND", trigger_after_steps=2),
    PnPDisturbanceScenario("LIFT", trigger_after_steps=2),
    PnPDisturbanceScenario("PLACE", trigger_after_steps=2),
    PnPDisturbanceScenario("RETURN", trigger_after_steps=2),
)


class PickPlaceTaskObjectDisturbancePolicy:
    """Attempt at most one state-targeted cube attack per PnP episode."""

    def __init__(
        self,
        *,
        target_name: str,
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        seed: int | None = 1,
        scenarios: Sequence[PnPDisturbanceScenario] | None = None,
    ) -> None:
        self._target_name = target_name
        self._workspace_x = workspace_x
        self._workspace_y = workspace_y
        self._scenarios = tuple(
            _DEFAULT_SCENARIOS if scenarios is None else scenarios
        )
        self._validate_configuration()
        self._rng = random.Random(seed)
        self._selected_scenario: PnPDisturbanceScenario | None = None
        self._active_state: str | None = None
        self._steps_in_state = 0
        self._has_fired = False

    @property
    def selected_state_name(self) -> str | None:
        """Return the phase selected for the current episode."""
        if self._selected_scenario is None:
            return None
        return self._selected_scenario.state_name

    def reset(self) -> None:
        """Sample one reproducible attack scenario for a new episode."""
        self._selected_scenario = self._rng.choices(
            self._scenarios,
            weights=[scenario.selection_weight for scenario in self._scenarios],
            k=1,
        )[0]
        self._active_state = None
        self._steps_in_state = 0
        self._has_fired = False

    def propose(
        self,
        context: TaskObjectDisturbanceContext,
    ) -> DisturbanceCommand | None:
        """Move the cube when the selected PnP phase reaches a safe window."""
        if self._selected_scenario is None:
            raise RuntimeError("Disturbance policy must be reset before use.")

        task_state = context.task_state
        if task_state.state_name != self._active_state:
            self._active_state = task_state.state_name
            self._steps_in_state = 1
        else:
            self._steps_in_state += 1

        scenario = self._selected_scenario
        if self._has_fired or task_state.state_name != scenario.state_name:
            return None
        if self._steps_in_state < scenario.trigger_after_steps:
            return None

        try:
            target = context.objects[self._target_name]
        except KeyError as error:
            raise KeyError(
                f"Missing observation for task object {self._target_name!r}"
            ) from error

        position_offset = self._choose_workspace_offset(
            target.position,
            scenario.displacement,
        )
        self._has_fired = True
        return ObjectPoseOffset(
            channel=DisturbanceChannel.TASK_OBJECT,
            reason=(
                f"PnP naughty policy targeted {scenario.state_name} "
                f"at state step {self._steps_in_state}"
            ),
            target_name=self._target_name,
            position_offset=position_offset,
        )

    def _choose_workspace_offset(
        self,
        position: tuple[float, float, float],
        displacement: float,
    ) -> tuple[float, float, float]:
        candidates = (
            (displacement, 0.0, 0.0),
            (-displacement, 0.0, 0.0),
            (0.0, displacement, 0.0),
            (0.0, -displacement, 0.0),
        )
        valid_candidates = tuple(
            offset
            for offset in candidates
            if self._inside_workspace(
                position[0] + offset[0],
                position[1] + offset[1],
            )
        )
        candidate_pool = valid_candidates or candidates
        center_x = sum(self._workspace_x) / 2.0
        center_y = sum(self._workspace_y) / 2.0
        width_x = self._workspace_x[1] - self._workspace_x[0]
        width_y = self._workspace_y[1] - self._workspace_y[0]
        return min(
            candidate_pool,
            key=lambda offset: (
                ((position[0] + offset[0] - center_x) / width_x) ** 2
                + ((position[1] + offset[1] - center_y) / width_y) ** 2
            ),
        )

    def _inside_workspace(self, x: float, y: float) -> bool:
        return (
            self._workspace_x[0] <= x <= self._workspace_x[1]
            and self._workspace_y[0] <= y <= self._workspace_y[1]
        )

    def _validate_configuration(self) -> None:
        if not self._scenarios:
            raise ValueError("At least one PnP disturbance scenario is required.")
        if self._workspace_x[0] >= self._workspace_x[1]:
            raise ValueError("workspace_x must contain increasing bounds")
        if self._workspace_y[0] >= self._workspace_y[1]:
            raise ValueError("workspace_y must contain increasing bounds")

        state_names: set[str] = set()
        for scenario in self._scenarios:
            if scenario.state_name not in _ATTACKABLE_STATE_NAMES:
                raise ValueError(
                    f"Unsupported PnP disturbance state {scenario.state_name!r}"
                )
            if scenario.state_name in state_names:
                raise ValueError(
                    f"Duplicate PnP disturbance state {scenario.state_name!r}"
                )
            if scenario.trigger_after_steps < 1:
                raise ValueError("trigger_after_steps must be at least one")
            if scenario.displacement <= 0.0:
                raise ValueError("displacement must be positive")
            if scenario.selection_weight <= 0.0:
                raise ValueError("selection_weight must be positive")
            state_names.add(scenario.state_name)
