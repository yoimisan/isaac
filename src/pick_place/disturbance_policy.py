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
    """Spend a bounded attack budget on state-targeted cube disturbances."""

    def __init__(
        self,
        *,
        target_name: str,
        target_region_name: str,
        target_half_clearance: float,
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        seed: int | None = 1,
        attack_count_range: tuple[int, int] = (0, 5),
        scenarios: Sequence[PnPDisturbanceScenario] | None = None,
    ) -> None:
        self._target_name = target_name
        self._target_region_name = target_region_name
        self._target_half_clearance = target_half_clearance
        self._workspace_x = workspace_x
        self._workspace_y = workspace_y
        self._attack_count_range = attack_count_range
        self._scenarios = tuple(
            _DEFAULT_SCENARIOS if scenarios is None else scenarios
        )
        self._validate_configuration()
        self._rng = random.Random(seed)
        self._pending_scenarios: list[PnPDisturbanceScenario] = []
        self._active_state_entry: tuple[str, int] | None = None
        self._last_attacked_state_entry: tuple[str, int] | None = None
        self._steps_in_state = 0
        self._attack_budget = 0
        self._attacks_executed = 0
        self._episode_initialized = False

    @property
    def selected_state_name(self) -> str | None:
        """Return the phase targeted by the next pending attack."""
        if not self._pending_scenarios:
            return None
        return self._pending_scenarios[0].state_name

    @property
    def attack_budget(self) -> int:
        """Return the number of attacks sampled for this episode."""
        return self._attack_budget

    @property
    def attacks_executed(self) -> int:
        """Return how many attacks have produced commands this episode."""
        return self._attacks_executed

    @property
    def attacks_remaining(self) -> int:
        """Return how many sampled attacks have not fired yet."""
        return len(self._pending_scenarios)

    @property
    def pending_state_names(self) -> tuple[str, ...]:
        """Return the remaining state-targeted attack plan."""
        return tuple(
            scenario.state_name for scenario in self._pending_scenarios
        )

    def reset(self) -> None:
        """Sample a reproducible attack budget and scenario queue."""
        minimum_attacks, maximum_attacks = self._attack_count_range
        self._attack_budget = self._rng.randint(
            minimum_attacks,
            maximum_attacks,
        )
        self._pending_scenarios = self._rng.choices(
            self._scenarios,
            weights=[scenario.selection_weight for scenario in self._scenarios],
            k=self._attack_budget,
        )
        self._active_state_entry = None
        self._last_attacked_state_entry = None
        self._steps_in_state = 0
        self._attacks_executed = 0
        self._episode_initialized = True

    def propose(
        self,
        context: TaskObjectDisturbanceContext,
    ) -> DisturbanceCommand | None:
        """Spend the next attack when its PnP phase reaches a safe window."""
        if not self._episode_initialized:
            raise RuntimeError("Disturbance policy must be reset before use.")
        if not self._pending_scenarios:
            return None

        task_state = context.task_state
        state_entry = (task_state.state_name, task_state.state_entry_id)
        if state_entry != self._active_state_entry:
            self._active_state_entry = state_entry
            self._steps_in_state = 1
        else:
            self._steps_in_state += 1

        scenario = self._pending_scenarios[0]
        if task_state.state_name != scenario.state_name:
            return None
        if state_entry == self._last_attacked_state_entry:
            return None
        if self._steps_in_state < scenario.trigger_after_steps:
            return None

        try:
            target = context.objects[self._target_name]
            target_region = context.objects[self._target_region_name]
        except KeyError as error:
            raise KeyError(
                f"Missing task-object observation {error.args[0]!r}"
            ) from error

        position_offset = self._choose_valid_offset(
            target.position,
            target_region.position,
            scenario.displacement,
        )
        attack_number = self._attacks_executed + 1
        trigger_state_step = self._steps_in_state
        self._pending_scenarios.pop(0)
        self._attacks_executed = attack_number
        self._last_attacked_state_entry = state_entry
        return ObjectPoseOffset(
            channel=DisturbanceChannel.TASK_OBJECT,
            reason=(
                f"PnP naughty policy attack {attack_number}/"
                f"{self._attack_budget} targeted {scenario.state_name} "
                f"entry {task_state.state_entry_id} "
                f"at state step {trigger_state_step}"
            ),
            target_name=self._target_name,
            position_offset=position_offset,
        )

    def _choose_valid_offset(
        self,
        position: tuple[float, float, float],
        target_position: tuple[float, float, float],
        displacement: float,
    ) -> tuple[float, float, float]:
        candidate_distances = (displacement, displacement * 1.5)
        candidates = tuple(
            offset
            for distance in candidate_distances
            for offset in (
                (distance, 0.0, 0.0),
                (-distance, 0.0, 0.0),
                (0.0, distance, 0.0),
                (0.0, -distance, 0.0),
            )
        )
        valid_candidates = tuple(
            offset
            for offset in candidates
            if self._inside_workspace(
                position[0] + offset[0],
                position[1] + offset[1],
            )
            and not self._inside_target_region(
                position[0] + offset[0],
                position[1] + offset[1],
                target_position,
            )
        )
        if not valid_candidates:
            raise RuntimeError(
                "No cube disturbance candidate avoids both workspace bounds "
                "and the target region."
            )
        center_x = sum(self._workspace_x) / 2.0
        center_y = sum(self._workspace_y) / 2.0
        width_x = self._workspace_x[1] - self._workspace_x[0]
        width_y = self._workspace_y[1] - self._workspace_y[0]
        return min(
            valid_candidates,
            key=lambda offset: (
                offset[0] ** 2 + offset[1] ** 2,
                (
                    ((position[0] + offset[0] - center_x) / width_x) ** 2
                    + ((position[1] + offset[1] - center_y) / width_y) ** 2
                ),
            ),
        )

    def _inside_workspace(self, x: float, y: float) -> bool:
        return (
            self._workspace_x[0] <= x <= self._workspace_x[1]
            and self._workspace_y[0] <= y <= self._workspace_y[1]
        )

    def _inside_target_region(
        self,
        x: float,
        y: float,
        target_position: tuple[float, float, float],
    ) -> bool:
        return (
            abs(x - target_position[0]) <= self._target_half_clearance
            and abs(y - target_position[1]) <= self._target_half_clearance
        )

    def _validate_configuration(self) -> None:
        if not self._scenarios:
            raise ValueError("At least one PnP disturbance scenario is required.")
        if self._workspace_x[0] >= self._workspace_x[1]:
            raise ValueError("workspace_x must contain increasing bounds")
        if self._workspace_y[0] >= self._workspace_y[1]:
            raise ValueError("workspace_y must contain increasing bounds")
        if self._target_half_clearance < 0.0:
            raise ValueError("target_half_clearance must be nonnegative")
        if len(self._attack_count_range) != 2:
            raise ValueError("attack_count_range must contain two values")
        minimum_attacks, maximum_attacks = self._attack_count_range
        if minimum_attacks < 0 or minimum_attacks > maximum_attacks:
            raise ValueError(
                "attack_count_range must contain nonnegative increasing bounds"
            )

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
