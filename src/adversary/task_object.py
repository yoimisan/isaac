"""State-aware policies for disturbing task goal objects."""

from __future__ import annotations

from typing import Protocol

from adversary.types import (
    DisturbanceChannel,
    DisturbanceCommand,
    ObjectPoseOffset,
    TaskStateView,
)


class TaskObjectDisturbancePolicy(Protocol):
    """Select a task-object change from a read-only task state."""

    def reset(self) -> None:
        """Begin a new episode."""

    def propose(self, task_state: TaskStateView) -> DisturbanceCommand | None:
        """Optionally propose one task-object disturbance for this step."""


class OneShotStateObjectPerturbation:
    """Offset one object once after a configured state has run long enough."""

    def __init__(
        self,
        *,
        trigger_state: str,
        trigger_after_steps: int,
        target_name: str,
        position_offset: tuple[float, float, float],
    ) -> None:
        if trigger_after_steps < 1:
            raise ValueError("trigger_after_steps must be at least one")
        if len(position_offset) != 3:
            raise ValueError("position_offset must contain exactly three values")

        self._trigger_state = trigger_state
        self._trigger_after_steps = trigger_after_steps
        self._target_name = target_name
        self._position_offset = tuple(float(value) for value in position_offset)
        self.reset()

    def reset(self) -> None:
        self._active_state: str | None = None
        self._steps_in_state = 0
        self._has_fired = False

    def propose(self, task_state: TaskStateView) -> DisturbanceCommand | None:
        if task_state.state_name != self._active_state:
            self._active_state = task_state.state_name
            self._steps_in_state = 1
        else:
            self._steps_in_state += 1

        if self._has_fired or task_state.state_name != self._trigger_state:
            return None
        if self._steps_in_state < self._trigger_after_steps:
            return None

        self._has_fired = True
        return ObjectPoseOffset(
            channel=DisturbanceChannel.TASK_OBJECT,
            reason=(
                f"{task_state.task_name} state {task_state.state_name} "
                f"reached step {self._steps_in_state}"
            ),
            target_name=self._target_name,
            position_offset=self._position_offset,
        )
