"""Orchestrator for independent background and task-object disturbances."""

from __future__ import annotations

from adversary.background import (
    BackgroundDisturbancePolicy,
    NoBackgroundDisturbance,
)
from adversary.task_object import TaskObjectDisturbancePolicy
from adversary.types import DisturbanceExecutor, TaskObjectDisturbanceContext


class NaughtyGhost:
    """Collect disturbance proposals and apply them at one safe loop point."""

    def __init__(
        self,
        *,
        executor: DisturbanceExecutor,
        task_object_policy: TaskObjectDisturbancePolicy,
        background_policy: BackgroundDisturbancePolicy | None = None,
    ) -> None:
        self._executor = executor
        self._task_object_policy = task_object_policy
        self._background_policy = (
            background_policy
            if background_policy is not None
            else NoBackgroundDisturbance()
        )
        self._episode_step = 0

    def reset(self) -> None:
        """Reset both disturbance channels for a new episode."""
        self._episode_step = 0
        self._background_policy.reset()
        self._task_object_policy.reset()

    def step(self, context: TaskObjectDisturbanceContext) -> None:
        """Evaluate both channels and execute their proposed disturbances."""
        background_command = self._background_policy.propose(self._episode_step)
        task_object_command = self._task_object_policy.propose(context)

        for command in (background_command, task_object_command):
            if command is not None:
                self._executor.execute(command)

        self._episode_step += 1
