"""Small, simulator-independent contracts for adversarial disturbances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class DisturbanceChannel(Enum):
    """The two independent sources of naughty-ghost disturbances."""

    BACKGROUND = "background"
    TASK_OBJECT = "task_object"


@dataclass(frozen=True)
class TaskStateView:
    """Read-only task state exposed to state-aware disturbance policies."""

    task_name: str
    state_name: str
    state_entry_id: int = 0


@dataclass(frozen=True)
class ObjectPoseView:
    """Simulator-independent pose snapshot for one observed scene object."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskObjectDisturbanceContext:
    """Read-only input available to task-object disturbance policies."""

    task_state: TaskStateView
    objects: Mapping[str, ObjectPoseView]


@dataclass(frozen=True)
class DisturbanceCommand:
    """A world mutation requested by a disturbance policy."""

    channel: DisturbanceChannel
    reason: str


@dataclass(frozen=True)
class ObjectPoseOffset(DisturbanceCommand):
    """Move one runtime physics object by a world-frame position offset."""

    target_name: str
    position_offset: tuple[float, float, float]


class DisturbanceExecutor(Protocol):
    """Apply disturbance commands to a concrete simulation backend."""

    def execute(self, command: DisturbanceCommand) -> None:
        """Apply one requested world mutation."""
