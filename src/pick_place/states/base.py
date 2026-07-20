"""State-machine contracts for the pick-and-place controller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from isaacsim.core.utils.types import ArticulationAction


class PickPlacePhase(Enum):
    """Execution phases for one pick-and-place episode."""

    IDLE = 0
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    PLACE = 5
    RELEASE = 6
    RETURN = 7


class CubeCollisionMode(Enum):
    """CuRobo collision treatment for the manipulated cube in one state."""

    WORLD_OBSTACLE = "world_obstacle"
    IGNORED = "ignored"
    ATTACHED = "attached"


@dataclass(frozen=True)
class Perturbation:
    """An observed violation of an active state's assumptions."""

    reason: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateStep:
    """Action and optional transition produced by one state-machine tick."""

    action: ArticulationAction | None = None
    next_phase: PickPlacePhase | None = None


class PnPState(ABC):
    """One local state in the pick-and-place state machine."""

    max_final_waypoint_hold_steps = 30
    phase: PickPlacePhase
    cube_collision_mode: CubeCollisionMode

    def enter(self) -> None:
        """Initialize state-local execution data."""

    def exit(self) -> None:
        """Release state-local execution data before a transition."""

    @abstractmethod
    def is_success(self) -> bool:
        """Return whether this state's execution goal is currently satisfied."""
        raise NotImplementedError

    @abstractmethod
    def detect_perturbation(self) -> Perturbation | None:
        """Return an invalidated state assumption, if one is observed."""
        return None

    @abstractmethod
    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Choose where this state recovers after a perturbation."""
        raise RuntimeError(
            f"{type(self).__name__} does not handle perturbation "
            f"{perturbation.reason!r}."
        )

    @abstractmethod
    def update(self) -> StateStep:
        """Advance normal execution by one simulation tick."""
