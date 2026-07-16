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
    WAIT_FOR_STABLE = 1
    APPROACH = 2
    DESCEND = 3
    GRASP = 4
    LIFT = 5
    PLACE = 6
    RELEASE = 7
    RETURN = 8


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

    phase: PickPlacePhase

    def enter(self) -> None:
        """Initialize state-local execution data."""

    def exit(self) -> None:
        """Release state-local execution data before a transition."""

    def detect_perturbation(self) -> Perturbation | None:
        """Return an invalidated state assumption, if one is observed."""
        return None

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Choose where this state recovers after a perturbation."""
        raise RuntimeError(
            f"{type(self).__name__} does not handle perturbation "
            f"{perturbation.reason!r}."
        )

    @abstractmethod
    def update(self) -> StateStep:
        """Advance normal execution by one simulation tick."""

