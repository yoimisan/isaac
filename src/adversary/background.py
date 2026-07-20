"""State-independent background-disturbance policy interface."""

from __future__ import annotations

from typing import Protocol

from adversary.types import DisturbanceCommand


class BackgroundDisturbancePolicy(Protocol):
    """Select background changes without observing the task-agent state."""

    def reset(self) -> None:
        """Begin a new episode."""

    def propose(self, episode_step: int) -> DisturbanceCommand | None:
        """Optionally propose one background disturbance for this step."""


class NoBackgroundDisturbance:
    """Placeholder used until the first lighting disturbance is added."""

    def reset(self) -> None:
        pass

    def propose(self, episode_step: int) -> DisturbanceCommand | None:
        del episode_step
        return None
