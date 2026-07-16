"""Idle state for the pick-and-place controller."""

from pick_place.states.base import (
    Perturbation,
    PickPlacePhase,
    PnPState,
    StateStep,
)


class IdleState(PnPState):
    """Hold execution until the controller starts or resets an episode."""

    phase = PickPlacePhase.IDLE

    def detect_perturbation(self) -> Perturbation | None:
        """Ignore world changes because idle has no active task assumption."""
        return None

    def recovery_phase(self, perturbation: Perturbation) -> PickPlacePhase:
        """Reject recovery requests because idle detects no perturbations."""
        return super().recovery_phase(perturbation)

    def update(self) -> StateStep:
        """Produce no action while the state machine is idle."""
        return StateStep()
