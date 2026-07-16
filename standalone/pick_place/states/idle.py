"""Idle state for the pick-and-place controller."""

from pick_place.states.base import PickPlacePhase, PnPState, StateStep


class IdleState(PnPState):
    """Hold execution until the controller starts or resets an episode."""

    phase = PickPlacePhase.IDLE

    def update(self) -> StateStep:
        """Produce no action while the state machine is idle."""
        return StateStep()
