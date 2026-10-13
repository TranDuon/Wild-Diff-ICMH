"""Opt-in time allowance for a new fit, without resetting learning state."""
import math

from lightning.pytorch.callbacks import Timer


class SessionTimer(Timer):
    """Restore the Timer slot but start each new fit with zero elapsed time."""

    @property
    def state_key(self):
        # Match ordinary Lightning Timer checkpoints, including existing runs.
        return Timer().state_key

    def load_state_dict(self, state_dict):
        # The public Timer loader resets its elapsed offset for an empty state.
        # Never mutate the checkpoint: all other callback/learning states restore.
        super().load_state_dict({})


def configure_session_timer(callbacks, trainer_options, enabled=False):
    """Return fresh callback/options lists; legacy calls remain unchanged."""
    callbacks = list(callbacks)
    options = dict(trainer_options)
    if not enabled:
        return callbacks, options
    if any(isinstance(callback, Timer) for callback in callbacks):
        raise ValueError("--session-time-limit cannot be combined with another Timer callback")
    duration = options.pop("max_time", None)
    if duration is None:
        raise ValueError("--session-time-limit requires a positive finite max_time")
    timer = SessionTimer(duration=duration, interval="step")
    remaining = timer.time_remaining()
    if remaining is None or not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("--session-time-limit requires a positive finite max_time")
    callbacks.append(timer)
    print(f"Session training allowance: {remaining:g}s (fresh per fit; setup excluded)", flush=True)
    # Explicit callback owns the limit. Do not ask Trainer to add another Timer.
    return callbacks, options
