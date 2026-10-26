"""Small signal filters shared by several nodes.

- ``Ema``: exponential moving average (smoothing).
- ``Debouncer``: switch state only after it has been seen N frames in a row.

Carried over from the gesture-control project. Used in ar_teach for pinch
smoothing and grab confirmation.
"""

from collections.abc import Callable


class Ema:
    """Exponential moving average: ``s = alpha * x + (1 - alpha) * s``.

    alpha near 1 follows the input quickly (little smoothing); alpha near 0
    is very smooth but slow to react. The time constant is roughly 1/alpha
    samples.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.value: float | None = None  # None until the first sample

    def update(self, sample: float) -> float:
        """Add one sample and return the smoothed value."""
        if self.value is None:
            self.value = sample  # start at the first sample, not at 0
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, value: float | None = None) -> None:
        """Restart the filter.

        Args:
            value: new state. ``None`` means "restart from the next sample".
        """
        self.value = value


class Debouncer:
    """Only switch to a new state after it has been seen N frames in a row.

    ``frames_for(state)`` lets different target states need different
    counts, e.g. quick to release but slower to grab.
    """

    def __init__(self, initial: str, frames_for: Callable[[str], int]) -> None:
        self.state = initial
        self.frames_for = frames_for
        self._candidate = initial
        self._count = 0

    def update(self, raw: str) -> str:
        """Feed one raw observation; return the debounced state."""
        if raw == self.state:
            self._candidate, self._count = raw, 0  # nothing pending
        elif raw == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw, 1  # a new candidate appears
        if self._candidate != self.state and self._count >= self.frames_for(self._candidate):
            self.state = self._candidate
            self._count = 0
        return self.state

    def force(self, state: str) -> None:
        """Set the state immediately (e.g. on tracking loss), clearing any pending change."""
        self.state = self._candidate = state
        self._count = 0
