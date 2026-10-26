"""Unit tests for sorting_common.filters."""

import pytest

from sorting_common.filters import Debouncer, Ema

OPEN, CLOSED = "OPEN", "CLOSED"


def frames(state: str) -> int:
    """Grab (CLOSED) needs 3 frames, release (OPEN) needs 1."""
    return 3 if state == CLOSED else 1


def test_ema_starts_at_first_sample_and_converges() -> None:
    f = Ema(0.5)
    assert f.update(4.0) == 4.0
    assert f.update(0.0) == 2.0
    assert f.update(0.0) == 1.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_ema_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        Ema(alpha)


def test_ema_reset_none_restarts_from_next_sample() -> None:
    f = Ema(0.1)
    f.update(10.0)
    f.reset()
    assert f.update(2.0) == 2.0


def test_ema_reset_to_value() -> None:
    f = Ema(0.5)
    f.update(10.0)
    f.reset(0.0)
    assert f.update(4.0) == 2.0


def test_debouncer_needs_consecutive_frames() -> None:
    d = Debouncer(OPEN, frames)
    assert d.update(CLOSED) == OPEN
    assert d.update(CLOSED) == OPEN
    assert d.update(CLOSED) == CLOSED


def test_debouncer_restarts_count_on_interruption() -> None:
    d = Debouncer(OPEN, frames)
    d.update(CLOSED)
    d.update(CLOSED)
    d.update(OPEN)  # flicker back: count restarts
    d.update(CLOSED)
    d.update(CLOSED)
    assert d.state == OPEN


def test_debouncer_asymmetric_counts() -> None:
    d = Debouncer(CLOSED, frames)
    assert d.update(OPEN) == OPEN  # 1 frame suffices for OPEN


def test_debouncer_force_clears_pending_change() -> None:
    d = Debouncer(OPEN, frames)
    d.update(CLOSED)
    d.update(CLOSED)
    d.force(OPEN)
    assert d.update(CLOSED) == OPEN  # pending count was cleared
