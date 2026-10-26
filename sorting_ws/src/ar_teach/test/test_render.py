"""Rendering tests: the overlay draws every element without errors and obeys the color rule."""

from pathlib import Path

import numpy as np
import pytest

from ar_teach.pinch import PinchState
from ar_teach.render import PATTERNS, OverlayRenderer, pattern_for_slot
from ar_teach.session import TeachSession
from sorting_scene import Item, generate_scene, load_config

SCENE_YAML = Path(__file__).resolve().parents[2] / "sorting_scene" / "config" / "scene.yaml"
FONT = (
    Path(__file__).resolve().parents[2]
    / "sorting_common"
    / "assets"
    / "fonts"
    / "BarlowSemiCondensed-Medium.ttf"
)


@pytest.fixture()
def setup() -> tuple[OverlayRenderer, TeachSession]:
    cfg = load_config(SCENE_YAML)
    session = TeachSession(cfg)
    for it in generate_scene(cfg, 5).items:
        session.add_item(it)
    session.add_item(Item("q", "blue", "large", "prism", 0.40, 0.05), is_query=True)
    return OverlayRenderer(cfg, str(FONT)), session


def test_draws_all_states_without_error(setup: tuple[OverlayRenderer, TeachSession]) -> None:
    renderer, session = setup
    frame = np.full((480, 640, 3), 120, np.uint8)
    hover_id = session.table_items()[0].item.id
    session.hover = hover_id
    for pinched in (False, True):
        out = frame.copy()
        pinch = PinchState(True, pinched, 0.2, (320.0, 240.0), 0.7)
        renderer.draw(out, session, pinch, banner="Hand lost. Item returned to its place.")
        assert out.shape == frame.shape and not np.array_equal(out, frame)


def test_untracked_hand_draws_no_cursor(setup: tuple[OverlayRenderer, TeachSession]) -> None:
    renderer, session = setup
    base = np.full((480, 640, 3), 120, np.uint8)
    a, b = base.copy(), base.copy()
    # Untracked: the cursor position must be ignored entirely.
    renderer.draw(a, session, PinchState(False, False, None, (100.0, 100.0), 0.0))
    renderer.draw(b, session, PinchState(False, False, None, (400.0, 300.0), 0.0))
    assert np.array_equal(a, b)
    # Tracked at the same spot: pixels near the cursor change.
    c = base.copy()
    renderer.draw(c, session, PinchState(True, False, 0.8, (100.0, 100.0), 0.0))
    assert not np.array_equal(a[85:116, 85:116], c[85:116, 85:116])


def test_saturated_paint_only_inside_item_footprints(setup: tuple[OverlayRenderer, TeachSession]) -> None:
    """Design rule D6: item color is data, UI color is state.

    Outside every item's footprint (plus margin), no pixel may be strongly
    saturated: chrome is greys and Signal white only, so neither item paint
    nor a status hue can appear anywhere else in the AR view.
    """
    renderer, session = setup
    frame = np.full((480, 640, 3), 128, np.uint8)
    renderer.draw(frame, session, PinchState(True, False, 0.8, (300.0, 200.0), 0.3), banner="Test")
    outside = np.ones(frame.shape[:2], bool)
    cfg = session.scene_cfg
    yy, xx = np.mgrid[0:480, 0:640]
    for vi in session.items.values():
        if vi.status == "boxed":
            continue
        u, v = cfg.ar_view.world_to_px(vi.item.x, vi.item.y)
        r = cfg.ar_view.meters_to_px(cfg.sizes[vi.item.size]) * 1.0 + 6
        outside &= (xx - u) ** 2 + (yy - v) ** 2 > r**2
    hsv_sat = frame.max(axis=2).astype(int) - frame.min(axis=2).astype(int)
    assert (hsv_sat[outside] > 60).sum() == 0


def test_box_patterns_repeat_by_slot() -> None:
    assert [pattern_for_slot(i) for i in range(len(PATTERNS))] == list(PATTERNS)
    assert pattern_for_slot(len(PATTERNS)) == PATTERNS[0]
