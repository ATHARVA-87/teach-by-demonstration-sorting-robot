"""Tests for pinch hysteresis and the teaching session (no webcam needed)."""

from pathlib import Path

import numpy as np
import pytest

from ar_teach.fixtures import Frame, drag_sequence, make_hand
from ar_teach.lighting import MSG_TOO_BRIGHT, MSG_TOO_DARK, lighting_message
from ar_teach.pinch import PinchConfig, PinchDetector, pinch_point, pinch_ratio
from ar_teach.session import (
    MSG_BETWEEN_BOXES,
    MSG_HAND_LOST,
    DemoEvent,
    MessageEvent,
    TeachSession,
)
from sorting_scene import Item, SceneConfig, load_config

SCENE_YAML = Path(__file__).resolve().parents[2] / "sorting_scene" / "config" / "scene.yaml"


@pytest.fixture(scope="module")
def cfg() -> SceneConfig:
    return load_config(SCENE_YAML)


# --- pinch geometry --------------------------------------------------------
def test_make_hand_has_requested_ratio_and_point() -> None:
    lm = make_hand((200.0, 150.0), hand_size_px=80.0, ratio=0.25)
    assert pinch_ratio(lm) == pytest.approx(0.25)
    assert pinch_point(lm) == pytest.approx((200.0, 150.0))


def test_ratio_is_scale_invariant() -> None:
    near = make_hand((300.0, 200.0), hand_size_px=160.0, ratio=0.3)
    far = make_hand((300.0, 200.0), hand_size_px=60.0, ratio=0.3)
    assert pinch_ratio(near) == pytest.approx(pinch_ratio(far))


def test_config_requires_hysteresis_band() -> None:
    with pytest.raises(ValueError):
        PinchConfig(grab_ratio=0.5, release_ratio=0.4)


# --- hysteresis and debouncing -----------------------------------------------
def _feed(det: PinchDetector, ratios: list[float]) -> list[bool]:
    return [det.update(make_hand((100.0, 100.0), 90.0, r)).pinched for r in ratios]


def test_grab_needs_ratio_below_grab_threshold_for_n_frames() -> None:
    det = PinchDetector(PinchConfig(ratio_alpha=1.0, grab_frames=2, release_frames=2))
    assert _feed(det, [0.8, 0.2, 0.2, 0.2]) == [False, False, True, True]


def test_hysteresis_band_holds_state() -> None:
    det = PinchDetector(
        PinchConfig(grab_ratio=0.3, release_ratio=0.45, ratio_alpha=1.0, grab_frames=1, release_frames=1)
    )
    # Close, then hover between the thresholds: stays closed.
    assert _feed(det, [0.2, 0.35, 0.40, 0.44]) == [True, True, True, True]
    # Only above release_ratio does it open.
    assert _feed(det, [0.5]) == [False]
    # And from open, values inside the band don't close it.
    assert _feed(det, [0.40, 0.35, 0.31]) == [False, False, False]


def test_noise_around_one_threshold_does_not_flicker() -> None:
    det = PinchDetector(PinchConfig(ratio_alpha=1.0, grab_frames=1, release_frames=1))
    ratios = [0.2] + [0.30 + 0.02 * ((-1) ** i) for i in range(20)]  # chatter around grab_ratio
    states = _feed(det, ratios)
    assert all(states)


def test_single_frame_spike_is_debounced() -> None:
    det = PinchDetector(PinchConfig(ratio_alpha=1.0, grab_frames=2, release_frames=2))
    assert _feed(det, [0.8, 0.1, 0.8, 0.8]) == [False, False, False, False]


def test_missing_hand_keeps_state_and_reports_untracked() -> None:
    det = PinchDetector(PinchConfig(ratio_alpha=1.0, grab_frames=1, release_frames=1))
    _feed(det, [0.1])
    s = det.update(None)
    assert not s.tracked and s.pinched
    det.reset()
    assert not det.update(None).pinched


def test_strength_increases_when_pinching() -> None:
    det = PinchDetector(PinchConfig(ratio_alpha=1.0))
    open_s = det.update(make_hand((0.0, 0.0), 90.0, 0.8)).strength
    closed_s = det.update(make_hand((0.0, 0.0), 90.0, 0.1)).strength
    assert open_s == 0.0 and closed_s == 1.0


# --- teaching session ---------------------------------------------------------
def _run(session: TeachSession, frames: list[Frame], det: PinchDetector | None = None) -> list:  # noqa: ANN401
    det = det or PinchDetector()
    events = []
    for f in frames:
        events += session.update(f.t, det.update(f.landmarks))
    return events


def _item(cfg: SceneConfig, x: float = 0.45, y: float = -0.1) -> Item:
    return Item(id="item_00", color="red", size="large", shape="cube", x=x, y=y)


def test_drag_into_box_emits_demonstration(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    start = cfg.ar_view.world_to_px(it.x, it.y)
    end = cfg.ar_view.world_to_px(cfg.box("B").x, cfg.box("B").y)
    events = _run(s, list(drag_sequence(start, end, noise_px=1.0, seed=1)))
    demos = [e for e in events if isinstance(e, DemoEvent)]
    assert len(demos) == 1
    d = demos[0]
    assert d.box == "B" and d.item.color == "red" and d.item.size == "large"
    assert (d.item.x, d.item.y) == pytest.approx((it.x, it.y))  # start position recorded
    assert d.duration_s > 0
    assert s.items["item_00"].status == "boxed" and s.held is None


def test_drop_between_boxes_returns_item(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    a, b = cfg.box("A"), cfg.box("B")
    gap_px = cfg.ar_view.world_to_px((a.x + b.x) / 2, a.y)  # between A and B
    events = _run(s, list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), gap_px)))
    assert any(isinstance(e, MessageEvent) and e.text == MSG_BETWEEN_BOXES for e in events)
    assert not any(isinstance(e, DemoEvent) for e in events)
    vi = s.items["item_00"]
    assert vi.status == "table" and (vi.item.x, vi.item.y) == pytest.approx((it.x, it.y))


def test_drop_on_table_moves_item_without_demo(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    target = cfg.ar_view.world_to_px(0.40, 0.0)
    events = _run(s, list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), target)))
    assert events == []
    assert s.items["item_00"].item.x == pytest.approx(0.40, abs=0.01)


def test_long_tracking_loss_mid_drag_returns_item(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    end = cfg.ar_view.world_to_px(cfg.box("A").x, cfg.box("A").y)
    # Hand disappears for 20 frames (0.67 s > 0.4 s timeout) during the drag.
    frames = list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), end, drop_out=(18, 37)))
    events = _run(s, frames)
    assert any(isinstance(e, MessageEvent) and e.text == MSG_HAND_LOST for e in events)
    assert not any(isinstance(e, DemoEvent) for e in events)
    vi = s.items["item_00"]
    assert (vi.item.x, vi.item.y) == pytest.approx((it.x, it.y))


def test_short_dropout_is_bridged(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    end = cfg.ar_view.world_to_px(cfg.box("C").x, cfg.box("C").y)
    frames = list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), end, drop_out=(20, 23)))  # 0.13 s
    demos = [e for e in _run(s, frames) if isinstance(e, DemoEvent)]
    assert len(demos) == 1 and demos[0].box == "C"


def test_overlapping_items_pick_the_nearest(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    s.add_item(Item(id="a", color="red", size="large", shape="cube", x=0.45, y=-0.10))
    s.add_item(Item(id="b", color="blue", size="large", shape="cube", x=0.45, y=-0.08))
    ub = cfg.ar_view.world_to_px(0.45, -0.075)  # closer to b
    assert s.item_under(*ub) == "b"
    ua = cfg.ar_view.world_to_px(0.45, -0.105)
    assert s.item_under(*ua) == "a"


def test_pinch_in_empty_space_grabs_nothing(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    s.add_item(_item(cfg))
    empty = cfg.ar_view.world_to_px(0.35, 0.10)
    end = cfg.ar_view.world_to_px(cfg.box("A").x, cfg.box("A").y)
    assert _run(s, list(drag_sequence(empty, end))) == []
    assert s.items["item_00"].status == "table"


def test_query_flag_is_carried_to_demo(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it, is_query=True)
    end = cfg.ar_view.world_to_px(cfg.box("A").x, cfg.box("A").y)
    demos = [
        e
        for e in _run(s, list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), end)))
        if isinstance(e, DemoEvent)
    ]
    assert demos and demos[0].is_query


def test_drop_target_highlight(cfg: SceneConfig) -> None:
    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    det = PinchDetector()
    start = cfg.ar_view.world_to_px(it.x, it.y)
    end = cfg.ar_view.world_to_px(cfg.box("B").x, cfg.box("B").y)
    seen = set()
    for f in drag_sequence(start, end):
        s.update(f.t, det.update(f.landmarks))
        seen.add(s.drop_target())
    assert "B" in seen


# --- lighting --------------------------------------------------------------------
def test_lighting_messages() -> None:
    assert lighting_message(np.full((48, 64, 3), 20, np.uint8)) == MSG_TOO_DARK
    assert lighting_message(np.full((48, 64, 3), 255, np.uint8)) == MSG_TOO_BRIGHT
    assert lighting_message(np.full((48, 64, 3), 120, np.uint8)) is None


def test_hand_returning_already_pinched_does_not_grab(cfg: SceneConfig) -> None:
    from ar_teach.pinch import PinchState

    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    s.disarm()  # as the node does after a tracking-loss reset
    over = cfg.ar_view.world_to_px(it.x, it.y)
    s.update(0.0, PinchState(True, True, 0.1, over, 1.0))  # comes back pinched over the item
    assert s.held is None
    s.update(0.1, PinchState(True, False, 0.8, over, 0.0))  # opens: re-armed
    s.update(0.2, PinchState(True, True, 0.1, over, 1.0))  # fresh pinch
    assert s.held == "item_00"


def test_drop_off_the_table_has_its_own_message(cfg: SceneConfig) -> None:
    from ar_teach.session import MSG_OFF_TABLE

    s = TeachSession(cfg)
    it = _item(cfg)
    s.add_item(it)
    off = cfg.ar_view.world_to_px(
        cfg.extent.x_max - 0.01, cfg.extent.y_min + 0.01
    )  # table corner, far from boxes
    events = _run(s, list(drag_sequence(cfg.ar_view.world_to_px(it.x, it.y), off)))
    assert any(isinstance(e, MessageEvent) and e.text == MSG_OFF_TABLE for e in events)
