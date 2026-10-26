"""Synthetic teaching source: a scripted hand sorts items by a fixed rule.

Lets the whole AR pipeline (pinch -> session -> demonstration events -> rule
learner) run headless, with no webcam and no person. The background is a
blurred noise texture so lighting checks and rendering behave as on video.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from ar_teach.fixtures import drag_sequence
from ar_teach.session import TeachSession


@dataclass(frozen=True)
class ScriptedRule:
    """Rule the synthetic teacher follows: attribute value -> box, else default."""

    attribute: str = "color"
    mapping: dict[str, str] = field(default_factory=lambda: {"red": "A", "blue": "B"})
    default: str = "C"

    def box_for(self, attributes: dict[str, str]) -> str:
        """Target box for an item with these attributes."""
        return self.mapping.get(attributes[self.attribute], self.default)


class SyntheticSource:
    """Yields (background frame, landmarks or None) at a fixed frame rate."""

    def __init__(
        self,
        session: TeachSession,
        rule: ScriptedRule,
        size: tuple[int, int] = (640, 480),
        fps: float = 30.0,
        seed: int = 0,
    ) -> None:
        self.session = session
        self.rule = rule
        self.fps = fps
        rng = np.random.default_rng(seed)
        w, h = size
        noise = rng.integers(60, 140, (h, w, 3), dtype=np.uint8)
        self.background = cv2.GaussianBlur(noise, (0, 0), 18)
        self._script = self._frames()

    def _frames(self) -> Iterator[np.ndarray | None]:
        while True:
            pending = [vi for vi in self.session.table_items() if vi.status == "table"]
            if not pending:
                yield None  # nothing left to sort: hand out of view
                continue
            vi = pending[0]
            box = self.session.scene_cfg.box(self.rule.box_for(vi.item.attributes()))
            ar = self.session.scene_cfg.ar_view
            start = ar.world_to_px(vi.item.x, vi.item.y)
            end = ar.world_to_px(box.x, box.y)
            for f in drag_sequence(start, end, fps=self.fps, noise_px=0.8, seed=len(pending)):
                yield f.landmarks

    def next(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Next frame: a copy of the background and the scripted landmarks."""
        return self.background.copy(), next(self._script)
