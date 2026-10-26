"""Seeded random scene generation (rejection sampling without overlaps)."""

from __future__ import annotations

import math

import numpy as np

from sorting_scene.model import Item, Scene, SceneConfig


class SceneGenerationError(RuntimeError):
    """Raised when no valid layout is found within the sampling budget."""


def generate_scene(
    cfg: SceneConfig,
    seed: int,
    n_items: int | None = None,
    fixed_attributes: list[dict[str, str]] | None = None,
) -> Scene:
    """Generate a reproducible random scene.

    The same ``seed`` and config always give the same scene: positions,
    yaw and attributes all come from one ``numpy.random.Generator``.

    Args:
        cfg: the scene configuration.
        seed: random seed (recorded in the returned scene).
        n_items: number of items; random in [n_items_min, n_items_max] if None.
        fixed_attributes: optional attributes for the first items (e.g. a
            query item from the rule learner); the rest are random.

    Raises:
        SceneGenerationError: if the items can't be placed without overlap.
    """
    rng = np.random.default_rng(seed)
    if n_items is None:
        n_items = int(rng.integers(cfg.n_items_min, cfg.n_items_max + 1))
    values = cfg.attribute_values
    fixed = fixed_attributes or []
    if len(fixed) > n_items:
        raise ValueError("more fixed items than n_items")

    items: list[Item] = []
    area = cfg.items_area
    for idx in range(n_items):
        attrs = (
            dict(fixed[idx])
            if idx < len(fixed)
            else {name: str(rng.choice(vals)) for name, vals in values.items()}
        )
        for name, val in attrs.items():
            if val not in values[name]:
                raise ValueError(f"unknown {name} value '{val}'")
        radius = cfg.footprint_radius(attrs["size"])
        for _ in range(cfg.max_tries):
            x = float(rng.uniform(area.x_min + radius, area.x_max - radius))
            y = float(rng.uniform(area.y_min + radius, area.y_max - radius))
            if all(
                math.hypot(x - o.x, y - o.y) >= radius + cfg.footprint_radius(o.size) + cfg.min_clearance
                for o in items
            ):
                break
        else:
            raise SceneGenerationError(
                f"Could not place item {idx + 1} of {n_items} without overlap "
                f"(seed {seed}). Reduce n_items or min_clearance in scene.yaml."
            )
        yaw = float(rng.uniform(-math.pi, math.pi))
        items.append(Item(id=f"item_{idx:02d}", x=x, y=y, yaw=yaw, **attrs))
    return Scene(seed=seed, items=items)


def place_item(
    cfg: SceneConfig,
    attributes: dict[str, str],
    rng: np.random.Generator,
    existing: list[Item],
    item_id: str,
) -> Item | None:
    """Place one item with these attributes in a free spot of the items area.

    Used for active-learning questions: the query item must not overlap the
    items already on the table. Returns None if there's no free spot within
    ``max_tries`` attempts (the caller tells the worker to clear space).

    Raises:
        ValueError: if an attribute value isn't in the scene vocabulary.
    """
    values = cfg.attribute_values
    for name in values:
        if attributes.get(name) not in values[name]:
            raise ValueError(f"unknown or missing {name} value '{attributes.get(name)}'")
    radius = cfg.footprint_radius(attributes["size"])
    area = cfg.items_area
    for _ in range(cfg.max_tries):
        x = float(rng.uniform(area.x_min + radius, area.x_max - radius))
        y = float(rng.uniform(area.y_min + radius, area.y_max - radius))
        if all(
            math.hypot(x - o.x, y - o.y) >= radius + cfg.footprint_radius(o.size) + cfg.min_clearance
            for o in existing
        ):
            return Item(
                id=item_id,
                x=x,
                y=y,
                yaw=float(rng.uniform(-math.pi, math.pi)),
                color=attributes["color"],
                size=attributes["size"],
                shape=attributes["shape"],
            )
    return None
