"""Unit tests for the shared scene model."""

import math
from pathlib import Path

import pytest
import yaml

from sorting_scene import (
    ArView,
    Item,
    Scene,
    SceneConfig,
    SceneGenerationError,
    generate_scene,
    load_config,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "scene.yaml"


@pytest.fixture(scope="module")
def cfg() -> SceneConfig:
    return load_config(CONFIG)


def test_config_loads_attributes_and_boxes(cfg: SceneConfig) -> None:
    assert set(cfg.attribute_values) == {"color", "size", "shape"}
    assert cfg.box_labels == ("A", "B", "C")
    assert [b.slot for b in cfg.boxes] == [0, 1, 2]
    # Design plan D6: large items are at least 1.5x the small ones.
    assert cfg.sizes["large"] >= 1.5 * cfg.sizes["small"]


def test_same_seed_same_scene(cfg: SceneConfig) -> None:
    assert generate_scene(cfg, 42).to_json() == generate_scene(cfg, 42).to_json()
    assert generate_scene(cfg, 42).to_json() != generate_scene(cfg, 43).to_json()


@pytest.mark.parametrize("seed", range(50))
def test_items_inside_area_and_not_overlapping(cfg: SceneConfig, seed: int) -> None:
    scene = generate_scene(cfg, seed)
    assert cfg.n_items_min <= len(scene.items) <= cfg.n_items_max
    for it in scene.items:
        r = cfg.footprint_radius(it.size)
        assert cfg.items_area.contains(it.x, it.y, margin=r - 1e-9)
        assert cfg.box_at(it.x, it.y) is None
    for i, a in enumerate(scene.items):
        for b in scene.items[i + 1 :]:
            dist = math.hypot(a.x - b.x, a.y - b.y)
            gap = dist - cfg.footprint_radius(a.size) - cfg.footprint_radius(b.size)
            assert gap >= cfg.min_clearance - 1e-9


def test_fixed_attributes_are_respected(cfg: SceneConfig) -> None:
    q = {"color": "blue", "size": "large", "shape": "prism"}
    scene = generate_scene(cfg, 3, n_items=4, fixed_attributes=[q])
    assert scene.items[0].attributes() == q


def test_unknown_attribute_value_rejected(cfg: SceneConfig) -> None:
    with pytest.raises(ValueError, match="purple"):
        bad = {"color": "purple", "size": "small", "shape": "cube"}
        generate_scene(cfg, 0, n_items=1, fixed_attributes=[bad])


def test_impossible_layout_raises_clear_error(cfg: SceneConfig) -> None:
    with pytest.raises(SceneGenerationError, match="Reduce n_items"):
        generate_scene(cfg, 0, n_items=200)


def test_scene_json_round_trip(cfg: SceneConfig) -> None:
    scene = generate_scene(cfg, 11)
    assert Scene.from_json(scene.to_json()) == scene


def test_ar_mapping_round_trip_is_exact(cfg: SceneConfig) -> None:
    ar = cfg.ar_view
    for x, y in [(0.3, -0.3), (0.48, 0.06), (0.64, 0.34), (0.7, 0.45)]:
        u, v = ar.world_to_px(x, y)
        x2, y2 = ar.px_to_world(u, v)
        assert x2 == pytest.approx(x, abs=1e-12) and y2 == pytest.approx(y, abs=1e-12)


def test_ar_mapping_orientation(cfg: SceneConfig) -> None:
    ar = cfg.ar_view
    assert ar.world_to_px(*ar.world_center) == pytest.approx(ar.center_px)
    u0, v0 = ar.world_to_px(0.48, 0.06)
    u1, _ = ar.world_to_px(0.48, 0.16)  # +y (worker's right) -> +u (screen right)
    _, v1 = ar.world_to_px(0.58, 0.06)  # +x (toward worker)  -> +v (screen down)
    assert u1 > u0 and v1 > v0
    assert ar.meters_to_px(0.1) == pytest.approx(76.0)


def test_whole_table_fits_in_image(cfg: SceneConfig) -> None:
    ar, e = cfg.ar_view, cfg.extent
    w, h = ar.image_size
    for x in (e.x_min, e.x_max):
        for y in (e.y_min, e.y_max):
            u, v = ar.world_to_px(x, y)
            assert 0 <= u <= w and 0 <= v <= h


def test_box_lookup(cfg: SceneConfig) -> None:
    b = cfg.box("B")
    assert cfg.box_at(b.x, b.y) == b
    assert cfg.box_at(b.x + b.size, b.y) != b


def test_parallel_axes_rejected() -> None:
    with pytest.raises(ValueError, match="parallel"):
        ArView((640, 480), (0, 0), (320, 240), 100.0, (1.0, 0.0), (2.0, 0.0))


def test_missing_field_names_the_field(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["generation"]["min_clearance"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="min_clearance"):
        load_config(p)


def test_box_overlapping_items_area_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["boxes"][0]["y"] = 0.0  # inside the items area
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlaps"):
        load_config(p)


def test_ar_axes_must_be_orthonormal() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        ArView((640, 480), (0, 0), (320, 240), 100.0, (2.0, 0.0), (0.0, 1.0))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["boxes"][1].update(x=0.33), "overlap"),
        (lambda r: r["attributes"]["size"].update(small=-0.01), "positive"),
        (lambda r: r["table"]["items_area"].update(x_max=0.9), "inside the table"),
        (lambda r: r["boxes"][0].pop("x"), "missing or malformed"),
        (lambda r: r["ar_view"].pop("pixels_per_meter"), "missing or malformed"),
    ],
)
def test_invalid_configs_raise_value_error(tmp_path: Path, mutate, message: str) -> None:  # noqa: ANN001
    raw = yaml.safe_load(CONFIG.read_text())
    mutate(raw)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load_config(p)


def test_place_item_finds_free_spot(cfg: SceneConfig) -> None:
    import numpy as np

    from sorting_scene import place_item

    scene = generate_scene(cfg, 4)
    attrs = {"color": "blue", "size": "large", "shape": "prism"}
    it = place_item(cfg, attrs, np.random.default_rng(0), scene.items, "query_01")
    assert it is not None and it.attributes() == attrs and it.id == "query_01"
    for o in scene.items:
        gap = (
            math.hypot(it.x - o.x, it.y - o.y) - cfg.footprint_radius(it.size) - cfg.footprint_radius(o.size)
        )
        assert gap >= cfg.min_clearance - 1e-9


def test_place_item_rejects_unknown_values_and_full_table(cfg: SceneConfig) -> None:
    import numpy as np

    from sorting_scene import place_item

    with pytest.raises(ValueError, match="purple"):
        place_item(
            cfg, {"color": "purple", "size": "small", "shape": "cube"}, np.random.default_rng(0), [], "q"
        )
    crowded = generate_scene(cfg, 0, n_items=1)
    crowded.items = [
        Item(f"i{i}", "red", "large", "cube", x, y)
        for i, (x, y) in enumerate(
            (x, y) for x in np.arange(0.33, 0.62, 0.05) for y in np.arange(-0.27, 0.16, 0.05)
        )
    ]
    assert (
        place_item(
            cfg,
            {"color": "red", "size": "large", "shape": "cube"},
            np.random.default_rng(0),
            crowded.items,
            "q",
        )
        is None
    )
