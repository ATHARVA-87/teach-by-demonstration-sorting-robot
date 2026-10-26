"""Shared scene model of the sorting cell (items, boxes, workspace, AR mapping)."""

from sorting_scene.generator import SceneGenerationError, generate_scene, place_item
from sorting_scene.model import (
    ATTRIBUTE_NAMES,
    ArView,
    Box,
    Item,
    Rect,
    Scene,
    SceneConfig,
    default_config_path,
    load_config,
)

__all__ = [
    "ATTRIBUTE_NAMES",
    "ArView",
    "Box",
    "Item",
    "Rect",
    "Scene",
    "SceneConfig",
    "SceneGenerationError",
    "default_config_path",
    "generate_scene",
    "load_config",
    "place_item",
]
