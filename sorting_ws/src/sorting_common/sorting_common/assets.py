"""Locate shared assets (fonts) installed with sorting_common."""

from __future__ import annotations

from pathlib import Path

UI_FONT = "BarlowSemiCondensed-Medium.ttf"


def font_path(name: str = UI_FONT) -> str | None:
    """Absolute path of a bundled font, or None if it can't be found."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("sorting_common")) / "assets" / "fonts" / name)
    except ImportError, LookupError:
        pass
    candidates.append(Path(__file__).resolve().parent.parent / "assets" / "fonts" / name)
    for c in candidates:
        if c.is_file():
            return str(c)
    return None
