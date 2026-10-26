"""Design tokens for the AR overlay, from docs/design_plan.md (decision D6).

OpenCV uses BGR order, so each hex token is converted here, once.
Item paint is NOT here: it comes from sorting_scene/config/scene.yaml.
"""

BGR = tuple[int, int, int]


def hex_to_bgr(hex_color: str) -> BGR:
    """'#RRGGBB' -> (B, G, R)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


BLACK_GREY: BGR = hex_to_bgr("#2E3236")  # RAL 7021, scrims and backgrounds
ANTHRACITE: BGR = hex_to_bgr("#383E42")  # RAL 7016, trays
SIGNAL_WHITE: BGR = hex_to_bgr("#ECECE7")  # RAL 9003, strokes, text, cursor
INK: BGR = hex_to_bgr("#1E2124")  # halos, item outlines, text on white tiles
DIVIDER: BGR = hex_to_bgr("#474D52")
