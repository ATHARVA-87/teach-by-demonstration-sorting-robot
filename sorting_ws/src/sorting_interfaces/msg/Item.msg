# One item in the sorting cell, as defined by sorting_scene.
# Positions are in the world frame (robot base), on the table surface.

string id          # unique within a scene, e.g. "item_03"
string color       # one of the colors in sorting_scene/config/scene.yaml
string size        # e.g. "small" | "medium" | "large"
string shape       # e.g. "cube" | "cylinder" | "prism"
float64 x          # m, world frame
float64 y          # m, world frame
float64 yaw        # rad, rotation about the table normal
