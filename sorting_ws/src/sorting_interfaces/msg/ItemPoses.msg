# GROUND TRUTH item poses from the simulator. Published only on
# /sim/debug/item_poses, for evaluation, debugging and the M3 scripted
# baseline. The perception pipeline must never subscribe to it.

std_msgs/Header header        # sim time
Item[] items                  # attributes (+ x, y, yaw on the table)
geometry_msgs/Pose[] poses    # full 3D pose of each item, same order
