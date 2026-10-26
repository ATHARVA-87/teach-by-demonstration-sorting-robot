"""Reachability sweep (decision D8 #7): where can the Panda grasp top-down on this table?

For each point of a grid over the table, an item counts as reachable when MoveIt
finds a collision-free, limit-respecting IK solution with the hand pointing
straight down, for some wrist angle, both at grasp height (TCP at the item's
mid-height) and at the pre-grasp height 10 cm above. Wrist angles tried: every
22.5 deg (a cylinder can use any; cubes and prisms need a few specific ones).

Writes ``reachability.csv`` (x, y, grasp_ok, pregrasp_ok, n_wrist_ok) and
``reachability.png`` (the map with the items area, the trays and the base),
and prints the share of the items area that is reachable and how many wrist
angles work at each tray centre (checked at item-grasp height, stricter than
the real release height above the tray walls).

Usage:
    ros2 run sorting_motion reachability [--step 0.02] [--out results/reachability]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

from sorting_motion.grasps import TCP_OFFSET, hand_down_quaternion
from sorting_motion.moveit_setup import PARK_POSE, READY_POSE, moveit_config, within_limits
from sorting_motion.pick_place import PickPlacePlanner, load_planner_config
from sorting_scene import SceneConfig, default_config_path, load_config

WRISTS = [k * math.pi / 8 for k in range(16)]  # every 22.5 deg
SEEDS = (READY_POSE, PARK_POSE)


class Sweeper:
    """Top-down IK checks against MoveIt's robot model (offline, no robot needed)."""

    def __init__(self, moveit: MoveItPy, cfg: SceneConfig) -> None:
        self.model = moveit.get_robot_model()
        self.psm = moveit.get_planning_scene_monitor()
        self.pc = load_planner_config()
        # The table and tray walls as obstacles: a pose that puts the arm through
        # the table must not count as reachable.
        PickPlacePlanner(moveit, cfg, self.pc).set_world([])

    def _ik_ok(self, x: float, y: float, z_hand: float, yaw: float) -> bool:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z_hand
        q = hand_down_quaternion(yaw)
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = q
        base = math.atan2(y, x)
        for seed in SEEDS:
            st = RobotState(self.model)
            st.set_to_default_values()
            s = np.array(seed)
            if seed is SEEDS[0]:
                s[0] = base  # point joint1 at the target
            st.set_joint_group_positions("panda_arm", s)
            st.update()
            if not st.set_from_ik("panda_arm", pose, "panda_hand", 0.05):
                continue
            st.update()
            if not within_limits(st.get_joint_group_positions("panda_arm"), self.pc.joint_limit_margin):
                continue
            with self.psm.read_only() as scene:
                if not scene.is_state_colliding(robot_state=st, joint_model_group_name="panda_arm"):
                    return True
        return False

    def point(self, x: float, y: float, cfg: SceneConfig) -> tuple[bool, bool, int]:
        """(grasp reachable, pre-grasp reachable, wrist angles working for both) at (x, y)."""
        z_tcp = cfg.surface_z + cfg.item_height / 2
        grasp = pre = False
        both = 0
        for yaw in WRISTS:
            g = self._ik_ok(x, y, z_tcp + TCP_OFFSET, yaw)
            p = self._ik_ok(x, y, z_tcp + TCP_OFFSET + self.pc.approach, yaw)
            grasp |= g
            pre |= p
            both += int(g and p)
        return grasp, pre, both


def plot(rows: list[dict[str, float]], cfg: SceneConfig, step: float, path: Path) -> None:
    """The map: one ordinal blue ramp (none < pre-grasp only < full), outlines for the cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fg, muted, surface = "#1E2124", "#5A6066", "#FFFFFF"
    colors = {0: "#EEEEEC", 1: "#86B6EF", 2: "#1C5CAB"}  # unreachable, pre-grasp only / grasp only, full
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for r in rows:
        level = (
            2
            if r["grasp_ok"] and r["pregrasp_ok"] and r["n_wrist_ok"] > 0
            else int(bool(r["grasp_ok"] or r["pregrasp_ok"]))
        )
        # Plot in the overhead camera's orientation: right = +y, up = -x (robot at the top).
        ax.add_patch(
            Rectangle(
                (r["y"] - step / 2 + 0.001, -r["x"] - step / 2 + 0.001),
                step - 0.002,
                step - 0.002,
                facecolor=colors[level],
                edgecolor="none",
            )
        )
    a = cfg.items_area
    x0, x1, y0, y1 = a.x_min, a.x_max, a.y_min, a.y_max
    # Overlays sit on the dark fill, so they use light ink.
    light = "#FFFFFF"
    ax.add_patch(Rectangle((y0, -x1), y1 - y0, x1 - x0, fill=False, edgecolor=light, lw=1.5, ls="--"))
    ax.text(y0 + 0.014, -x0 - 0.012, "items area", color=light, fontsize=7, va="top")
    for b in cfg.boxes:
        h = b.size / 2
        ax.add_patch(Rectangle((b.y - h, -b.x - h), b.size, b.size, fill=False, edgecolor=light, lw=1.5))
        ax.text(
            b.y, -b.x, f"tray {b.label}", color=light, fontsize=8, ha="center", va="center", weight="bold"
        )
    ax.plot([0], [0], marker="o", ms=9, mfc=surface, mec=fg, mew=1.5)
    ax.text(0.03, 0.0, "robot base", color=fg, fontsize=7, va="center")
    ys = [r["y"] for r in rows]
    xs = [r["x"] for r in rows]
    ax.set_xlim(min(ys) - step, max(ys) + step)
    ax.set_ylim(-max(xs) - step, max(0.03, -min(xs) + step))
    ax.set_aspect("equal")
    ax.set_xlabel("y (m), as seen by the overhead camera", fontsize=8, color=muted)
    ax.set_ylabel("−x (m), robot at the top", fontsize=8, color=muted)
    ax.tick_params(colors=muted, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#D5D7D8")
    ax.set_title("Top-down reach of the Panda over the sorting table", fontsize=9, color=fg, loc="left")
    ax.legend(
        handles=[
            Patch(facecolor=colors[2], label="grasp and pre-grasp reachable"),
            Patch(facecolor=colors[1], label="partly: not both heights with one wrist angle"),
            Patch(facecolor=colors[0], edgecolor="#D5D7D8", label="unreachable"),
        ],
        fontsize=7,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.12),
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Entry point. Exits via ``os._exit`` (moveit_py teardown crash, troubleshooting #48)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=float, default=0.02, help="grid spacing (m)")
    ap.add_argument("--out", type=Path, default=Path("results/reachability"))
    ap.add_argument("--scene-config", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from an existing CSV")
    args = ap.parse_args(argv)
    cfg = load_config(args.scene_config or default_config_path())
    if args.plot_only:
        with (args.out / "reachability.csv").open() as f:
            rows_in = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        plot(rows_in, cfg, args.step, args.out / "reachability.png")
        return

    rclpy.init()
    moveit = MoveItPy(node_name="reachability", config_dict=moveit_config(use_sim_time=False, offline=True))
    sweeper = Sweeper(moveit, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    for x in np.arange(0.10, 0.86 + 1e-9, args.step):
        for y in np.arange(-0.46, 0.50 + 1e-9, args.step):
            g, p, n = sweeper.point(float(x), float(y), cfg)
            rows.append(
                {
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "grasp_ok": int(g),
                    "pregrasp_ok": int(p),
                    "n_wrist_ok": n,
                }
            )
    with (args.out / "reachability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, cfg, args.step, args.out / "reachability.png")

    a = cfg.items_area
    x0, x1, y0, y1 = a.x_min, a.x_max, a.y_min, a.y_max
    inside = [r for r in rows if x0 <= r["x"] <= x1 and y0 <= r["y"] <= y1]
    full = [r for r in inside if r["n_wrist_ok"] > 0]
    summary: dict[str, object] = {
        "grid_step_m": args.step,
        "items_area_points": len(inside),
        "items_area_reachable_share": round(len(full) / len(inside), 4) if inside else None,
        "min_wrist_angles_ok_in_items_area": min((r["n_wrist_ok"] for r in inside), default=0),
        "box_centres": {b.label: sweeper.point(b.x, b.y, cfg)[2] for b in cfg.boxes},
        "max_reach_m_full": round(
            max((math.hypot(r["x"], r["y"]) for r in rows if r["n_wrist_ok"] > 0), default=0.0), 3
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    rclpy.shutdown()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
