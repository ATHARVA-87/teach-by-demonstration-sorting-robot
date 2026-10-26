"""End to end (M3): the scripted executor sorts a randomized scene in the simulator.

Ground-truth poses in, scored episode out. Runs one small scene at 3x real
time; the full 10-scene numbers come from ``ros2 run sorting_motion pick_place``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("moveit.planning")

from conftest import PANDA, run_moveit_cli, running_sim  # noqa: E402

pytestmark = pytest.mark.skipif(not PANDA.is_file(), reason="run scripts/fetch_robot_models.sh first")


def test_scripted_pick_and_place_sorts_a_scene(tmp_path: Path) -> None:
    log = tmp_path / "sim.log"
    out_dir = tmp_path / "out"
    with running_sim(3.0, log) as env:
        out = run_moveit_cli(
            env,
            log,
            ["-m", "sorting_motion.executor", "--seeds", "0", "--n-items", "4", "--out", str(out_dir)],
            timeout=300,
        )
    assert out.returncode == 0, f"{out.stderr[-3000:]}\nsim log:\n{log.read_text()[-2000:]}"
    summary = json.loads((out_dir / "summary.json").read_text())
    episode = json.loads((out_dir / "episode_0000.json").read_text())
    assert summary["n_items"] == 4
    assert summary["sorting_accuracy"] >= 0.75, json.dumps(episode, indent=1)
    for item in episode["items"]:  # every item accounted for, with where it ended up
        assert item["status"] in (
            "picked_and_released",
            "skipped",
            "grasp_missed",
            "dropped",
            "execution_failed",
        )
        assert len(item["final_xyz"]) == 3
