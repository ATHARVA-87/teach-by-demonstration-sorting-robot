"""Contract test (D8 #3): MoveIt executes through the simulator's action servers.

Starts ``sorting_sim`` in a subprocess, plans with MoveIt (moveit_py) and
executes through ``/panda_arm_controller/follow_joint_trajectory`` and
``/panda_hand_controller/gripper_cmd``. The same test can run against
mujoco_ros2_control or a real Panda driver: only the robot side changes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("moveit.planning")
from conftest import PANDA, run_moveit_cli, running_sim

pytestmark = pytest.mark.skipif(not PANDA.is_file(), reason="run scripts/fetch_robot_models.sh first")


@pytest.fixture(scope="module", params=[1.0, 0.5], ids=["rtf1.0", "rtf0.5"])
def sim(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[dict[str, str], Path]]:
    """The sim at real-time factor 1.0 and 0.5 (the slow run proves MoveIt times executions in sim time)."""
    log = tmp_path_factory.mktemp("sim") / "sim.log"
    with running_sim(request.param, log) as env:
        yield env, log


def test_moveit_executes_through_sim_actions(sim: tuple[dict[str, str], Path]) -> None:
    env, log = sim
    # MoveIt runs in its own process: moveit_py crashes at interpreter exit
    # (see contract_check.main), which would take pytest down with it.
    out = run_moveit_cli(env, log, ["-m", "sorting_motion.contract_check"], timeout=180)
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    assert lines, f"no report; stderr tail:\n{out.stderr[-2000:]}\nsim log tail:\n{log.read_text()[-2000:]}"
    report = json.loads(lines[-1])
    assert report["ok"], json.dumps(report, indent=1) + "\nsim log tail:\n" + log.read_text()[-2000:]
    assert out.returncode == 0
    arm_moves = [m for m in report["moves"] if m["group"] == "panda_arm"]
    assert len(arm_moves) == 2 and all(m["final_error_mrad"] < 5.0 for m in arm_moves)
    assert [m["goal"] for m in report["moves"] if m["group"] == "hand"] == ["close", "open"]
