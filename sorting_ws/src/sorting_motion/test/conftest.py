"""Shared fixtures: an isolated simulator process for live MoveIt tests."""

from __future__ import annotations

import ctypes
import os
import random
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PANDA = Path(
    os.environ.get("SORTING_PANDA_XML", ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
)


def _die_with_parent() -> None:
    """Linux: SIGTERM the sim if the test process dies (e.g. a crash skips teardown)."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def ros2_cli(env: dict[str, str], *args: str, timeout: float = 20.0) -> str:
    """Run a ``ros2`` command without a daemon (daemons pile up on random domains); '' on timeout."""
    try:
        return subprocess.run(
            [shutil.which("ros2") or "ros2", *args, "--no-daemon"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except subprocess.TimeoutExpired:
        return ""


@contextmanager
def running_sim(rtf: float, log: Path, seed: int = 1) -> Iterator[dict[str, str]]:
    """Start ``sorting_sim`` on its own ROS domain; yields the env for clients; stops it afterwards.

    Own domain: a sim left running elsewhere can't interfere (two /clock
    sources make time jump and MoveIt times out).
    """
    env = dict(
        os.environ,
        SORTING_PANDA_XML=str(PANDA),
        MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
        # Linux: domain IDs 102-214 put DDS ports in the ephemeral range (random
        # discovery failures); stay within 1-101.
        ROS_DOMAIN_ID=str(random.randint(30, 101)),
        # UDP only: shared-memory segments left in /dev/shm by killed processes
        # can block a new participant (suspected cause of rare hangs).
        FASTDDS_BUILTIN_TRANSPORTS="UDPv4",
        SORTING_HANG_DUMP_S="150",
    )
    info = ros2_cli(env, "topic", "info", "/clock")
    assert "Publisher count: 0" in info or not info, f"/clock already has a publisher on this domain:\n{info}"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sorting_sim.sim_node",
                "--ros-args",
                "-p",
                f"scene_seed:={seed}",
                "-p",
                f"real_time_factor:={rtf}",
            ],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            preexec_fn=_die_with_parent,
        )
    try:
        clock = ros2_cli(env, "topic", "echo", "--once", "/clock", "rosgraph_msgs/msg/Clock", timeout=30.0)
        assert proc.poll() is None and "clock" in clock, (
            f"sim did not start; its log:\n{log.read_text()[-3000:]}"
        )
        yield env
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_moveit_cli(
    env: dict[str, str], log: Path, args: list[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a MoveIt command-line tool in its own process (moveit_py crashes at exit, troubleshooting #48)."""
    import pytest

    try:
        return subprocess.run(
            [sys.executable, *args], env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        errs = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        pytest.fail(
            f"{args} hung for {timeout} s\nstdout:\n{partial[-1500:]}\nstderr:\n{errs[-3000:]}"
            f"\nsim log tail:\n{log.read_text()[-2000:]}"
        )
