"""Tests for sorting_common.node_runner: clean shutdown and error propagation."""

import pytest
import rclpy
from rclpy.node import Node

from sorting_common.node_runner import run_node


def test_factory_error_propagates_and_rclpy_is_shut_down() -> None:
    def broken() -> Node:
        raise RuntimeError("bad config")

    with pytest.raises(RuntimeError, match="bad config"):
        run_node(broken)
    assert not rclpy.ok()


def test_external_shutdown_exits_cleanly() -> None:
    class StopsItself(Node):
        def __init__(self) -> None:
            super().__init__("stops_itself")
            self.create_timer(0.05, rclpy.shutdown)

    run_node(StopsItself)  # returns instead of hanging or raising
    assert not rclpy.ok()
