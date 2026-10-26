"""Shared ``main()`` logic: start a node, spin it, and shut down cleanly on Ctrl+C."""

import logging
import signal
from collections.abc import Callable

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

_log = logging.getLogger(__name__)


def run_node(node_factory: Callable[[], Node], args: list[str] | None = None) -> None:
    """Create a node with ``node_factory()``, spin until Ctrl+C, then clean up.

    On Ctrl+C, rclpy shuts its context down from the signal handler. Depending
    on timing, ``spin()`` then raises KeyboardInterrupt, ExternalShutdownException,
    or an RCLError ("context is not valid"). All three mean "we are stopping",
    so they are swallowed only when the context really is shut down; any
    other exception (or one raised while the context is still valid) propagates.
    """
    rclpy.init(args=args)
    node: Node | None = None
    try:
        node = node_factory()
        rclpy.spin(node)
    except KeyboardInterrupt, ExternalShutdownException:
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error, not a shutdown race
        _log.debug("Exception during shutdown ignored (context already invalid)", exc_info=True)
    finally:
        # In a terminal, Ctrl+C reaches both `ros2 launch` and the node, and
        # launch then forwards a second SIGINT. Ignore further SIGINTs while
        # cleaning up, so a second one can't interrupt destroy_node() halfway
        # (seen: KeyboardInterrupt inside timer destruction, exit code -2).
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            signal.signal(signal.SIGINT, previous)
