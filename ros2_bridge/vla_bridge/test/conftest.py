"""Skip rclpy-dependent modules without aborting collection.

This suite is deliberately run in two places with different things installed:

    ROS2 container  rclpy yes, gr00t no   (the deployment target; no torch by design)
    Mac, grootN1_Robotics   rclpy no,  gr00t yes  (where the wire format is cross-checked)

Neither environment can run the whole suite, and that is the point — the parts that must
agree with upstream are testable where upstream lives, and the parts that must run in ROS
are testable where ROS lives.

`collect_ignore` is used rather than a module-level skip because raising `Skipped` (or an
ImportError) during collection takes the entire directory with it: `pytest test` then
reports "1 skipped" and exits 0, having run none of the other sixty tests. That happened
once already — see ../../bug_log.txt [6] — and it is worth going out of the way to avoid,
because a suite that passes by not running looks exactly like a suite that passes.
"""

import importlib.util

collect_ignore = []

if importlib.util.find_spec("rclpy") is None:
    # Needs a ROS2 environment: run it in the container.
    collect_ignore.append("test_node_endtoend.py")
    collect_ignore.append("test_slam_node_endtoend.py")
