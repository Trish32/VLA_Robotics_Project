"""MuJoCo pick-and-place arena, and the closed loop that runs the stack on it.

Nothing here is imported at package level: `env.py` needs `mujoco`, and importing the
package must not require it, so the rest of the repo still imports on a machine where
the simulator is not installed.
"""
