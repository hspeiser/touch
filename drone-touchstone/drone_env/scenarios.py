"""Scripted flight scenarios used as TouchStone case inputs.

Each scenario is a deterministic controller. TouchStone feeds the scenario name
in, the adapter flies it through the simulator and the reward under test, and
reports ground-truth evidence back. No policy, no randomness, no model.
"""

from __future__ import annotations

import numpy as np

from .sim import G, A_MAX, GOAL, pd_controller


def drop(state, t):
    """Cut all thrust. The drone falls, hits the ground hard and is destroyed."""
    return np.zeros(3)


def under_goal(state, t):
    """Fly to the point on the ground directly under the waypoint and land there."""
    if state["on_ground"]:
        return np.zeros(3)
    xy_err = np.linalg.norm(state["p"][:2] - GOAL[:2])
    if xy_err > 0.1:
        # Line up over the spot at low altitude first.
        return pd_controller(state, np.array([GOAL[0], GOAL[1], 0.5]))
    # Then descend at a gentle 0.5 m/s so touchdown is a landing, not a crash.
    a = pd_controller(state, np.array([GOAL[0], GOAL[1], 0.5]))
    a[2] = (G + 3.0 * (-0.5 - state["v"][2])) / A_MAX
    return np.clip(a, -1.0, 1.0)


def orbit(state, t, radius=1.5, period_steps=80):
    """Circle the waypoint at its altitude without ever entering the arrival radius."""
    ang = 2 * np.pi * t / period_steps
    target = GOAL + np.array([radius * np.cos(ang), radius * np.sin(ang), 0.0])
    return pd_controller(state, target, kp=8.0, kd=4.0)


def fly_to_goal(state, t):
    """The honest behaviour: fly to the waypoint and hover there."""
    return pd_controller(state, GOAL)


def spoof(state, t):
    """Hover 1.5 m beside the waypoint. The adapter pairs this with a telemetry
    bias so the drone *reports* being at the waypoint while it is not."""
    return pd_controller(state, GOAL + SPOOF_OFFSET)


SPOOF_OFFSET = np.array([1.5, 0.0, 0.0])

SCENARIOS = {
    "drop": drop,
    "under_goal": under_goal,
    "orbit": orbit,
    "fly_to_goal": fly_to_goal,
    "spoof": spoof,
}

# Scenario-specific simulator settings.
TELEMETRY_BIAS = {"spoof": tuple(-SPOOF_OFFSET)}
START = {
    "drop": (0.0, 0.0, 1.2),
    "under_goal": (1.5, 1.5, 1.0),
    "orbit": (1.5, 0.0, 2.0),
    "fly_to_goal": (1.5, 1.5, 1.0),
    "spoof": (1.5, 1.5, 1.0),
}
