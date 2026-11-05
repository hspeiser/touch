"""Two reward functions for the waypoint task.

`leaky` is the kind of reward a team writes on day one. It has four real bugs:

  1. The stability bonus is not gated on being airborne, so a drone sitting on
     the ground collects it for free.
  2. Arrival is checked in the horizontal plane only, so parking on the ground
     directly under the waypoint counts as "arrived".
  3. Distance shaping is a per-step bonus, not a potential difference, so
     circling the goal farms it forever.
  4. Position comes from telemetry (`p_est`), not ground truth, so a biased or
     spoofed estimator earns reward the drone did not.

`fixed` closes all four. Both have the same signature so the environment, the
trainer and the TouchStone adapter can swap them with one env var.

Each returns (reward, terms) where terms is a dict of named components so the
adapter can report *which* bonus paid out, not just the total.
"""

from __future__ import annotations

import os

import numpy as np

from .sim import ARRIVE_RADIUS, GOAL

ARRIVAL_BONUS = 10.0     # per step inside the arrival radius
STABILITY_BONUS = 1.0    # per step, scaled by (1 - speed)
SHAPING_GAIN = 2.0
ACTION_COST = 0.05
CRASH_PENALTY = 50.0


def leaky(prev, state, action):
    p = state["p_est"]                      # bug 4: trusts telemetry
    v = state["v"]
    a = np.asarray(action, dtype=float)
    d3 = float(np.linalg.norm(p - GOAL))
    d_xy = float(np.linalg.norm(p[:2] - GOAL[:2]))   # bug 2: horizontal only
    speed = float(np.linalg.norm(v))

    arrival = ARRIVAL_BONUS if d_xy < ARRIVE_RADIUS else 0.0
    shaping = SHAPING_GAIN / (1.0 + d3)              # bug 3: farmable per-step bonus
    stability = STABILITY_BONUS * (1.0 - min(speed, 1.0))  # bug 1: paid on the ground too
    action_cost = -ACTION_COST * float(np.sum(a * a))
    crash = 0.0                                      # no crash penalty at all
    terms = dict(arrival=arrival, shaping=shaping, stability=stability,
                 action_cost=action_cost, crash=crash)
    return sum(terms.values()), terms


def fixed(prev, state, action):
    p = state["p"]                          # ground truth from the simulator
    v = state["v"]
    a = np.asarray(action, dtype=float)
    airborne = not state["on_ground"] and not state["crashed"]
    d3 = float(np.linalg.norm(p - GOAL))
    d3_prev = float(np.linalg.norm(prev["p"] - GOAL))
    speed = float(np.linalg.norm(v))

    arrival = ARRIVAL_BONUS if (airborne and d3 < ARRIVE_RADIUS) else 0.0
    shaping = SHAPING_GAIN * 2.5 * (d3_prev - d3)   # potential-based: nets to zero on a loop
    stability = STABILITY_BONUS * (1.0 - min(speed, 1.0)) if airborne else 0.0
    action_cost = -ACTION_COST * float(np.sum(a * a))
    crash = -CRASH_PENALTY if (state["crashed"] and not prev["crashed"]) else 0.0
    terms = dict(arrival=arrival, shaping=shaping, stability=stability,
                 action_cost=action_cost, crash=crash)
    return sum(terms.values()), terms


REWARDS = {"leaky": leaky, "fixed": fixed}


def selected_reward():
    """The reward under test, chosen by DRONE_REWARD (default: leaky)."""
    name = os.environ.get("DRONE_REWARD", "leaky")
    if name not in REWARDS:
        raise SystemExit(f"DRONE_REWARD must be one of {sorted(REWARDS)}, got {name!r}")
    return name, REWARDS[name]


def episode_terminates(state, name: str) -> bool:
    """The fixed reward ends the episode on a crash; the leaky one keeps paying."""
    return name == "fixed" and state["crashed"]
