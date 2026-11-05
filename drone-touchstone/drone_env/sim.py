"""Minimal 3-D quadrotor point-mass simulator.

State: position p (m), velocity v (m/s), ground-contact and crash flags.
Action: normalised acceleration command in [-1, 1]^3, scaled by A_MAX. The
controller must supply its own gravity compensation (a_z ~ G / A_MAX to hover).

Telemetry: the drone's *reported* position is the true position plus an
estimator bias. A perfect estimator has zero bias. Graders that trust telemetry
instead of ground truth are exactly the kind of bug TouchStone is meant to catch.
"""

from __future__ import annotations

import numpy as np

G = 9.81          # m/s^2
DT = 0.05         # s per step
A_MAX = 15.0      # m/s^2 commanded acceleration limit per axis
DRAG = 0.3        # linear drag coefficient
CRASH_SPEED = 2.0 # touchdown faster than this is a crash; drone is then stuck
GOAL = np.array([0.0, 0.0, 2.0])
ARRIVE_RADIUS = 0.5
MAX_STEPS = 200   # 10 s episodes
ARENA_XY = 4.0    # soft arena half-width; leaving it ends the episode


class Quadrotor:
    def __init__(self, seed: int = 0, telemetry_bias=(0.0, 0.0, 0.0)):
        self.rng = np.random.default_rng(seed)
        self.telemetry_bias = np.asarray(telemetry_bias, dtype=float)
        self.reset()

    def reset(self, start=None):
        if start is None:
            start = np.array(
                [
                    self.rng.uniform(-2.0, 2.0),
                    self.rng.uniform(-2.0, 2.0),
                    self.rng.uniform(0.5, 1.5),
                ]
            )
        self.p = np.asarray(start, dtype=float).copy()
        self.v = np.zeros(3)
        self.on_ground = False
        self.crashed = False
        self.t = 0
        return self.state()

    # --- observation helpers -------------------------------------------------
    def state(self) -> dict:
        return {
            "p": self.p.copy(),
            "v": self.v.copy(),
            "p_est": self.p + self.telemetry_bias,
            "on_ground": self.on_ground,
            "crashed": self.crashed,
            "t": self.t,
        }

    def dist_true(self) -> float:
        return float(np.linalg.norm(self.p - GOAL))

    # --- dynamics ----------------------------------------------------------------
    def step(self, action) -> dict:
        a_norm = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.t += 1
        if self.crashed:
            # A crashed drone does not fly again this episode.
            return self.state()
        acc = a_norm * A_MAX - np.array([0.0, 0.0, G]) - DRAG * self.v
        if self.on_ground and acc[2] <= 0.0:
            # Resting on the ground: the ground pushes back, no vertical motion.
            acc[2] = 0.0
            self.v[2] = 0.0
            # Ground friction kills lateral drift.
            self.v[:2] *= 0.5
        self.v = self.v + acc * DT
        self.p = self.p + self.v * DT
        if self.p[2] <= 0.0:
            speed = float(np.linalg.norm(self.v))
            self.p[2] = 0.0
            if not self.on_ground and speed > CRASH_SPEED:
                self.crashed = True
            self.on_ground = True
            self.v[:] = 0.0
        else:
            self.on_ground = False
        return self.state()

    def out_of_arena(self) -> bool:
        return bool(np.any(np.abs(self.p[:2]) > ARENA_XY) or self.p[2] > 6.0)


def pd_controller(state, target, kp=4.0, kd=3.0):
    """Hover-capable PD controller returning a normalised action."""
    target = np.asarray(target, dtype=float)
    acc = kp * (target - state["p"]) + kd * (0.0 - state["v"]) + np.array([0.0, 0.0, G])
    return np.clip(acc / A_MAX, -1.0, 1.0)
