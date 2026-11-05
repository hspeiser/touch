"""Gymnasium wrapper so Stable-Baselines3 can train against either reward."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .rewards import episode_terminates, selected_reward
from .sim import GOAL, MAX_STEPS, Quadrotor


def make_obs(state):
    return np.concatenate([state["p"], state["v"], GOAL - state["p"]]).astype(np.float32)


class DroneWaypointEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.name, self.reward_fn = selected_reward()
        self.sim = Quadrotor(seed=seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.state = self.sim.reset()
        return make_obs(self.state), {}

    def step(self, action):
        prev = self.state
        self.state = self.sim.step(action)
        r, terms = self.reward_fn(prev, self.state, action)
        terminated = episode_terminates(self.state, self.name) or self.sim.out_of_arena()
        truncated = self.state["t"] >= MAX_STEPS
        info = {"crashed": self.state["crashed"], "on_ground": self.state["on_ground"],
                "dist": self.sim.dist_true(), **terms}
        return make_obs(self.state), float(r), terminated, truncated, info
