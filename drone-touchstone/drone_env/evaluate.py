"""Fly one scenario (or one trained policy) through the reward under test.

Reads a JSON case input on stdin, writes a JSON evidence object on stdout.
Everything reported is ground truth from the simulator, never the reward's own
opinion of what happened, which is what makes it usable as TouchStone evidence.

    {"scenario": "drop"}                       scripted controller
    {"scenario": "policy", "policy": "ppo_leaky.zip", "seed": 3}

Environment: DRONE_REWARD=leaky|fixed selects the reward function under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .rewards import episode_terminates, selected_reward
from .scenarios import SCENARIOS, START, TELEMETRY_BIAS
from .sim import ARRIVE_RADIUS, GOAL, MAX_STEPS, Quadrotor


def run_episode(controller, *, start=None, telemetry_bias=(0, 0, 0), seed=0, record=False):
    name, reward_fn = selected_reward()
    sim = Quadrotor(seed=seed, telemetry_bias=telemetry_bias)
    state = sim.reset(start)
    totals = {}
    ret = 0.0
    reached = False
    min_dist = sim.dist_true()
    arrived_steps = 0
    frames = []
    steps = 0
    for t in range(MAX_STEPS):
        action = controller(state, t)
        prev = state
        state = sim.step(action)
        r, terms = reward_fn(prev, state, action)
        ret += r
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.0) + v
        d = sim.dist_true()
        min_dist = min(min_dist, d)
        airborne = not state["on_ground"] and not state["crashed"]
        if airborne and d < ARRIVE_RADIUS:
            reached = True
            arrived_steps += 1
        if record:
            frames.append(dict(p=state["p"].tolist(), p_est=state["p_est"].tolist(),
                               r=r, ret=ret, crashed=bool(state["crashed"]),
                               on_ground=bool(state["on_ground"])))
        steps = t + 1
        if episode_terminates(state, name) or sim.out_of_arena():
            break
    evidence = {
        "grader": name,
        "steps": steps,
        "return": round(ret, 3),
        "arrival_bonus": round(totals.get("arrival", 0.0), 3),
        "shaping_total": round(totals.get("shaping", 0.0), 3),
        "stability_total": round(totals.get("stability", 0.0), 3),
        "crashed": bool(state["crashed"]),
        "on_ground_at_end": bool(state["on_ground"]),
        "reached_goal": reached,
        "steps_at_goal": arrived_steps,
        "min_dist_to_goal": round(min_dist, 3),
        "final_pos": [round(float(x), 3) for x in state["p"]],
        "final_reported_pos": [round(float(x), 3) for x in state["p_est"]],
        "goal": GOAL.tolist(),
    }
    return evidence, frames


def policy_controller(path: str):
    from stable_baselines3 import PPO
    from .gym_env import make_obs

    model = PPO.load(path, device="cpu")

    def controller(state, t):
        action, _ = model.predict(make_obs(state), deterministic=True)
        return action

    return controller


def evaluate(inputs: dict, record=False):
    scenario = inputs["scenario"]
    seed = int(inputs.get("seed", 0))
    if scenario == "policy":
        root = Path(__file__).resolve().parent.parent
        controller = policy_controller(str(root / "policies" / inputs["policy"]))
        return run_episode(controller, seed=seed, record=record)
    if scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    return run_episode(
        SCENARIOS[scenario],
        start=START.get(scenario),
        telemetry_bias=TELEMETRY_BIAS.get(scenario, (0, 0, 0)),
        seed=seed,
        record=record,
    )


if __name__ == "__main__":
    inputs = json.loads(sys.stdin.read() or "{}")
    evidence, _ = evaluate(inputs)
    json.dump(evidence, sys.stdout)
