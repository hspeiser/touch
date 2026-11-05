"""Probe the real ManiSkill PushCube-v1 environment and report ground-truth evidence.

Reads a JSON case input on stdin, prints a JSON evidence object on stdout.
Runs inside whichever ManiSkill checkout's venv invokes it (MS_ROOT), so the
same probe exercises the buggy commit and the fixed commit unchanged.

Scenarios
  height_gradient   drive the gripper to the push pose at two heights, teleport the
                    cube into the push pose and the goal under it, and compare the
                    normalized dense reward low vs high. The success check says a
                    lifted cube is not success; does the reward agree?
  on_table_success  cube on the table inside the goal -> success must be true
  lifted_not_success cube 10 cm above the goal -> success must be false
  policy            roll out a trained PPO checkpoint for N episodes
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  (registers PushCube-v1)
from mani_skill.utils.structs import Pose

HERE = Path(__file__).resolve().parent


def make_env(num_envs=1, control_mode="pd_ee_delta_pos", seed=0):
    env = gym.make(
        "PushCube-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode=control_mode,
        reward_mode="normalized_dense",
        sim_backend="physx_cpu" if num_envs == 1 else "physx_cuda",
        render_mode=None,
        render_backend=os.environ.get("MS_RENDER_BACKEND", "cpu"),
    )
    env.reset(seed=seed)
    return env


def drive_tcp_to_height(env, z_target, max_steps=60):
    """Move the end effector vertically with the real controller until it is near z_target."""
    base = env.unwrapped
    for _ in range(max_steps):
        z = float(base.agent.tcp.pose.p[0, 2])
        dz = z_target - z
        if abs(dz) < 0.004:
            break
        act = np.zeros((1, env.action_space.shape[-1]), dtype=np.float32)
        act[0, 2] = float(np.clip(dz * 20, -1, 1))
        env.step(act)
    return float(base.agent.tcp.pose.p[0, 2])


def place_cube_in_push_pose(env):
    """Teleport the cube so the gripper is exactly at the task's push pose, and the goal under the cube."""
    base = env.unwrapped
    tcp = base.agent.tcp.pose.p.clone()
    cube_p = tcp + torch.tensor([[base.cube_half_size + 0.005, 0.0, 0.0]])
    base.obj.set_pose(Pose.create_from_pq(p=cube_p, q=torch.tensor([[1.0, 0, 0, 0]])))
    base.obj.set_linear_velocity(torch.zeros((1, 3)))
    base.obj.set_angular_velocity(torch.zeros((1, 3)))
    goal_p = base.goal_region.pose.p.clone()
    goal_p[:, :2] = cube_p[:, :2]
    base.goal_region.set_pose(Pose.create_from_pq(p=goal_p, q=base.goal_region.pose.q))
    if base.gpu_sim_enabled:
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
    return float(cube_p[0, 2])


def measure(env):
    base = env.unwrapped
    info = base.evaluate()
    reward = base.compute_normalized_dense_reward(obs=None, action=None, info=info)
    tcp = base.agent.tcp.pose.p[0]
    cube = base.obj.pose.p[0]
    push_pose = cube + torch.tensor([-base.cube_half_size - 0.005, 0.0, 0.0])
    return {
        "success": bool(info["success"][0]),
        "reward": round(float(reward[0]), 4),
        "cube_z": round(float(cube[2]), 4),
        "tcp_to_push_pose": round(float(torch.linalg.norm(push_pose - tcp)), 4),
    }


def height_gradient(inputs):
    env = make_env()
    z_low = float(inputs.get("z_low", 0.035))
    z_high = float(inputs.get("z_high", 0.15))
    out = {}
    for name, z in (("low", z_low), ("high", z_high)):
        drive_tcp_to_height(env, z)
        place_cube_in_push_pose(env)
        out[name] = measure(env)
    env.close()
    gap = round(out["low"]["reward"] - out["high"]["reward"], 4)
    return {
        "low": out["low"],
        "high": out["high"],
        "reward_gap_low_minus_high": gap,
        "reward_depends_on_cube_height": gap > 0.01,
    }


def set_cube_at_goal(env, z):
    base = env.unwrapped
    goal_p = base.goal_region.pose.p.clone()
    cube_p = goal_p.clone()
    cube_p[:, 2] = z
    base.obj.set_pose(Pose.create_from_pq(p=cube_p, q=torch.tensor([[1.0, 0, 0, 0]])))
    if base.gpu_sim_enabled:
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()


def on_table_success(inputs):
    env = make_env()
    set_cube_at_goal(env, env.unwrapped.cube_half_size)
    m = measure(env)
    env.close()
    return m


def lifted_not_success(inputs):
    env = make_env()
    set_cube_at_goal(env, float(inputs.get("z", 0.12)))
    m = measure(env)
    env.close()
    return m


def policy(inputs):
    """Roll out a Stable-Baselines3 PPO checkpoint in the real env and report what the cube actually did."""
    from stable_baselines3 import PPO
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    episodes = int(inputs.get("episodes", 64))
    seed = int(inputs.get("seed", 0))
    ckpt = Path(os.environ["MS_ROOT"]) / "runs" / inputs["checkpoint"]
    model = PPO.load(str(ckpt), device="cpu")
    env = CPUGymWrapper(make_env(control_mode="pd_joint_delta_pos", seed=seed))
    base = env.unwrapped
    n_success_end = n_success_ever = n_lifted_end = n_over_goal = n_over_goal_lifted = 0
    final_z = []; max_z = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ever = False; mz = 0.0
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            z = float(base.obj.pose.p[0, 2]); mz = max(mz, z)
            ever |= bool(info["success"])
        z = float(base.obj.pose.p[0, 2])
        d = float(torch.linalg.norm(base.obj.pose.p[0, :2] - base.goal_region.pose.p[0, :2]))
        n_success_end += bool(info["success"]); n_success_ever += ever
        n_lifted_end += z > 0.03; n_over_goal += d < 0.1; n_over_goal_lifted += (d < 0.1 and z > 0.03)
        final_z.append(z); max_z.append(mz)
    env.close()
    return {
        "episodes": episodes,
        "success_rate_at_end": round(n_success_end / episodes, 3),
        "success_rate_ever": round(n_success_ever / episodes, 3),
        "mean_final_cube_z": round(float(np.mean(final_z)), 4),
        "mean_max_cube_z": round(float(np.mean(max_z)), 4),
        "frac_cube_lifted_at_end": round(n_lifted_end / episodes, 3),
        "frac_over_goal_xy_at_end": round(n_over_goal / episodes, 3),
        "frac_over_goal_but_lifted": round(n_over_goal_lifted / episodes, 3),
    }


SCENARIOS = {
    "height_gradient": height_gradient,
    "on_table_success": on_table_success,
    "lifted_not_success": lifted_not_success,
    "policy": policy,
}

if __name__ == "__main__":
    inputs = json.loads(sys.stdin.read() or "{}")
    fn = SCENARIOS[inputs["scenario"]]
    evidence = fn(inputs)
    evidence["ms_root"] = os.environ.get("MS_ROOT", "")
    json.dump(evidence, sys.stdout)
