"""Record a PPO checkpoint (or random actions) acting in the real PushCube-v1 env to MP4.

    MS_ROOT=~/ms-buggy $MS_ROOT/.venv/bin/python probe/render_policy.py --checkpoint ppo_seed1.zip --episodes 4 --out media/buggy_seed1
Uses the software (lavapipe) renderer, so it is slow but works under WSL2.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = gym.make("PushCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos",
                   reward_mode="normalized_dense", sim_backend="physx_cpu", render_backend="cpu",
                   render_mode="rgb_array")
    env = RecordEpisode(env, output_dir=args.out, save_trajectory=False, video_fps=20,
                        max_steps_per_video=50 * args.episodes)
    env = CPUGymWrapper(env)
    model = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        model = PPO.load(str(Path(os.environ["MS_ROOT"]).expanduser() / "runs" / args.checkpoint), device="cpu")
    base = env.unwrapped
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for _ in range(50):
            action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
        print(f"episode {ep}: success={bool(info['success'])} cube_z={float(base.obj.pose.p[0, 2]):.3f}")
    env.close()
    print("wrote", sorted(Path(args.out).glob("*.mp4")))


if __name__ == "__main__":
    main()
