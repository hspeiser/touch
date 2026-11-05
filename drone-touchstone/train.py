"""Train PPO on the drone waypoint task against the selected reward.

    DRONE_REWARD=leaky .venv/bin/python train.py --steps 3000000
    DRONE_REWARD=fixed .venv/bin/python train.py --steps 3000000

Saves policies/ppo_<reward>.zip and prints what the trained policy actually
does, measured from simulator ground truth.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from drone_env.gym_env import DroneWaypointEnv
from drone_env.rewards import selected_reward
from drone_env.sim import MAX_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu is faster than cuda for a tiny MLP policy")
    args = ap.parse_args()

    name, _ = selected_reward()
    out = Path("policies") / f"ppo_{name}.zip"
    out.parent.mkdir(exist_ok=True)
    device = args.device
    torch.set_num_threads(4)
    print(f"reward={name} device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    env = make_vec_env(DroneWaypointEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy", env, device=device, seed=args.seed, verbose=0,
        n_steps=MAX_STEPS * 2, batch_size=1600, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f"runs/tb_{name}",
    )
    t0 = time.time()
    ckpt = CheckpointCallback(save_freq=max(250_000 // args.n_envs, 1), save_path=f"policies/ckpt_{name}",
                              name_prefix="ppo", verbose=0)
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=ckpt)
    print(f"trained {args.steps:,} steps in {time.time() - t0:.0f}s")
    model.save(out)
    print(f"saved {out}")

    # What did it learn? Ground truth, 20 episodes.
    from drone_env.evaluate import run_episode, policy_controller
    ctrl = policy_controller(str(out))
    rows = [run_episode(ctrl, seed=s)[0] for s in range(20)]
    print("\nground-truth behaviour of the trained policy (20 episodes):")
    for k in ["return", "arrival_bonus", "steps_at_goal", "min_dist_to_goal"]:
        print(f"  mean {k:16s} {np.mean([r[k] for r in rows]):10.2f}")
    print(f"  crashed            {sum(r['crashed'] for r in rows)}/20")
    print(f"  on ground at end   {sum(r['on_ground_at_end'] for r in rows)}/20")
    print(f"  reached goal       {sum(r['reached_goal'] for r in rows)}/20")
    print(f"  example final pos  {rows[0]['final_pos']}")


if __name__ == "__main__":
    main()
