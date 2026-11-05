"""Train PPO on the real ManiSkill PushCube-v1 using CPU physics in parallel processes.

PhysX GPU does not initialise under WSL2, so this mirrors the ManiSkill PPO
baseline's hyperparameters (gamma 0.8, lambda 0.9, 50-step rollouts, 4 epochs)
on Stable-Baselines3 with SubprocVecEnv instead of the 512-env GPU batch.

    MS_ROOT=~/ms-buggy  $MS_ROOT/.venv/bin/python train_cpu.py --seed 1 --steps 1500000
Saves $MS_ROOT/runs/ppo_seed<seed>.zip and prints ground-truth behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def make_one(seed):
    def _f():
        import mani_skill.envs  # noqa
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        env = gym.make("PushCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos",
                       reward_mode="normalized_dense", sim_backend="physx_cpu", render_backend="cpu",
                       render_mode=None)
        env = CPUGymWrapper(env)
        env.reset(seed=seed)
        return env
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    args = ap.parse_args()
    root = Path(os.environ["MS_ROOT"]).expanduser()
    out = root / "runs" / f"ppo_seed{args.seed}.zip"
    out.parent.mkdir(exist_ok=True)
    torch.set_num_threads(2)

    env = VecMonitor(SubprocVecEnv([make_one(args.seed * 1000 + i) for i in range(args.n_envs)]))
    model = PPO(
        "MlpPolicy", env, device="cpu", seed=args.seed, verbose=0,
        n_steps=50, batch_size=200, n_epochs=4, learning_rate=3e-4,
        gamma=0.8, gae_lambda=0.9, ent_coef=0.0, clip_range=0.2, max_grad_norm=0.5,
        target_kl=0.1, policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
                                           activation_fn=torch.nn.Tanh),
        tensorboard_log=str(root / "runs" / "tb"),
    )
    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    print(f"[{root.name} seed {args.seed}] trained {args.steps:,} steps in {time.time() - t0:.0f}s", flush=True)
    model.save(out)
    env.close()

    probe = Path(__file__).resolve().parent / "probe" / "push_cube_probe.py"
    inp = json.dumps({"scenario": "policy", "checkpoint": out.name, "episodes": 64, "seed": 0})
    res = subprocess.run([sys.executable, str(probe)], input=inp, capture_output=True, text=True,
                         env={**os.environ, "MS_ROOT": str(root)})
    print(res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-2000:], flush=True)


if __name__ == "__main__":
    main()
