"""Render one episode (scripted scenario or trained policy) to an MP4.

    DRONE_REWARD=leaky .venv/bin/python render.py --scenario policy --policy ppo_leaky.zip --out media/hack.mp4
    DRONE_REWARD=fixed .venv/bin/python render.py --scenario fly_to_goal --out media/honest.mp4

Side view (x-z) and top view (x-y), the waypoint's 0.5 m arrival sphere, the
flight trail, and a live return counter. Red tint when the drone is on the ground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from drone_env.evaluate import evaluate
from drone_env.sim import ARRIVE_RADIUS, DT, GOAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="fly_to_goal")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="media/episode.mp4")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    inputs = {"scenario": args.scenario, "seed": args.seed}
    if args.policy:
        inputs["policy"] = args.policy
    evidence, frames = evaluate(inputs, record=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(evidence, indent=2))

    label = args.title or (f"PPO policy trained on '{evidence['grader']}' reward"
                           if args.policy else f"scenario '{args.scenario}' under '{evidence['grader']}' reward")
    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=100)
    fig.patch.set_facecolor("#0e1116")
    for ax in (ax_side, ax_top):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#9aa4b2")
        for s in ax.spines.values():
            s.set_color("#2d333b")
    ax_side.set_xlim(-3, 3); ax_side.set_ylim(-0.1, 4); ax_side.set_aspect("equal")
    ax_side.set_title("side view (x, z)", color="#e6edf3")
    ax_side.axhspan(-0.1, 0, color="#3b2a1a")
    ax_side.add_patch(Circle((GOAL[0], GOAL[2]), ARRIVE_RADIUS, fill=False, ec="#3fb950", ls="--", lw=1.5))
    ax_side.plot(GOAL[0], GOAL[2], "+", color="#3fb950", ms=14, mew=2)
    ax_top.set_xlim(-3, 3); ax_top.set_ylim(-3, 3); ax_top.set_aspect("equal")
    ax_top.set_title("top view (x, y)", color="#e6edf3")
    ax_top.add_patch(Circle((GOAL[0], GOAL[1]), ARRIVE_RADIUS, fill=False, ec="#3fb950", ls="--", lw=1.5))
    ax_top.plot(GOAL[0], GOAL[1], "+", color="#3fb950", ms=14, mew=2)

    trail_side, = ax_side.plot([], [], color="#58a6ff", lw=1, alpha=0.6)
    trail_top, = ax_top.plot([], [], color="#58a6ff", lw=1, alpha=0.6)
    drone_side, = ax_side.plot([], [], "o", color="#58a6ff", ms=10)
    drone_top, = ax_top.plot([], [], "o", color="#58a6ff", ms=10)
    est_side, = ax_side.plot([], [], "x", color="#d29922", ms=9, mew=2)
    text = fig.text(0.5, 0.95, "", ha="center", va="top", color="#e6edf3", fontsize=12, family="monospace")
    fig.text(0.5, 0.995, label, ha="center", va="top", color="#8b949e", fontsize=11)

    writer = imageio.get_writer(args.out, fps=int(round(1 / DT)), codec="libx264", quality=8,
                                macro_block_size=None)
    P = np.array([f["p"] for f in frames])
    for i, f in enumerate(frames):
        p = f["p"]; pe = f["p_est"]
        trail_side.set_data(P[: i + 1, 0], P[: i + 1, 2]); trail_top.set_data(P[: i + 1, 0], P[: i + 1, 1])
        drone_side.set_data([p[0]], [p[2]]); drone_top.set_data([p[0]], [p[1]])
        spoofed = np.linalg.norm(np.array(pe) - np.array(p)) > 1e-6
        est_side.set_data([pe[0]], [pe[2]]) if spoofed else est_side.set_data([], [])
        colour = "#f85149" if f["crashed"] else ("#d29922" if f["on_ground"] else "#58a6ff")
        drone_side.set_color(colour); drone_top.set_color(colour)
        state = "CRASHED" if f["crashed"] else ("on ground" if f["on_ground"] else "airborne")
        d = np.linalg.norm(np.array(p) - GOAL)
        text.set_text(f"t={i * DT:5.2f}s  {state:9s}  dist to goal={d:4.2f} m   "
                      f"step reward={f['r']:+6.2f}   return={f['ret']:8.1f}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    # Hold the last frame for a second.
    for _ in range(int(1 / DT)):
        writer.append_data(img)
    writer.close()
    print(json.dumps(evidence, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
