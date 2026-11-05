# TouchStone on a real upstream reward bug: ManiSkill PushCube-v1

This is the honest version of the question "is TouchStone useful?". Instead of a reward I wrote
with bugs I planted, this runs TouchStone against a reward bug that a real lab shipped, a real
user hit, and a real maintainer fixed.

**The bug.** ManiSkill's `PushCube-v1` success condition requires the cube inside the goal
radius *and still on the table*. Its dense reward only measured horizontal distance to the goal.
A user reported in [issue 1016](https://github.com/haosulab/ManiSkill/issues/1016) that their
trained policy held the cube above the target instead of pushing it. The maintainers merged
[PR 1022](https://github.com/haosulab/ManiSkill/pull/1022) on 2025-04-29 (commit `dfdc881e`),
adding a height term to the reward.

**The test.** One TouchStone case written from the task's own docstring: with the gripper at the
push pose and the cube over the goal, holding the cube higher must earn strictly less reward.
Two sanity cases confirm the success check itself. One deployment-gate case runs a candidate
policy checkpoint through the real environment and reports what the cube actually did.

## Result on the real code

| | buggy commit `2563643f` (parent of fix) | fixed commit `dfdc881e` (PR 1022) |
|---|---|---|
| normalized reward, cube 3 cm over goal | 0.6667 | 0.7353 |
| normalized reward, cube 15 cm over goal | 0.6667 | 0.6088 |
| gap (low minus high) | **0.0000** | 0.1265 |
| `lifting-cube-over-goal-loses-reward` | **FAIL** | pass |
| `cube-on-table-in-goal-is-success` | pass | pass |
| `cube-lifted-over-goal-is-not-success` | pass | pass |
| `touchstone check` exit code (reward cases) | **2** | 0 |

At the buggy commit the reward is byte-identical whether the cube is 3 cm or 15 cm above the
goal. The success check already knew lifting was wrong; the reward did not. That is the whole
bug, and one declarative case over the real environment catches it. Full records with every
observation and adapter hash are in `runs/gate_buggy.json` and `runs/gate_fixed.json`.

## Result on trained policies (did NOT reproduce the exploit)

I trained PPO on both commits to see whether the policy would learn the lifting hack.
PhysX GPU does not initialise under WSL2, so training used CPU physics with Stable-Baselines3
and 8 parallel envs, 2M steps, 3 seeds per commit (about 36 minutes per run). That is far less
data per update than ManiSkill's 512-env GPU baseline.

| commit | seed | success rate (64 eps) | cube lifted over goal |
|---|---|---|---|
| buggy | 1 | 0.33 | 0.00 |
| buggy | 2 | 0.00 | 0.00 |
| buggy | 3 | 0.00 | 0.00 |
| fixed | 1 | 0.20 | 0.00 |
| fixed | 2 | 0.13 | 0.00 |
| fixed | 3 | 0.14 | 0.00 |

No policy lifted the cube. All of them are simply undertrained. The original report used
DrQ-v2 with visual observations, and the lifting behaviour did not appear with state-based PPO
at this budget. Both seed-1 checkpoints fail the deployment gate on success rate, so the gated
check exits 2 at both commits once the policy case is included (`runs/demo_transcript.txt`).
Two 10M-step runs (seed 11, 10 envs) were started on both commits but had not finished when
this was written, so they are not reported here.

Videos of the seed-1 policies are in `media/buggy_seed1/` and `media/fixed_seed1/`.

## What this does and does not show

Shows:
- TouchStone's model (scripted inputs, independent evidence, declarative assertions, hash-bound
  approvals, CI exit codes) works unchanged against a real GPU-era robotics environment,
  not just code graders.
- A single case written from the task's documented success condition would have failed at the
  commit that shipped the bug, and passes at the fix. The regression stays locked.

Does not show:
- That TouchStone finds bugs. The case was written by me after reading the issue. A reviewer
  would have had to think "does the reward agree with the success condition?" on their own.
  That is a good habit, and TouchStone gives it a home, but it is a habit, not a feature.
- That the policy exploit reproduces with PPO in this compute budget. It did not.
- That this beats a pytest file. Technically it does not. The sell is the review workflow,
  the evidence record, and the CI gate, not detection.

## Layout

- `probe/push_cube_probe.py`  drives the real env: teleports the cube, drives the gripper with
  the real controller, reads success/reward from the env's own `evaluate` and
  `compute_normalized_dense_reward`, rolls out checkpoints.
- `suite/adapter.py`  TouchStone adapter; `MS_ROOT` selects the checkout under test.
- `suite/suite.json`  the four cases.
- `train_cpu.py`  PPO on CPU physics with SB3, baseline hyperparameters.
- `probe/render_policy.py`  record a checkpoint to MP4 through the software renderer.
- `demo.sh`  explore, approve, gated check at both commits.

## Reproduce (gipsydanger, WSL2 Ubuntu 24.04)

```bash
git clone https://github.com/haosulab/ManiSkill.git ~/ManiSkill
cd ~/ManiSkill && git worktree add ~/ms-buggy 2563643f && git worktree add ~/ms-fixed dfdc881e
for d in ms-buggy ms-fixed; do (cd ~/$d && uv venv --python 3.11 .venv && . .venv/bin/activate \
  && uv pip install torch --index-url https://download.pytorch.org/whl/cu128 && uv pip install -e . stable-baselines3 imageio imageio-ffmpeg); done
uv tool install touchstone-us
bash demo.sh
MS_ROOT=~/ms-buggy ~/ms-buggy/.venv/bin/python train_cpu.py --seed 1 --steps 2000000
```
