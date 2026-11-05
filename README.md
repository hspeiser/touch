# TouchStone experiments

Two experiments with [TouchStone](https://www.touchstone-us.com/) (`touchstone-us` on PyPI),
a CLI for deterministic, human-reviewed regression tests of RL reward functions.

- `maniskill-touchstone/`  TouchStone against a real upstream reward bug in ManiSkill
  `PushCube-v1` (issue 1016 / PR 1022). The reward case fails at the buggy commit and passes at
  the fix. Training PPO on both commits did not reproduce the lifting exploit. Start here.
- `drone-touchstone/`  a self-contained quadrotor demo with a deliberately leaky reward and a
  fixed one. The bugs were planted, so it shows the workflow, not detection.

Each folder has its own README with results, caveats and reproduction steps. Training ran on a
single RTX 5090 box under WSL2; ManiSkill training used CPU physics because PhysX GPU does not
initialise there.
