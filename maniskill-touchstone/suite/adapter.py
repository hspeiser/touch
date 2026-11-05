"""TouchStone adapter for ManiSkill PushCube-v1.

MS_ROOT selects which ManiSkill checkout is under test (its .venv is used):
    MS_ROOT=~/ms-buggy   commit 2563643f, the parent of the reward fix
    MS_ROOT=~/ms-fixed   commit dfdc881e, PR #1022 "Updated and more stable reward for PushCubeEnv"
PROBE_DIR points at the probe script (default: ../probe next to this suite's project).
"""

import json
import os
import subprocess
from pathlib import Path

MS_ROOT = Path(os.environ["MS_ROOT"]).expanduser()
PROBE_DIR = Path(os.environ.get("PROBE_DIR", Path.home() / "maniskill-touchstone" / "probe"))
PYTHON = MS_ROOT / ".venv" / "bin" / "python"


def setup(case):
    return {}


def exercise(context, inputs):
    completed = subprocess.run(
        [str(PYTHON), str(PROBE_DIR / "push_cube_probe.py")],
        input=json.dumps(inputs),
        capture_output=True,
        text=True,
        cwd=MS_ROOT,
        env={**os.environ, "MS_ROOT": str(MS_ROOT)},
        timeout=280,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"probe failed ({completed.returncode}): {completed.stderr[-4000:]}")
    context.update(json.loads(completed.stdout.strip().splitlines()[-1]))


def collect(context):
    return context


def cleanup(context):
    pass
