"""TouchStone adapter for the drone waypoint environment.

TouchStone copies this file to a temp dir and runs it under its own Python, so
it shells out to the project's venv to fly the scenario. The case input is
passed as JSON on stdin; the evidence JSON that comes back is the observation.

DRONE_PROJECT  path to the drone-touchstone checkout (default ~/drone-touchstone)
DRONE_REWARD   leaky | fixed, the reward function under test (default leaky)
"""

import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone-touchstone"))
PYTHON = PROJECT / ".venv" / "bin" / "python"


def setup(case):
    return {}


def exercise(context, inputs):
    completed = subprocess.run(
        [str(PYTHON), "-m", "drone_env.evaluate"],
        input=json.dumps(inputs),
        capture_output=True,
        text=True,
        cwd=PROJECT,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"evaluate failed ({completed.returncode}): {completed.stderr[-4000:]}")
    context.update(json.loads(completed.stdout))


def collect(context):
    # Ground-truth evidence from the simulator, never the reward's own verdict.
    return context


def cleanup(context):
    pass
