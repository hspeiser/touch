#!/usr/bin/env bash
# TouchStone against the real ManiSkill PushCube-v1 reward, before and after upstream PR #1022.
#   ms-buggy = commit 2563643f (parent of the fix)   ms-fixed = commit dfdc881e (the fix)
set -u
cd "$(dirname "$0")"
export PROBE_DIR="$PWD/probe"
REVIEWER="${REVIEWER:-Henry Speiser}"
SUITE=suite
PY=$HOME/ms-fixed/.venv/bin/python
mkdir -p runs
step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# The deployment-gate case needs a candidate checkpoint in both checkouts; use whatever is there, if anything.
for r in ms-buggy ms-fixed; do
  [ -f "$HOME/$r/runs/candidate.zip" ] || { ls "$HOME/$r"/runs/ppo_seed*.zip >/dev/null 2>&1 && cp "$(ls "$HOME/$r"/runs/ppo_seed*.zip | head -1)" "$HOME/$r/runs/candidate.zip"; }
done

step "1. Explore pending cases against the BUGGY commit"
MS_ROOT=$HOME/ms-buggy touchstone check $SUITE --experimental --timeout 300 > runs/explore_buggy.json
$PY tools/summarize.py runs/explore_buggy.json

step "2. Approve every pending case (bound to content hash)"
touchstone report $SUITE > runs/report.json
$PY tools/case_hashes.py runs/report.json | while read -r id sha; do
  touchstone review $SUITE "$id" --decision approve --reviewer "$REVIEWER" --sha256 "$sha" > /dev/null && echo "approved $id ($sha)"
done

step "3. Gated check at the BUGGY commit 2563643f"
MS_ROOT=$HOME/ms-buggy touchstone check $SUITE --timeout 300 > runs/gate_buggy.json; code=$?
$PY tools/summarize.py runs/gate_buggy.json
echo "exit code: $code"

step "4. Gated check at the FIXED commit dfdc881e (PR #1022)"
MS_ROOT=$HOME/ms-fixed touchstone check $SUITE --timeout 300 > runs/gate_fixed.json; code=$?
$PY tools/summarize.py runs/gate_fixed.json
echo "exit code: $code"
