#!/usr/bin/env bash
# End-to-end TouchStone demo on the drone waypoint reward.
#   1. explore pending cases against the leaky reward
#   2. a human approves each case (content-hash bound)
#   3. gated check against the leaky reward  -> exit 2 (CI would block)
#   4. gated check against the fixed reward  -> exit 0 (CI passes)
set -u
cd "$(dirname "$0")"
export DRONE_PROJECT="$PWD"
REVIEWER="${REVIEWER:-Henry Speiser}"
SUITE=suite
PY=.venv/bin/python

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

step "1. Explore pending cases against DRONE_REWARD=leaky"
DRONE_REWARD=leaky touchstone check $SUITE --experimental --timeout 120 > runs/explore_leaky.json
$PY tools/summarize.py runs/explore_leaky.json

step "2. Approve every case (bound to its content hash)"
touchstone report $SUITE > runs/report.json
$PY tools/case_hashes.py runs/report.json | while read -r id sha; do
  touchstone review $SUITE "$id" --decision approve --reviewer "$REVIEWER" --sha256 "$sha" > /dev/null \
    && echo "approved $id ($sha)"
done

# Steps 3-4 test the reward function; give the policy gate a known-good candidate so it is not inconclusive.
cp policies/ppo_fixed.zip policies/candidate.zip

step "3. Gated check against the LEAKY reward (this is what CI runs)"
DRONE_REWARD=leaky touchstone check $SUITE --timeout 120 > runs/gate_leaky.json; code=$?
$PY tools/summarize.py runs/gate_leaky.json
echo "exit code: $code   (2 = approved assertions failed; CI blocks the merge)"

step "4. Gated check against the FIXED reward"
DRONE_REWARD=fixed touchstone check $SUITE --timeout 120 > runs/gate_fixed.json; code=$?
$PY tools/summarize.py runs/gate_fixed.json
echo "exit code: $code   (0 = every approved assertion passed)"

step "5. Deployment gate: the PPO policy trained on the LEAKY reward"
cp policies/ppo_leaky.zip policies/candidate.zip
DRONE_REWARD=fixed touchstone check $SUITE --timeout 120 > runs/gate_policy_leaky.json; code=$?
$PY tools/summarize.py runs/gate_policy_leaky.json | grep -A6 candidate-policy
echo "exit code: $code   (the policy learned to park under the waypoint; CI blocks it)"

step "6. Deployment gate: the PPO policy trained on the FIXED reward"
cp policies/ppo_fixed.zip policies/candidate.zip
DRONE_REWARD=fixed touchstone check $SUITE --timeout 120 > runs/gate_policy_fixed.json; code=$?
$PY tools/summarize.py runs/gate_policy_fixed.json | grep -A6 candidate-policy
echo "exit code: $code"
