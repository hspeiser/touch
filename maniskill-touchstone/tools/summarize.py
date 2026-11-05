"""Print a TouchStone run record (JSON from `touchstone check`/`report`) as a table."""

import json
import sys

run = json.load(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
print(f"run {run['id'][:8]}  disposition={run['disposition']}  state={run['state']}"
      f"  gating={run['gating']}  error={run['error']}")
for x in run["results"]:
    obs = x["observations"] or {}
    print(f"\n[{x['status'].upper():12s}] {x['case_id']}  — {x['title']}")
    if obs:
        shown = {k: v for k, v in obs.items() if k not in ("ms_root",)}
        print("    obs:", json.dumps(shown)[:400])
    for a in x["assertions"]:
        mark = "ok  " if a["passed"] else "FAIL"
        print(f"    {mark} {a['path']} {a['operator']} {a['expected']!r}   actual={a['actual']!r}")
    if x["error"]:
        print("    error:", x["error"][:600])
        ex = x.get("execution") or {}
        if ex.get("stderr"):
            print("    stderr:", ex["stderr"][-600:])
