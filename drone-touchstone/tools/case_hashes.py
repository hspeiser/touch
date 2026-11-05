"""Print `case_id case_sha256` for every case in a `touchstone report` JSON."""
import json, sys
r = json.load(open(sys.argv[1]))
cases = r.get("cases") or r.get("suite", {}).get("cases") or []
for c in cases:
    if c.get("review_status", "pending") != "pending":
        continue
    print(c["id"], c["case_sha256"])
