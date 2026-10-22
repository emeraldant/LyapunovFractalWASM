"""Wait for jobs, save each finished result to okc/runs/<board>/<job>.json, print a summary.

  python3 okc/tools/wait.py <job> [<job> ...]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okc  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
pending = list(sys.argv[1:])
while pending:
    for job in list(pending):
        d = okc.call("GET", f"/jobs/{job}")
        if d.get("status") in ("queued", "received", "running"):
            continue
        os.makedirs(os.path.join(RUNS, d.get("board", "unknown")), exist_ok=True)
        path = os.path.join(RUNS, d.get("board", "unknown"), f"{job}.json")
        json.dump(d, open(path, "w"), indent=1)
        print(f"{job} {d.get('board')} {d.get('status')}", flush=True)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "analyze.py"), path])
        pending.remove(job)
    time.sleep(20)
