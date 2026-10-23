"""Minimal client for the Open Kernel Challenge HTTP API (https://openkernelchallenge.dev/llms.txt).

The token comes from $OKC_TOKEN or ~/.config/okc/token (never from the repo).

  python3 okc/tools/okc.py check  <problem> <file>   # a check run: tests + timing, off the board
  python3 okc/tools/okc.py submit <problem> <file>   # a scored run
  python3 okc/tools/okc.py status <job>              # wait for a job and save it
  python3 okc/tools/okc.py limits                    # today's runs left per problem

Every finished job is saved to okc/runs/<problem>/<job>.json (and its log to .log).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://openkernelchallenge.dev/api/v1"
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def token():
    t = os.environ.get("OKC_TOKEN")
    if not t:
        with open(os.path.expanduser("~/.config/okc/token")) as f:
            t = f.read().strip()
    return t


def call(method, path, body=None, retries=5):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(API + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {token()}",
                                              "Content-Type": "application/json",
                                              # Cloudflare refuses Python's default agent (error 1010).
                                              "User-Agent": "okc-client/0.1 (python)"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429 and attempt + 1 < retries:
                time.sleep(15 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path}: HTTP {e.code}: {msg[:2000]}")
        except urllib.error.URLError:
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise


def submit(problem, path, mode):
    code = open(path).read()
    job = call("POST", f"/problems/{problem}/submissions", {"code": code, "mode": mode})
    print(json.dumps(job))
    return job


def wait(job_id, poll=10, problem=None):
    while True:
        j = call("GET", f"/jobs/{job_id}")
        if j.get("status") in ("done", "failed", "rejected", "duplicate", "timeout", "error"):
            break
        time.sleep(poll)
    problem = j.get("problem") or j.get("board") or problem or "unknown"
    os.makedirs(os.path.join(RUNS, problem), exist_ok=True)
    base = os.path.join(RUNS, problem, job_id)
    log = j.pop("log_tail", None) or j.get("result", {}).pop("log_tail", None) if isinstance(j.get("result"), dict) else j.pop("log_tail", None)
    with open(base + ".json", "w") as f:
        json.dump(j, f, indent=1)
    if log:
        with open(base + ".log", "w") as f:
            f.write(log)
    return j


def main():
    cmd, *args = sys.argv[1:]
    if cmd in ("check", "submit"):
        problem, path = args
        job = submit(problem, path, "check" if cmd == "check" else "run")
        j = wait(job["job_id"], problem=problem)
        print(json.dumps({k: v for k, v in j.items() if k != "log_tail"}, indent=1)[:6000])
    elif cmd == "status":
        j = wait(args[0])
        print(json.dumps(j, indent=1)[:6000])
    elif cmd == "limits":
        d = call("GET", "/problems")
        for p in d["problems"]:
            print(p["name"], p["limits"])
    elif cmd == "get":
        print(json.dumps(call("GET", args[0]), indent=1)[:20000])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
