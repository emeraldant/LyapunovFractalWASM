"""Summarize a judge result: tests, per-shape timings (raw paired medians, the replay
overhead the judge subtracts, net times, speedup), kernel profiles and the log tail.

  python3 okc/tools/analyze.py <result.json> [--profiles] [--log]
"""
import json
import sys


def main(path, profiles=False, log=False):
    d = json.load(open(path))
    r = d.get("result") or d
    print(f"== {d.get('board')} {d.get('id', '')} mode={d.get('mode')} status={r.get('status')} passed={r.get('passed')}"
          f" speedup={r.get('speedup')} ci={r.get('speedup_ci')} error={str(r.get('error'))[:500]}")
    m = (r.get("machine") or {}).get("gpu_before") or {}
    if m:
        print(f"   gpu {m.get('name')} sm {m.get('clocks.sm')} MHz mem {m.get('clocks.mem')} MHz")
    for rec in r.get("records", []):
        ev = rec["event"]
        case = {k: v for k, v in rec.get("case", {}).items() if k not in ("seed", "weight")}
        if ev == "setup":
            print(f"   setup {rec.get('status')} prepare {rec.get('prepare_s')} s {str(rec.get('error'))[:300]}")
        elif ev == "test":
            if rec["status"] != "pass":
                print(f"   TEST {rec['status']}: {case}: {rec.get('message')}")
        elif ev == "benchmark":
            ov = (rec.get("replay_overhead") or {}).get("overhead_ns", 0)
            ours, base = rec.get("median_raw_ns", 0), rec.get("baseline_raw_ns", 0)
            sol = rec.get("speed_of_light_ns", 0)
            print(f"   {case.get('load', ''):10s} {json.dumps({k: v for k, v in case.items() if k != 'load'})[:80]:80s}"
                  f" ours {(ours - ov) / 1e3:8.1f}  sglang {(base - ov) / 1e3:8.1f}  floor {sol / 1e3:7.1f} us"
                  f"  x{rec.get('speedup', 0):.3f}  ({sol / max(ours - ov, 1):.0%} of judge floor)")
        elif ev == "profile" and profiles:
            print(f"   -- profile {case}\n" + "\n".join("      " + l for l in rec["report"].splitlines()))
    if log:
        print("   -- log tail\n" + (r.get("log_tail") or ""))


if __name__ == "__main__":
    main(sys.argv[1], "--profiles" in sys.argv, "--log" in sys.argv)
