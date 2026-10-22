"""GPU sim harness: the judge's timing protocol, for candidate-vs-SGLang comparisons run
inside a host submission's prepare() on the judge's H200 (bundled with tools/bundle.py).

Protocol (as TASK.md describes the judge):
  - each side is captured as a CUDA graph and replayed;
  - every replay starts with a cold L2 (a buffer larger than L2 is written first);
  - the two sides take turns on the same inputs, in pairs whose order alternates, so which
    side went first cancels;
  - the speedup is the median of paired ratios (baseline / candidate), with a bootstrap 95%
    interval; times are reported net of a measured empty-graph replay cost.

Inputs are generated fresh per case from seeds the scored runs never use, so results speak
to held-out data, not to the scored seeds.
"""
import random
import time

import torch


class SimBench:
    def __init__(self, device="cuda"):
        self.flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device)
        self.overhead_us = 0.0
        dummy = torch.zeros(1, device=device)
        self.overhead_us = self._replay_times(self._capture(lambda: dummy.add_(1)), 21)[10]

    def _capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            fn()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            fn()
        return graph

    def _replay_times(self, graph, reps):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        with torch.cuda.stream(self.stream):
            for i in range(reps):
                self.flush.zero_()
                starts[i].record()
                graph.replay()
                ends[i].record()
        ends[-1].synchronize()
        return sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))

    def compare(self, candidate, baseline, pairs=30):
        """(candidate µs, baseline µs, median paired speedup, (lo, hi) 95% CI), net of
        the replay overhead."""
        ga, gb = self._capture(candidate), self._capture(baseline)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(8 * pairs)]
        order = []
        with torch.cuda.stream(self.stream):
            k = 0
            for i in range(2 * pairs):
                first = (ga, gb) if i % 2 == 0 else (gb, ga)
                for g in first:
                    self.flush.zero_()
                    ev[k].record()
                    g.replay()
                    ev[k + 1].record()
                    k += 2
                order.append(i % 2 == 0)
        ev[k - 1].synchronize()
        ta, tb = [], []
        k = 0
        for a_first in order:
            t1 = ev[k].elapsed_time(ev[k + 1]) * 1e3
            t2 = ev[k + 2].elapsed_time(ev[k + 3]) * 1e3
            k += 4
            a, b = (t1, t2) if a_first else (t2, t1)
            ta.append(max(a - self.overhead_us, 0.05))
            tb.append(max(b - self.overhead_us, 0.05))
        ratios = sorted(b / a for a, b in zip(ta, tb))
        med = ratios[len(ratios) // 2]
        rng = random.Random(0)
        boots = sorted(sorted(rng.choice(ratios) for _ in ratios)[len(ratios) // 2] for _ in range(200))
        del ga, gb
        return sorted(ta)[len(ta) // 2], sorted(tb)[len(tb) // 2], med, (boots[5], boots[194])


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def row_worst(a, b, floor=0.1):
    """Worst row's relative error, rows measured against max(own norm, floor x mean norm)."""
    d = (a.float() - b.float()).reshape(-1, a.shape[-1]).norm(dim=-1)
    n = b.float().reshape(-1, b.shape[-1]).norm(dim=-1)
    return float((d / torch.maximum(n, floor * n.mean()).clamp(min=1e-12)).max())


class SimReport:
    """Collects rows and prints a compact table plus a verdict per group."""

    def __init__(self, title):
        self.title = title
        self.rows = []
        self.t0 = time.time()

    def add(self, group, case, ok, err, bound, cand_us=None, base_us=None, speedup=None, ci=None, note=""):
        self.rows.append(dict(group=group, case=case, ok=ok, err=err, bound=bound, cand=cand_us, base=base_us,
                              speedup=speedup, ci=ci, note=note))
        # Compact: the judge returns only the last 16,000 characters of output.
        s = f"{self.title[-12:]}|{group[0]}|{case}|{'ok' if ok else 'BAD'} {err:.1e}/{bound:.2f}"
        if speedup is not None:
            s += f"|{cand_us:.1f}/{base_us:.1f}us x{speedup:.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
        print(s + (f" {note}" if note else ""), flush=True)

    def summary(self):
        import math
        print(f"sim {self.title}: {len(self.rows)} cases in {time.time() - self.t0:.0f} s")
        for group in dict.fromkeys(r["group"] for r in self.rows):
            rows = [r for r in self.rows if r["group"] == group]
            bad = [r["case"] for r in rows if not r["ok"]]
            sp = [r["speedup"] for r in rows if r["speedup"]]
            if sp:
                geo = math.exp(sum(math.log(x) for x in sp) / len(sp))
                worst = min(rows, key=lambda r: r["speedup"] or 9)
                print(f"sim {group}: {len(rows)} cases, {len(bad)} wrong, speedup geomean {geo:.3f},"
                      f" min {worst['speedup']:.3f} ({worst['case']})")
            else:
                print(f"sim {group}: {len(rows)} cases, {len(bad)} wrong")
            for c in bad[:5]:
                print(f"sim   wrong: {c}")
