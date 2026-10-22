"""Predictive model of our CUDA MoE pipeline (moe_cuda/kernel.py) on an H200.

Time for one call = route + GEMM1 + GEMM2 + post-norm + fixed overhead, where each GEMM is
simulated item by item:

  - work: per active expert, n_blocks x ceil(T_e / (MT * 16)) items; an item streams 32 chunks;
  - per chunk, a CTA is limited by
      issue: its warps' SASS clocks per iteration (tools/sass.py), spread over 4 sub-partitions,
             times the number of CTAs sharing the SM;
      latency: (STAGES - 1) chunks in flight per CTA -> chunk_bytes / ((STAGES-1) * chunk_bytes / L);
  - items are dealt to a persistent grid of SMS * CPS CTAs, CPS per SM; makespan = slowest SM;
  - the whole GEMM is also bounded by total bytes / measured bandwidth (4.06 TB/s).

Calibrated (L, fixed overhead) on measured v1 and v2 sim runs; then used to choose v3's
configuration. Routing per batch size is sampled from the judge's generator.

  python3 okc/tools/moe_model.py
"""
import math
import os
import random
import sys

SMS, CLK = 132, 1.98e9
BW = 4.06e12
OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# SASS clocks per warp-iteration (one 128-k chunk), from tools/sass.py: max over issue and
# per-pipe limits, for the 1-tile instantiation, plus the tensor cost per extra tile.
#   (version, mode): (clocks for 1 tile, extra clocks per extra tile, weight+scale bytes per warp-chunk, warps' groups)
SASS = {
    ("v2", 1): (382, 72, 2 * 576), ("v2", 2): (193, 48, 576),
    ("v3", 1): (278, 128, 2 * 576), ("v3", 2): (150, 64, 576),
}
# v1 (K-split, register prefetch): per item 32 output rows x full K, 8 warps x 4 chunks each;
# modelled as one stage of 8 warps x 4 groups x 24 B x 32 lanes in flight.


def routing_samples(bs, n=24, seed=0):
    sys.path.insert(0, os.path.join(OKC, "muse_moe_w4a16"))
    import importlib
    R = importlib.import_module("reference")
    out = []
    for i in range(n):
        d = R.generate_input(bs, "ep8", 50_000 + seed + i, device="cpu")
        ids = d["topk_ids"].flatten()
        loc = ids[ids < 32]
        out.append([int((loc == e).sum()) for e in range(32) if int((loc == e).sum()) > 0])
    return out


def gemm_time(counts, version, mode, mt, stages, cps, L):
    clk1, clk_tile, wbytes = SASS[(version, mode)]
    chunk_bytes = 8 * wbytes                      # per CTA per chunk (weights + scales)
    items = []
    for t in counts:
        for c in range(math.ceil(t / (mt * 16))):
            tiles = min(mt, math.ceil((t - c * mt * 16) / 16))
            items.append(clk1 + (tiles - 1) * clk_tile)
    items = items * 64                            # n_blocks = 64 for both GEMMs
    ctas = SMS * cps
    per_cta = [[] for _ in range(ctas)]
    for i, it in enumerate(items):
        per_cta[i % ctas].append(it)
    sm_time = 0.0
    for sm in range(SMS):
        mine = [per_cta[sm + k * SMS] for k in range(cps)]
        # CTAs resident together share the SM's issue; time-step through their item lists.
        t_sm = 0.0
        queues = [list(q) for q in mine]
        while any(queues):
            active = [q for q in queues if q]
            share = len(active)
            # each active CTA runs its head item; the shortest finishes first
            durs = []
            for q in active:
                issue = 32 * (8 / 4) * q[0] * share / CLK          # 8 warps over 4 sub-partitions
                lat = 32 * L / (stages - 1)                         # (S-1) chunks in flight
                durs.append(max(issue, lat) + (stages - 1) * L / 4)  # pipeline fill
            d = min(durs)
            t_sm += d
            for q, dd in zip(active, durs):
                if dd - d <= 1e-12:
                    q.pop(0)
                else:
                    q[0] *= (dd - d) / dd
        sm_time = max(sm_time, t_sm)
    total_bytes = sum(len(items) for _ in [0]) * 32 * chunk_bytes
    return max(sm_time, total_bytes / BW)


def v1_gemm_time(counts, mode, L):
    # v1: items of 32 rows (GEMM1: 16 gate/up pairs) x full K; 264 CTAs; per item 4 chunks per
    # warp with one chunk prefetched: latency-bound at ~ 4 * L per item, plus INT issue.
    groups_per_item = 4
    n_items = sum(1 for t in counts for _ in range(math.ceil(t / 16))) * (1024 if mode == 1 else 512) // groups_per_item
    waves = math.ceil(n_items / (SMS * 2))
    return waves * 4 * L * 1.0 + 0.0


def predict(bs_list, version, cfg, L, overhead, samples):
    res = {}
    for bs in bs_list:
        times = []
        for counts in samples[bs]:
            if version == "v1":
                g = v1_gemm_time(counts, 1, L) + v1_gemm_time(counts, 2, L)
            else:
                mt = cfg["mt"](bs)
                g = (gemm_time(counts, version, 1, mt, cfg["stages"], cfg["cps"], L)
                     + gemm_time(counts, version, 2, mt, cfg["stages"], cfg["cps"], L))
            times.append(g + overhead)
        res[bs] = sum(times) / len(times) * 1e6
    return res


MEASURED = {  # µs, sim runs 5cc17464 (v1) and 0cf6af63 (v2, 2 CTAs/SM)
    "v1": {1: 45.1, 16: 230.0, 32: 310.4, 8: 113.6, 128: 931.2, 256: 1956.6},
    "v2": {1: 70.6, 16: 216.8, 32: 285.0, 8: 130.5, 128: 1067.4, 256: 2020.4},
}


def mt_for(bs):
    return 1 if bs <= 16 else 2 if bs <= 32 else 4


if __name__ == "__main__":
    bs_list = (1, 8, 16, 32, 128, 256)
    samples = {bs: routing_samples(bs, n=12) for bs in bs_list}
    best = None
    for L in [x * 0.1e-6 for x in range(5, 40)]:
        for ov in [x * 1e-6 for x in range(4, 30, 2)]:
            p2 = predict(bs_list, "v2", {"mt": mt_for, "stages": 4, "cps": 2}, L, ov, samples)
            err = sum(math.log(p2[b] / MEASURED["v2"][b]) ** 2 for b in bs_list)
            if best is None or err < best[0]:
                best = (err, L, ov, p2)
    err, L, ov, p2 = best
    print(f"calibrated on v2: L = {L * 1e6:.2f} us, fixed overhead = {ov * 1e6:.0f} us, rms log err {math.sqrt(err / len(bs_list)):.2f}")
    for b in bs_list:
        print(f"  bs {b:3d}: v2 model {p2[b]:7.1f} us, measured {MEASURED['v2'][b]:7.1f}")
