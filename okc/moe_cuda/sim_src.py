# MoE CUDA sim: bundled at the end of the MoE submission's prepare() (tools/make_host.py
# --after). Inputs from the judge's own generate_input (copied from reference.py) on unused
# seeds; correctness against the submission's FP32 reference with the judge's bound; timing
# against SGLang's path and our current path with the judge's protocol.

import random
import time

import torch
import triton
import triton.language as tl

# @bundle muse_moe_w4a16/reference.py D_HIDDEN D_EXPERT N_EXPERTS ROUTED_PER_TOKEN LOCAL POPULARITY generate_input
# @bundle muse_moe_w4a16/nvfp4.py outlier_channels
# @bundle moe_cuda/kernel.py MOE_DEVICE_SRC MOE_HOST_SRC MOE_CUDA_SRC MOE_CUDA_CPP _MOE_MODULES moe_cuda_module _TARGETS _LEFTOVER _bit_source _lane_gather_index repack_expert_weights repack_gemv gemv_layout scales_supported _route_compact_kernel _gather_rows_kernel _splitk_silu_kernel _post_norm_parts_kernel _post_norm_rows_kernel build_cuda_bank ws_layout moe_gemv_stages moe_tiny_stages moe_cuda_stages moe_cuda_forward
# @bundle sim/harness.py SimBench rel_l2 row_worst SimReport

# Every shape the judge can plausibly draw near, not just the scored ones: a kernel that wins
# one shape and regresses the rest must fail here.
MOE_SIM_CASES = (
    [("scored", dict(bs=b)) for b in (1, 8, 16, 32, 128, 256)]
    + [("nearby", dict(bs=b)) for b in (2, 5, 12, 24, 48, 64, 96, 200)]
    + [("nearby", dict(bs=b, dist="skewed")) for b in (8, 32, 256)]
    + [("tests", dict(bs=b)) for b in (3, 33, 257)]
    + [("tests", dict(bs=b, dist="skewed")) for b in (1, 128)])

# Bytes streamed per touched expert: E2M1 codes + E4M3 block-16 scales.
W13_BYTES = 8192 * 4096 // 2 + 8192 * 4096 // 16
W2_BYTES = 4096 * 4096 // 2 + 4096 * 4096 // 16
REGRESSION_CI = 0.98   # a shape fails the gate if the candidate is slower than the plan with CI upper < this


def _plan_for(bs, state):
    plan = ("base",)
    for bk in BUCKETS:
        if bs <= bk:
            return state["plans"].get(bk, plan)
    return plan


def run_moe_cuda_sim(weights, state, variants=None, budget_s=300.0, seed=9_100_000, stages_for=None):
    """Gate for candidate MoE paths. Per case and variant:
      correct  - finite, rel L2 and worst-row error within the judge's bounds vs FP32 reference;
      stable   - a second run is bitwise identical (async pipelines: catches races);
      speed    - paired, cold-L2 graph replays vs the CURRENT submission's plan (not SGLang),
                 so per-shape regressions show up, not just a geomean.
    Prints one GATE line per variant: PASS only with 0 wrong, 0 unstable, 0 regressions."""
    t0 = time.time()
    try:
        moe_cuda_module()
    except Exception as e:
        print(f"moe cuda: compile failed: {type(e).__name__}: {str(e)[-1200:]}")
        return
    print(f"moe cuda: compiled in {time.time() - t0:.0f} s")
    bank = build_cuda_bank(weights)
    if bank is None:
        print("moe cuda: scales unsupported, skipped")
        return
    ws_layout(bank)
    torch.cuda.synchronize()
    bench = SimBench()
    variants = variants or {"ws k1": dict(stages=4, ctas_per_sm=4, mt=1, ws=True),
                            "ws k2s4c2": dict(stages=4, ctas_per_sm=2, mt=1, ws=True, ks=2),
                            "ws k2s8c1": dict(stages=8, ctas_per_sm=1, mt=1, ws=True, ks=2)}
    stages_for = stages_for or "ws k2s4c2"
    gate = {k: dict(wrong=[], unstable=[], regress=[], ratios={}) for k in variants}
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    for i, (group, case) in enumerate(MOE_SIM_CASES):
        if time.time() - t0 > budget_s:
            print(f"moe cuda: budget reached at {case}; gate incomplete")
            for g in gate.values():
                g["wrong"].append("budget")
            break
        data = generate_input(sharding="ep8", seed=seed + i, device="cuda", **case)
        bs = case["bs"]
        name = f"{group[0]} bs{bs}{' skew' if case.get('dist') == 'skewed' else ''}"
        plan = _plan_for(bs, state)
        ref = _reference(data, weights)
        _, _, p_sg, _ = bench.compare(lambda: _run(plan, data, state), lambda: _baseline(data, state), pairs=9)
        for vname, kw in variants.items():
            g = gate[vname]
            kw = dict(kw)
            mt_v = kw.pop("mt", 1)

            def ours(kw=kw, mt_v=mt_v):
                return moe_cuda_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"], ls,
                                        bank, state["norm_w"], POST_NORM_EPS, mt=mt_v, **kw)
            try:
                out = ours()
                again = ours()
                torch.cuda.synchronize()
            except Exception as e:
                g["wrong"].append(name)
                print(f"{vname}|{name}|ERROR {type(e).__name__}: {str(e)[:200]}")
                continue
            err, worst = rel_l2(out, ref), row_worst(out, ref)
            ok = bool(torch.isfinite(out.float()).all()) and err <= 1.5e-2 and worst <= 7.5e-2
            stable = torch.equal(out, again)
            if not ok:
                g["wrong"].append(name)
            if not stable:
                g["unstable"].append(name)
            a, b, sp, ci = bench.compare(ours, lambda: _run(plan, data, state), pairs=15)
            g["ratios"][name] = sp * p_sg
            if ci[1] < REGRESSION_CI:
                g["regress"].append(f"{name} x{sp:.2f}")
            print(f"{vname}|{name}|{'ok' if ok else 'BAD'} {err:.1e} w{worst:.1e}{'' if stable else ' UNSTABLE'}"
                  f"|{a:.1f} vs plan {b:.1f}us x{sp:.3f} [{ci[0]:.2f},{ci[1]:.2f}] vs sglang x{sp * p_sg:.2f}",
                  flush=True)
            if group == "scored" and vname == stages_for:
                fns, _ = moe_cuda_stages(data["hidden_states"], data["topk_ids"], data["topk_weights"], ls,
                                         bank, state["norm_w"], POST_NORM_EPS, mt=mt_v, **kw)
                for fn in fns:
                    fn()
                ts = [bench.compare(fn, fns[0], pairs=6)[0] for fn in fns]
                ne = int(torch.unique(data["topk_ids"][data["topk_ids"] < 32]).numel())
                cnt = torch.bincount(data["topk_ids"][data["topk_ids"] < 32].flatten().long(), minlength=32)
                chunks = int(((cnt + 15 * mt_v) // (16 * mt_v)).sum())   # weight streams per GEMM
                d1, d2 = ne * W13_BYTES / 4.06e6, ne * W2_BYTES / 4.06e6
                print(f"  stages bs{bs}: route {ts[0]:.1f} gemm1 {ts[1]:.1f} ({100 * d1 / ts[1]:.0f}% of {d1:.0f})"
                      f" gemm2 {ts[2]:.1f} ({100 * d2 / ts[2]:.0f}% of {d2:.0f}) post {ts[3]:.1f} us, {ne} experts, {chunks} chunks (x{chunks / ne:.2f} weight reads), max {int(cnt.max())} rows/expert")
    import math
    for vname, g in gate.items():
        sc = [v for k, v in g["ratios"].items() if k.startswith("s ")]
        nb = [v for k, v in g["ratios"].items() if not k.startswith("s ")]
        geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
        verdict = "PASS" if not (g["wrong"] or g["unstable"] or g["regress"]) else "FAIL"
        print(f"GATE {vname}: {verdict} | vs sglang scored {geo(sc):.3f} other {geo(nb):.3f} | wrong {g['wrong']}"
              f" unstable {g['unstable']} regress {g['regress']}")


def _prof_report(tag, buf, n_items, bytes_total, grid, clk_ghz=1.98):
    """One line from a profiling-build launch: where the GEMM's time went."""
    r = buf[: n_items * 12].view(n_items, 12).cpu().double()
    sm, t0, t1, cw, pw, cyc, p0 = (r[:, i] for i in range(7))
    t_dec, t_iss = r[:, 8], r[:, 9]
    start, end = float(torch.minimum(t0, p0).min()), float(t1.max())
    span = end - start                                                  # ns
    dur = t1 - t0
    busy = float(dur.sum()) / (span * grid)                             # item-time / (span x CTA slots)
    done = torch.sort(t1).values
    t90 = float(done[int(0.9 * (n_items - 1))]) - start
    cons_wait = float(cw.sum() / cyc.sum())                             # consumer warp 0 starved for data
    prod_wait = float(pw.sum() / cyc.sum())                             # producer blocked on free stages
    per_sm = torch.bincount(sm.long(), minlength=132)
    t_mma = r[:, 7]
    ok = (dur > 0) & (cyc > 0)
    clk = float(cyc[ok].sum() / dur[ok].sum()) if bool(ok.any()) else 0.0     # SM GHz: clock64 / globaltimer
    st = torch.sort(torch.minimum(t0, p0) - start).values / 1e3                # item start times, us
    starts = (f" starts {float(st[int(0.5 * (n_items - 1))]):.1f}/{float(st[int(0.9 * (n_items - 1))]):.1f}"
              f"/{float(st[-1]):.1f} us (p50/p90/max);")
    waits = ""
    if float(t_mma.sum()) > 0 or float(cw.sum()) > 0:   # timed build: warp 0's split of its item time
        waits = (f" warp time: data wait {100 * float(cw.sum() / cyc.sum()):.0f}%, tensor wait"
                 f" {100 * float(t_mma.sum() / cyc.sum()):.0f}%, rest {100 * float(1 - (cw.sum() + t_mma.sum()) / cyc.sum()):.0f}%")
        if float(t_dec.sum()) > 0:
            waits += (f" (decode {100 * float(t_dec.sum() / cyc.sum()):.0f}%, fence+issue"
                      f" {100 * float(t_iss.sum() / cyc.sum()):.0f}%)")
        waits += ";"
    print(f"  prof {tag}: {n_items} items on {int((per_sm > 0).sum())} SMs, span {span / 1e3:.1f} us,"
          f" {bytes_total / span / 1e3:.2f} TB/s; slot use {100 * busy:.0f}%; 90% done at {t90 / 1e3:.1f} us"
          f" (tail {(span - t90) / 1e3:.1f});{starts} SM clock {clk:.3f} GHz; item {float(dur.median()) / 1e3:.1f} us med,"
          f" {float(dur.quantile(0.9)) / 1e3:.1f} p90;{waits} producer waits {100 * prod_wait:.0f}%; items/SM {int(per_sm.max())} max {float(per_sm[per_sm > 0].float().mean()):.1f} mean",
          flush=True)


ABLATIONS = {0: "full", 1: "no math", 2: "no act rows", 3: "weights only", 4: "math only",
             12: "decode only", 20: "mma only", 28: "loop only", 32: "timed", 36: "timed math", 96: "timed fine"}


def run_moe_profile(weights, state, cfgs=None, seed=9_200_000, sizes=(16, 128, 256), abls=(0, 1, 2, 3, 4)):
    """Timeline of each GEMM (profiling build, cold L2) on scored shapes, and the same
    pipeline with parts switched off (ABLATIONS): which part closes the gap to the roofline."""
    try:
        moe_cuda_module(prof=True)
    except Exception as e:
        print(f"moe prof: compile failed: {str(e)[-800:]}")
        return
    bank = state.get("cuda_bank")
    if bank is None:
        bank = build_cuda_bank(weights)
        ws_layout(bank)
    cfgs = cfgs or {"k1": dict(stages=4, ctas_per_sm=4, mt=1, ws=True, ks=1)}
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    for i, bs in enumerate(sizes):
        data = generate_input(bs=bs, sharding="ep8", seed=seed + i, device="cuda")
        ids = data["topk_ids"]
        ne = int(torch.unique(ids[ids < 32]).numel())
        for name, kw in cfgs.items():
            for abl in abls:
                maxi = (32 + bs * 8 // ((8 if kw.get("sab") else 16) * kw["mt"]) + 1) * 64 * max(kw.get("kparts", 1), kw.get("kparts2") or 1)
                pcta = 2048   # v6 per-CTA entry / exit slots (PCTA) before the item records
                prof_full = (torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"),
                             torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"))
                prof = (prof_full[0][pcta:], prof_full[1][pcta:])
                fns, _ = moe_cuda_stages(data["hidden_states"], ids, data["topk_weights"], ls, bank, state["norm_w"],
                                         _SIM_EPS, prof=prof, abl=abl, **kw)
                try:
                    for fn in fns:
                        fn()
                except Exception as e:
                    print(f"  prof {name} abl {abl}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                fns[0]()
                torch.cuda.synchronize()
                for g, (fn, nbytes) in enumerate(((fns[1], ne * W13_BYTES), (fns[2], ne * W2_BYTES))):
                    prof_full[g].zero_()
                    flush.zero_()
                    fn()
                    torch.cuda.synchronize()
                    if kw.get("sab"):   # CTA timeline: entry -> first producer issue -> first data -> exits
                        cta = prof_full[g][:pcta].view(-1, 2).cpu().double()
                        cta = cta[cta[:, 0] > 0]
                        if len(cta):
                            it = prof[g].view(-1, 12).cpu().double()
                            it = it[it[:, 2] > 0]
                            e0 = float(cta[:, 0].min())
                            q = lambda v: "/".join(f"{(float(v.quantile(x)) - e0) / 1e3:.1f}" for x in (0.0, 0.5, 1.0))
                            print(f"  cta bs{bs} {name} abl {abl} gemm{g + 1}: {len(cta)} CTAs; entry {q(cta[:, 0])};"
                                  f" producer first issue {q(it[:, 6])}; consumer first data {q(it[:, 1])};"
                                  f" last item end {(float(it[:, 2].max()) - e0) / 1e3:.1f}; exit {q(cta[:, 1])} us (min/p50/max)")
                    n_items = int((prof[g].view(-1, 12)[:, 2] > 0).sum())
                    grid = 132 * (kw.get("ctas_per_sm", 4) if kw.get("ks", 1) == 1 else 1)
                    _prof_report(f"bs{bs} {name} {ABLATIONS[abl]:>12} gemm{g + 1} ({ne} exp)", prof[g], n_items,
                                 nbytes, grid)


_SIM_EPS = globals().get("POST_NORM_EPS", 1e-8)   # the MoE host defines POST_NORM_EPS; other hosts do not


@triton.jit
def _stamp_kernel(buf_ptr, IDX: tl.constexpr):
    tl.store(buf_ptr + IDX, tl.extra.cuda.globaltimer())


def _launch_probe(bank, state, cfgs, bs, seed, reps=7):
    """Where a GEMM stage's time goes around its CTAs: [stamp] -> GEMM -> [stamp] in one CUDA graph
    after a cold-L2 flush, with the profiling build's per-CTA entry / exit stamps. Also an empty
    GEMM (no local routes) and a stamp-only graph for the bare boundary cost."""
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    stamps = torch.zeros(4, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    data = generate_input(bs=bs, sharding="ep8", seed=seed, device="cuda")
    empty_ids = torch.full_like(data["topk_ids"], 100)   # no local expert: zero items
    pcta = 2048
    med = lambda v: sorted(v)[len(v) // 2]

    def timeline(fn, pfull, tag):
        fn()   # warm (compile, module load)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                _stamp_kernel[(1,)](stamps, 0)
                fn()
                _stamp_kernel[(1,)](stamps, 1)
        rows = []
        for _ in range(reps):
            if pfull is not None:
                pfull.zero_()
            flush.zero_()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            st = stamps.cpu().double()
            r = [float(st[1] - st[0]) / 1e3]
            if pfull is not None:
                cta = pfull[:pcta].view(-1, 2).cpu().double()
                cta = cta[cta[:, 0] > 0]
                r += [float(cta[:, 0].min() - st[0]) / 1e3, float(cta[:, 1].max() - cta[:, 0].min()) / 1e3,
                      float(st[1] - cta[:, 1].max()) / 1e3]
            rows.append(r)
        cols = list(zip(*rows))
        if pfull is None:
            print(f"  launch bs{bs} {tag}: stamp -> stamp {med(cols[0]):.1f} us", flush=True)
        else:
            print(f"  launch bs{bs} {tag}: stamp -> stamp {med(cols[0]):.1f} us = stamp -> first CTA entry "
                  f"{med(cols[1]):.1f} + CTA life {med(cols[2]):.1f} + last exit -> stamp {med(cols[3]):.1f}", flush=True)

    timeline(lambda: None, None, "no kernel")
    for name, kw in cfgs.items():
        if not kw.get("sab"):
            continue
        for ids, label in ((data["topk_ids"], ""), (empty_ids, " (no items)")):
            maxi = (32 + bs * 8 // (8 * kw["mt"]) + 1) * 64 * max(kw.get("kparts", 1), kw.get("kparts2") or 1)
            pf = (torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"),
                  torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"))
            fns, _ = moe_cuda_stages(data["hidden_states"], ids, data["topk_weights"], ls, bank, state["norm_w"],
                                     _SIM_EPS, prof=(pf[0][pcta:], pf[1][pcta:]), abl=0, **kw)
            for fn in fns:
                fn()
            torch.cuda.synchronize()
            for gi in (0, 1):
                timeline(fns[1 + gi], pf[gi], f"{name} gemm{gi + 1}{label}")
            timeline(fns[0], None, f"{name} route{label}")


def _graph_abl(bank, state, cfgs, bs, seed, abls):
    """Our GEMMs under the judge's timing (graph replay after the cold-L2 flush, overhead
    subtracted), full and with parts switched off (profiling build): which part of the pipeline
    keeps a GEMM above the pure-streaming floor at the same bytes (run_bw_judge)."""
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    bench = SimBench()
    tiny = torch.zeros(1, device="cuda")
    data = generate_input(bs=bs, sharding="ep8", seed=seed, device="cuda")
    ids = data["topk_ids"]
    ne = int(torch.unique(ids[ids < 32]).numel())
    pcta = 2048
    for name, kw in cfgs.items():
        line = []
        for abl in abls:
            maxi = (32 + bs * 8 // (8 * kw["mt"]) + 1) * 64 * max(kw.get("kparts", 1), kw.get("kparts2") or 1)
            pf = (torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"),
                  torch.zeros(pcta + maxi * 12, dtype=torch.int64, device="cuda"))
            try:
                fns, _ = moe_cuda_stages(data["hidden_states"], ids, data["topk_weights"], ls, bank, state["norm_w"],
                                         _SIM_EPS, prof=(pf[0][pcta:], pf[1][pcta:]), abl=abl, **kw)
                for fn in fns:
                    fn()
                torch.cuda.synchronize()
                g1 = bench.compare(fns[1], lambda: tiny.add_(1), pairs=8)[0]
                g2 = bench.compare(fns[2], lambda: tiny.add_(1), pairs=8)[0]
                line.append(f"{ABLATIONS[abl]} {g1:.1f} / {g2:.1f}")
            except Exception as e:
                line.append(f"{ABLATIONS[abl]} failed ({type(e).__name__}: {str(e)[:80]})")
        print(f"graph abl bs{bs} {name} ({ne} exp, GEMM1 {ne * 18.9:.0f} MB / GEMM2 {ne * 9.4:.0f} MB) gemm1 / gemm2 us: "
              + "; ".join(line), flush=True)


def run_moe_profile_synth(cfgs, sizes=(16, 256), abls=(0,), seed=7, timed=True, stage_detail=False, seed_base=8_800_000,
                          launch=False, graph_abls=()):
    """run_moe_profile on random NVFP4 weights for the 32 local experts: runs inside ANY
    problem's host (the kernels' speed does not depend on the weight values), so kernel
    experiments do not spend the MoE problem's checks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    E, I, H = 32, 4096, 4096

    def scales(n, k):   # positive normal E4M3 in [2^-3, 2^2): inside the fast decode's range
        e = torch.randint(4, 9, (n, k), generator=g, device="cuda", dtype=torch.int32)
        m = torch.randint(0, 8, (n, k), generator=g, device="cuda", dtype=torch.int32)
        return ((e << 3) | m).to(torch.uint8).view(torch.float8_e4m3fn)

    weights = {
        "w13": torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, device="cuda", dtype=torch.uint8),
        "w2": torch.randint(0, 256, (E, H, I // 2), generator=g, device="cuda", dtype=torch.uint8),
        "w13_gs": torch.full((E,), 1e-3, device="cuda"), "w2_gs": torch.full((E,), 1e-3, device="cuda"),
    }
    weights["w13_sf"] = torch.stack([scales(2 * I, H // 16) for _ in range(E)])
    weights["w2_sf"] = torch.stack([scales(H, I // 16) for _ in range(E)])
    state = {"norm_w": torch.ones(H, dtype=torch.bfloat16, device="cuda")}
    if abls:
        run_moe_profile(weights, state, cfgs=cfgs, sizes=sizes, abls=abls)
    # Correctness without the MoE host: every config's forward output against the gate-verified
    # v6 configuration (s2) on the same routing, and bitwise repeatability.
    bank = state.get("cuda_bank")
    if bank is None:
        bank = build_cuda_bank(weights)
        ws_layout(bank)
        gemv_layout(bank, weights)
    if graph_abls:
        try:
            moe_cuda_module(prof=True)
            for i, bs in enumerate(sizes):
                _graph_abl(bank, state, cfgs, bs, seed_base + i, graph_abls)
        except Exception as e:
            import traceback
            print("graph abl failed:", traceback.format_exc()[-1500:], flush=True)
    if launch:
        try:
            moe_cuda_module(prof=True)
            for i, bs in enumerate(sizes):
                _launch_probe(bank, state, cfgs, bs, seed_base + i)
        except Exception as e:
            import traceback
            print("launch probe failed:", traceback.format_exc()[-1500:], flush=True)
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    bench = SimBench() if timed else None
    for i, bs in enumerate(sizes):
        data = generate_input(bs=bs, sharding="ep8", seed=seed_base + i, device="cuda")
        args = (data["hidden_states"], data["topk_ids"], data["topk_weights"], ls, bank, state["norm_w"], _SIM_EPS)
        ref_kw = dict(mt=2, stages=4, ctas_per_sm=4, sab=True)
        ref = moe_cuda_forward(*args, **ref_kw)
        ids = data["topk_ids"].flatten()
        per_exp = torch.bincount(ids[ids < 32].long(), minlength=32).tolist()
        print(f"routing bs{bs} seed {seed_base + i}: tokens per local expert {sorted((c for c in per_exp if c), reverse=True)}",
              flush=True)
        for name, kw in cfgs.items():
            kw = {k: v for k, v in kw.items() if k != "abl"}
            out = moe_cuda_forward(*args, **kw)
            again = moe_cuda_forward(*args, **kw)
            torch.cuda.synchronize()
            line = (f"check {name} bs{bs}: rel diff vs v6 {rel_l2(out, ref):.1e}, row worst {row_worst(out, ref):.1e},"
                    f" repeat {'equal' if torch.equal(out, again) else 'DIFFERENT'}")
            if bench is not None:   # whole forward, CUDA graph, cold L2, paired against v6 s2
                a_us, b_us, sp, ci = bench.compare(lambda kw=kw: moe_cuda_forward(*args, **kw),
                                                   lambda: moe_cuda_forward(*args, **ref_kw), pairs=9)
                line += f"; {a_us:.1f} us (v6 s2 {b_us:.1f}, x{sp:.3f})"
                if stage_detail:   # each stage as its own graph (cold L2), vs the route stage
                    fns, _ = moe_cuda_stages(*args, **kw)
                    for fn in fns:
                        fn()
                    ts = [bench.compare(fn, fns[0], pairs=6)[0] for fn in fns]
                    line += " | stages " + " ".join(f"{t:.1f}" for t in ts)
            print(line, flush=True)

    del weights
    torch.cuda.empty_cache()


def run_tiny_stamps(sizes=(1,), seeds=(8_950_000, 8_970_000, 8_990_000), rings=(4, 8)):
    """moe_tiny_kernel phase timeline (per-CTA globaltimer stamps) under graph replay after the
    cold-L2 flush, on random NVFP4 weights."""
    g = torch.Generator(device="cuda").manual_seed(7)
    E, I, H = 32, 4096, 4096
    def scales(n, k):
        e = torch.randint(4, 9, (n, k), generator=g, device="cuda", dtype=torch.int32)
        m = torch.randint(0, 8, (n, k), generator=g, device="cuda", dtype=torch.int32)
        return ((e << 3) | m).to(torch.uint8).view(torch.float8_e4m3fn)
    weights = {"w13": torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, device="cuda", dtype=torch.uint8),
               "w2": torch.randint(0, 256, (E, H, I // 2), generator=g, device="cuda", dtype=torch.uint8),
               "w13_gs": torch.full((E,), 1e-3, device="cuda"), "w2_gs": torch.full((E,), 1e-3, device="cuda")}
    weights["w13_sf"] = torch.stack([scales(2 * I, H // 16) for _ in range(E)])
    weights["w2_sf"] = torch.stack([scales(H, I // 16) for _ in range(E)])
    bank = build_cuda_bank(weights)
    ws_layout(bank)
    norm_w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    for bs in sizes:
        for seed in seeds:
            data = generate_input(bs=bs, sharding="ep8", seed=seed, device="cuda")
            for ring in rings:
                fns, _ = moe_cuda_stages(data["hidden_states"], data["topk_ids"], data["topk_weights"], ls, bank, norm_w,
                                         _SIM_EPS, mt=1, tiny=ring, chain=True, _tiny_stamps=True)
                for fn in fns:
                    fn()
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr, stream=stream):
                        for fn in fns:
                            fn()
                rows = []
                for _ in range(5):
                    bank["tiny_cnt"][4096:].zero_()
                    flush.zero_()
                    torch.cuda.synchronize()
                    gr.replay()
                    torch.cuda.synchronize()
                    st = bank["tiny_cnt"][4096:].view(torch.int64).view(-1, 4).cpu().double()
                    st = st[st[:, 0] > 0]
                    t0 = float(st[:, 0].min())
                    rows.append([(float(st[:, j].quantile(q)) - t0) / 1e3 for j in (0, 1, 2, 3) for q in (0.5, 1.0)])
                med = [sorted(c)[2] for c in zip(*rows)]
                ids = data["topk_ids"]
                ne = int(torch.unique(ids[ids < 32]).numel())
                print(f"tiny stamps bs{bs} seed {seed} ring {ring} ({ne} exp): start p50/max {med[0]:.1f}/{med[1]:.1f};"
                      f" GEMM1 done {med[2]:.1f}/{med[3]:.1f}; GEMM2 wait passed {med[4]:.1f}/{med[5]:.1f};"
                      f" GEMM2 done {med[6]:.1f}/{med[7]:.1f} us", flush=True)


def run_soak(cfgs, sizes=(1,), reps=2000, seed=8_960_000):
    """Replay each config's forward as one CUDA graph `reps` times (cold-L2 flush before each, as the
    judge) on fixed inputs; every output must equal the first bitwise (catches rare races: Guild #42)."""
    g = torch.Generator(device="cuda").manual_seed(7)
    E, I, H = 32, 4096, 4096

    def scales(n, k):
        e = torch.randint(4, 9, (n, k), generator=g, device="cuda", dtype=torch.int32)
        m = torch.randint(0, 8, (n, k), generator=g, device="cuda", dtype=torch.int32)
        return ((e << 3) | m).to(torch.uint8).view(torch.float8_e4m3fn)

    weights = {"w13": torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, device="cuda", dtype=torch.uint8),
               "w2": torch.randint(0, 256, (E, H, I // 2), generator=g, device="cuda", dtype=torch.uint8),
               "w13_gs": torch.full((E,), 1e-3, device="cuda"), "w2_gs": torch.full((E,), 1e-3, device="cuda")}
    weights["w13_sf"] = torch.stack([scales(2 * I, H // 16) for _ in range(E)])
    weights["w2_sf"] = torch.stack([scales(H, I // 16) for _ in range(E)])
    bank = build_cuda_bank(weights)
    ws_layout(bank)
    gemv_layout(bank, weights)
    norm_w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    for i, bs in enumerate(sizes):
        # Two different inputs alternate through the captured graph's input buffers (fresh inputs
        # every call, as the judge): a stale partial from the previous replay then shows as a mismatch.
        dA = generate_input(bs=bs, sharding="ep8", seed=seed + i, device="cuda")
        dB = generate_input(bs=bs, sharding="ep8", seed=seed + 500 + i, device="cuda")
        xs, ids_s, tw_s = (dA["hidden_states"].clone(), dA["topk_ids"].clone(), dA["topk_weights"].clone())
        for name, kw in cfgs.items():
            refs = []
            for d in (dA, dB):
                refs.append(moe_cuda_forward(d["hidden_states"], d["topk_ids"], d["topk_weights"], ls, bank, norm_w, _SIM_EPS, **kw))
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    res = moe_cuda_forward(xs, ids_s, tw_s, ls, bank, norm_w, _SIM_EPS, **kw)
            bad = 0
            bad_by = [0, 0]
            worst = 0.0
            for r in range(reps):
                d = (dA, dB)[r % 2]
                with torch.cuda.stream(stream):
                    xs.copy_(d["hidden_states"]); ids_s.copy_(d["topk_ids"]); tw_s.copy_(d["topk_weights"])
                    flush.zero_()
                    graph.replay()
                stream.synchronize()
                ref = refs[r % 2]
                if not torch.equal(res, ref):
                    bad_by[r % 2] += 1
                    bad += 1
                    worst = max(worst, float((res.float() - ref.float()).norm() / ref.float().norm()))
            na = [int(torch.unique(d["topk_ids"][d["topk_ids"] < 32]).numel()) for d in (dA, dB)]
            print(f"soak {name} bs{bs}: {reps} replays alternating two inputs ({na[0]} / {na[1]} local experts),"
                  f" {bad} differ from eager (worst rel {worst:.1e}); by input {bad_by}", flush=True)
            del graph
    del weights, bank
    torch.cuda.empty_cache()
