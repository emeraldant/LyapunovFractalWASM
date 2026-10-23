# Attention sim: bundled into a host submission (tools/make_host.py). Generates inputs exactly
# as the judge does (its generate_input and reference math, copied from
# muse_attention_decode/reference.py), on seeds the scored runs never use, and checks and
# times attention variants against SGLang's path.

import math
import random
import time

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

# @bundle muse_attention_decode/reference.py HEAD_DIM HEADS SLIDING_WINDOW ROPE_THETA QK_SCALE_FACTOR SM_SCALE RMS_EPS PAGE FP8 K_SCALE REL_L2_TOL FP8_QUERY_FACTOR FP8_QUERY_FLOOR FP8_QUERY_SLACK QUERY_DTYPE CACHE_TOL SOFT_C2 COHERENT_HEAD COHERENT_ALPHA _coherent_tail AGENTIC_SPREAD _longest _lengths _quant _dequant CACHE_CHUNK _fill_cache _fa3_scheduler_metadata generate_input split_qkv _rms _rope _rows new_slots ref_kernel

# @bundle muse_attention_decode/submission.py LOG2E NUM_SMS _INV_FREQ _inv_freq _load_heads _attn_split_kernel _norm_gate_store _attn_combine_kernel _plan _fused_attention _attn3_step _attn3_new_keys _attn3_split_kernel _attn3_combine_kernel _attn3 _NORMS _ROPE ROPE_POSITIONS _qk_norm _cos_sin _sglang_attention

# @bundle sim/harness.py SimBench rel_l2 row_worst SimReport

ATTN_SCORED = [
    dict(batch=b, ctx=80000, heads="tp8", layer=layer, page=1, q_len=L,
         dist="soft" if layer == "sliding" else "coherent", **({} if b == 1 else {"spread": "agentic"}))
    for L in (1, 8) for b in (1, 16, 32) for layer in ("sliding", "full")]
ATTN_TESTS = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=4, ctx=3000, heads="tp8", layer="full"),
    dict(batch=8, ctx=8192, heads="dp", layer="full", spread="mixed"),
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8),
    dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent"),
]
# Shapes near the scored ones: other batches, contexts, block lengths and distributions.
ATTN_NEARBY = [
    dict(batch=b, ctx=ctx, heads="tp8", layer=layer, page=1, q_len=L, dist=dist, spread="agentic")
    for (b, ctx, L, layer, dist) in (
        (2, 80000, 1, "sliding", "soft"), (8, 80000, 1, "full", "coherent"), (24, 80000, 1, "sliding", "soft"),
        (40, 80000, 1, "full", "coherent"), (16, 20000, 1, "full", "soft"), (16, 160000, 1, "full", "coherent"),
        (4, 300000, 1, "full", "normal"), (16, 80000, 4, "sliding", "soft"), (16, 80000, 2, "full", "coherent"),
        (8, 80000, 8, "full", "normal"), (24, 120000, 8, "full", "coherent"), (40, 80000, 8, "sliding", "edges"),
        (32, 2500, 1, "sliding", "edges"), (32, 40000, 8, "full", "soft"))]


def _attn_check(data, out):
    """The judge's check: new cache entries within 8%; output within its FP8-query bound of an
    FP32 reference attending our own new entries. Returns (ok, err, bound)."""
    cfg = data["config"]
    slots = new_slots(data)
    hkv = cfg["num_kv_heads"]
    k = data["k_cache"].view(-1, hkv, HEAD_DIM)[slots]
    v = data["v_cache"].view(-1, hkv, HEAD_DIM)[slots]
    _, rk, rv = ref_kernel(data)
    for got, want, sc in ((k, rk, data["k_scale"]), (v, rv, data["v_scale"])):
        if rel_l2(_dequant(got, sc), _dequant(want, sc)) > CACHE_TOL:
            return False, rel_l2(_dequant(got, sc), _dequant(want, sc)), CACHE_TOL
    ref = ref_kernel(data, new_kv=(k, v))[0].to(torch.bfloat16)
    e4m3 = rel_l2(ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16), ref)
    bound = min(REL_L2_TOL, max(FP8_QUERY_FLOOR, FP8_QUERY_FACTOR * e4m3 + FP8_QUERY_SLACK))
    err = rel_l2(out, ref)
    ok = bool(torch.isfinite(out.float()).all()) and err <= bound and row_worst(out, ref) <= 5 * bound
    return ok, err, bound


def _attn_name(c):
    return (f"B{c['batch']} ctx{c['ctx']} L{c.get('q_len', 1)} {c['layer'][:4]} {c.get('heads')}"
            f" {c.get('dist', 'normal')[:4]} p{c.get('page', 64)}" + (f" s{c['kv_scale']:.3g}" if 'kv_scale' in c else ""))


def _clone_inputs(data):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


def run_attention_sim(variants, budget_s=240.0, seed=7_000_000, groups=("tests", "scored", "nearby"), only=None):
    """variants: {name: fn(data) -> out [B, L, Hq*64]}. Checks every variant on every case and
    times each against SGLang on the scored and nearby cases."""
    t0 = time.time()
    bench = SimBench()
    print(f"sim attention: replay overhead {bench.overhead_us:.1f} us")
    reports = {name: SimReport(f"attention {name}") for name in variants}
    sets = {"tests": ATTN_TESTS, "scored": ATTN_SCORED, "nearby": ATTN_NEARBY}
    i = 0
    for group in groups:
        for case in sets[group]:
            if only is not None and not only(case):
                continue
            if time.time() - t0 > budget_s:
                print(f"sim attention: budget reached at {group} {_attn_name(case)}")
                break
            i += 1
            data = generate_input(seed=seed + i, device="cuda", **case)
            for name, fn in variants.items():
                rep = reports[name]
                try:
                    trial = _clone_inputs(data)
                    out = fn(trial)
                    ok, err, bound = _attn_check(trial, out)
                    if group == "tests":
                        rep.add(group, _attn_name(case), ok, err, bound)
                        continue
                    a, b, sp, ci = bench.compare(lambda: fn(data), lambda: _sglang_attention(data), pairs=15)
                    rep.add(group, _attn_name(case), ok, err, bound, a, b, sp, ci)
                except Exception as e:
                    rep.add(group, _attn_name(case), False, float("nan"), 0.0, note=f"{type(e).__name__}: {str(e)[:120]}")
            del data
            torch.cuda.empty_cache()
    for rep in reports.values():
        rep.summary()
# Phase profile of the v4 attention kernel: bundled after attention_sim_src.py. Per case, one
# cold-L2 call with the kernel's clock64 stamps on; prints, per event, the median and max over
# CTAs of its time since the CTA's start, plus the kernel's span and the CTAs' start skew.

ATT_EVENTS = {0: "start", 1: "lengths", 2: "segments", 3: "prologue", 40: "seg end", 41: "partial+atomic",
              42: "merged", 61: "end"}


def run_attn_prof(cases, variants, seed=7_300_000):
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for ci, case in enumerate(cases):
        data = generate_input(seed=seed + ci, device="cuda", **case)
        for name, fn in variants.items():
            fn(_clone_inputs(data))
            prof = torch.zeros(4096 * 64, dtype=torch.int64, device="cuda")
            trial = _clone_inputs(data)
            torch.cuda.synchronize()
            flush.zero_()
            fn(trial, prof=prof)
            torch.cuda.synchronize()
            pr = prof.view(-1, 64).cpu()
            live = pr[(pr[:, 63] != 0) & (pr[:, 61] != 0)]
            if len(live) == 0:
                print(f"prof {name} {_attn_name(case)}: no stamps")
                continue
            g0 = int(live[:, 63].min())
            t0g, t1g = (live[:, 63] - g0).double(), (live[:, 62] - g0).double()  # int64 first: ns since 1970
            cyc = (live[:, 61] - live[:, 0]).double()
            clk = float(cyc.sum() / (t1g - t0g).clamp(min=1).sum())  # cycles per ns
            span = float(t1g.max() - t0g.min()) / 1e3
            skew = float(t0g.max() - t0g.min()) / 1e3
            line = [f"prof {name} {_attn_name(case)}: {len(live)} CTAs, span {span:.1f}us, start skew {skew:.1f}us, clk {clk:.2f}GHz"]
            for k in list(range(0, 4)) + list(range(4, 40)) + [40, 41, 42] + list(range(44, 60)) + [61]:
                col = live[:, k]
                ok = col != 0
                if k >= 4 and k < 40 and int(ok.sum()) < len(live) // 2:
                    continue
                d = (col[ok] - live[ok, 0]).double() / clk / 1e3
                if len(d) == 0:
                    continue
                if 36 <= k < 40:
                    nm = f"T{k - 36}:qk-issued"
                elif 44 <= k < 60:
                    nm = f"T{(k - 44) // 4}:" + ("qk", "smax", "pv-issued", "pv-done")[(k - 44) % 4]
                else:
                    nm = ATT_EVENTS.get(k, (f"t{(k - 4) // 2} ready" if k % 2 == 0 else f"t{(k - 4) // 2} done"))
                line.append(f"{nm} {float(d.median()):.2f}/{float(d.max()):.2f}")
            print(" | ".join(line), flush=True)
        del data
        torch.cuda.empty_cache()
