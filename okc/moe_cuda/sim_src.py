# MoE CUDA sim: bundled at the end of the MoE submission's prepare() (tools/make_host.py
# --after). Inputs from the judge's own generate_input (copied from reference.py) on unused
# seeds; correctness against the submission's FP32 reference with the judge's bound; timing
# against SGLang's path and our current path with the judge's protocol.

import random
import time

import torch
import triton
import triton.language as tl

# @bundle muse_moe_w4a16/reference.py D_HIDDEN D_EXPERT ROUTED_PER_TOKEN LOCAL POPULARITY generate_input
# @bundle muse_moe_w4a16/nvfp4.py outlier_channels
# @bundle moe_cuda/kernel.py MOE_CUDA_SRC MOE_CUDA_CPP _MOE_MODULE moe_cuda_module _NIBBLE_POS _lane_gather_index repack_expert_weights scales_supported _route_compact_kernel _post_norm_rows_kernel build_cuda_bank moe_cuda_forward
# @bundle sim/harness.py SimBench rel_l2 row_worst SimReport

MOE_SIM_CASES = (
    [("scored", dict(bs=b)) for b in (1, 16, 32, 8, 128, 256)]
    + [("nearby", dict(bs=b)) for b in (2, 5, 12, 24, 48, 64, 96, 200, 250)]
    + [("tests", dict(bs=128, dist="skewed")), ("tests", dict(bs=257)), ("tests", dict(bs=33)),
       ("nearby", dict(bs=32, dist="skewed")), ("nearby", dict(bs=8, dist="skewed"))])


def _mt_for(bs):
    return 1 if bs <= 16 else 2 if bs <= 32 else 4


def run_moe_cuda_sim(weights, state, budget_s=200.0, seed=9_000_000):
    t0 = time.time()
    try:
        moe_cuda_module()
    except Exception as e:
        print(f"moe cuda: compile failed: {type(e).__name__}: {str(e)[-1200:]}")
        return
    print(f"moe cuda: compiled in {time.time() - t0:.0f} s")
    t1 = time.time()
    bank = build_cuda_bank(weights)
    if bank is None:
        print("moe cuda: scales unsupported, skipped")
        return
    torch.cuda.synchronize()
    print(f"moe cuda: repacked in {time.time() - t1:.1f} s")
    bench = SimBench()
    rep_c = SimReport("cuda vs sglang")
    rep_p = SimReport("plan vs sglang")
    ls = torch.zeros(1, dtype=torch.int32, device="cuda")
    for i, (group, case) in enumerate(MOE_SIM_CASES):
        if time.time() - t0 > budget_s:
            print(f"moe cuda: budget reached at {case}")
            break
        data = generate_input(sharding="ep8", seed=seed + i, device="cuda", **case)
        bs = case["bs"]
        name = f"bs{bs} {case.get('dist', 'normal')}"
        mt = _mt_for(bs)

        def ours():
            return moe_cuda_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"], ls, bank,
                                    state["norm_w"], POST_NORM_EPS, mt=mt)
        try:
            ref = _reference(data, weights)
            out = ours()
            err = rel_l2(out, ref)
            ok = bool(torch.isfinite(out.float()).all()) and err <= 1.5e-2 and row_worst(out, ref) <= 7.5e-2
            if group == "tests":
                rep_c.add(group, name, ok, err, 1.5e-2)
                continue
            a, b, sp, ci = bench.compare(ours, lambda: _baseline(data, state), pairs=15)
            rep_c.add(group, name, ok, err, 1.5e-2, a, b, sp, ci)
            plan = ("base",)
            for bk in BUCKETS:
                if bs <= bk:
                    plan = state["plans"].get(bk, plan)
                    break
            a, b, sp, ci = bench.compare(lambda: _run(plan, data, state), lambda: _baseline(data, state), pairs=15)
            rep_p.add(group, name, True, 0.0, 1.0, a, b, sp, ci)
        except Exception as e:
            import traceback
            rep_c.add(group, name, False, float("nan"), 0.0, note=f"{type(e).__name__}: {str(e)[:150]}")
            print(traceback.format_exc()[-800:])
    rep_c.summary()
    rep_p.summary()
