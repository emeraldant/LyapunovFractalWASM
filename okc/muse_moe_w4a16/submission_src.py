# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) for Muse Spark Open: SGLang's Marlin GEMMs with lighter glue.

SGLang's path around its two Marlin GEMMs costs several small launches: the expert-map
gather, moe_align_block_size over all 256 experts (+1 for "not local"), and below 32 tokens a
post-expert norm in eager PyTorch (about ten kernels). Here:

- _align_local_kernel: one launch maps global expert ids to this GPU's experts and sorts
  only the local (token, expert) pairs into Marlin's padded blocks, so no block is ever
  spent on the non-local bucket.
- the Marlin GEMMs and silu_and_mul are SGLang's own.
- _post_norm_sum_kernel: one launch for the post-expert norm and the weighted sum at every
  batch size.

Plus our own NVFP4 expert GEMMs (okc/moe_cuda/kernel.py, bundled below): a warp-specialized
kernel streaming each touched expert's weights once (producer warp with bulk copies, 8 or 16
consumer warps decoding E2M1 at 8 instructions per 32-bit word straight into mma.sync
fragments; v6 makes the decoded weights the A operand and the tokens B, so the activation
tile costs a quarter of the shared-memory bandwidth).

prepare() times, per batch-size bucket, SGLang's path against these over Marlin's block
sizes (and its atomic-add mode) and the CUDA kernel's configurations, with CUDA graph replays
and a cold L2, after checking each candidate against an FP32 reference. A CUDA configuration
must also be bitwise repeatable, correct on extra held-out routings, and faster than the
incumbent by a margin on two routings; SGLang's path stays wherever nothing beats it.
"""

import math
import time
import types
from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
# Token buckets: a call with bs tokens runs the plan tuned for the smallest bucket >= bs.
BUCKETS = (1, 8, 16, 32, 128, 256)
TUNE_BUDGET_S = 220.0
ALIGN_CHUNK = 256


# ---------------------------------------------------------------------------------------
# Kernels


@triton.jit
def _align_local_kernel(ids_ptr, local_start_ptr, sorted_ptr, block_expert_ptr, n_padded_ptr, local_ids_ptr,
                        P, MAX_BLOCKS, NE: tl.constexpr,
                        E: tl.constexpr, BS: tl.constexpr, BS_P2: tl.constexpr, CHUNK: tl.constexpr,
                        BLOCKS_CHUNK: tl.constexpr):
    """Sort the P = tokens * top_k routes that land on local experts [local_start,
    local_start + E) (mod NE; local_start is read from device memory) into Marlin's layout:
    each expert's routes (in route order) padded to a multiple of BS with the sentinel P;
    block_expert[i] is block i's local expert (-1 past the end); local_ids[p] is route p's
    local expert or -1.

    One program per local expert: each counts every expert's routes (for the padded
    offsets), then places only its own expert's routes, padding and blocks."""
    e = tl.program_id(0)
    experts = tl.arange(0, E)
    offs = tl.arange(0, CHUNK)
    local_start = tl.load(local_start_ptr)
    counts = tl.zeros((E,), tl.int32)
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        loc = (tl.load(ids_ptr + p, mask=p < P, other=0) - local_start + NE) % NE
        # Non-local routes and the chunk's tail land in bin E, which is dropped.
        loc = tl.where((p < P) & (loc < E), loc, E)
        counts += _bins(loc, E)
    padded = (counts + BS - 1) // BS * BS
    ends = tl.cumsum(padded, axis=0)
    starts = ends - padded
    total = tl.sum(padded, axis=0)
    my_start = tl.sum(tl.where(experts == e, starts, 0), axis=0)
    my_count = tl.sum(tl.where(experts == e, counts, 0), axis=0)
    my_padded = tl.sum(tl.where(experts == e, padded, 0), axis=0)

    running = my_start
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        in_range = p < P
        loc = (tl.load(ids_ptr + p, mask=in_range, other=0) - local_start + NE) % NE
        mine = (loc == e) & in_range
        rank = tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(sorted_ptr + running + rank, p, mask=mine)
        running += tl.sum(mine.to(tl.int32), axis=0)
        if e == 0:
            tl.store(local_ids_ptr + p, tl.where(loc < E, loc, -1), mask=in_range)

    # This expert's padding and its blocks' expert ids.
    j = tl.arange(0, BS_P2)
    tl.store(sorted_ptr + my_start + my_count + j, P, mask=j < my_padded - my_count)
    bo = tl.arange(0, BLOCKS_CHUNK)
    for b0 in range(0, my_padded // BS, BLOCKS_CHUNK):
        blk = b0 + bo
        tl.store(block_expert_ptr + my_start // BS + blk, e, mask=blk < my_padded // BS)
    if e == 0:
        tl.store(n_padded_ptr, total)
        # Blocks past the end.
        for b0 in range(0, MAX_BLOCKS, BLOCKS_CHUNK):
            blk = b0 + bo
            tl.store(block_expert_ptr + blk, -1, mask=(blk < MAX_BLOCKS) & (blk * BS >= total))


@triton.jit
def _bins(loc, E: tl.constexpr):
    """counts[i] = #(loc == i) for i < E (loc in [0, E]), via a one-hot sum."""
    return tl.sum((loc[:, None] == tl.arange(0, E)[None, :]).to(tl.int32), axis=0)


@triton.jit
def _post_norm_sum_kernel(y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps,
                          TOPK: tl.constexpr, H: tl.constexpr):
    """out[t] = sum over local slots k of s[t, k] * RMSNorm(y[t, k] * w); zero if none."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):
        live = tl.load(ids_ptr + t * TOPK + k) >= 0
        if live:
            y = tl.load(y_ptr + (t * TOPK + k).to(tl.int64) * H + offs).to(tl.float32) * w
            inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (inv * tl.load(s_ptr + t * TOPK + k))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------------------
# Paths


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _sglang_block_size(num_tokens, count):
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / count / block_size_m < 0.9:
            break
    return block_size_m


def _experts(x, bank, state, sorted_ids, block_experts, n_padded, local_ids, topk_weights,
             block_size_m, atomic):
    """The two Marlin GEMMs and the activation between them; returns y [M, top_k, H]."""
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    alloc = torch.zeros if atomic else torch.empty
    cache1 = alloc((num_tokens * TOP_K, gemm1_n), device=x.device, dtype=x.dtype)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache3 = alloc((num_tokens * TOP_K, size_k), device=x.device, dtype=x.dtype)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common)
    return y.view(num_tokens, TOP_K, size_k)


def _baseline(data, state):
    """SGLang's path (baseline.py)."""
    cfg = data["config"]
    bank = state["bank"]
    x = data["hidden_states"].contiguous()
    num_tokens = x.shape[0]
    local_ids = state["expert_map"][data["topk_ids"].to(torch.int32)].contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    block_size_m = _sglang_block_size(num_tokens, bank["count"])
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    y = _experts(x, bank, state, sorted_ids, block_experts, n_padded, local_ids, topk_weights,
                 block_size_m, False)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _moe_fast(x, ids, topk_weights, local_start, state, block_size_m, atomic):
    """The local experts' share of the MoE output, [M, H] BF16: x [M, H] BF16, ids [M, 8]
    global expert ids, topk_weights [M, 8] FP32, local_start an int32 [1] device tensor."""
    bank = state["bank"]
    x = x.contiguous()
    num_tokens, hidden = x.shape
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    p = num_tokens * TOP_K
    count = bank["count"]
    max_padded = (p + count * (block_size_m - 1) + 3) // 4 * 4
    max_blocks = triton.cdiv(max_padded, block_size_m)
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=x.device)
    block_experts = torch.empty((max_blocks,), dtype=torch.int32, device=x.device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=x.device)
    local_ids = torch.empty((num_tokens, TOP_K), dtype=torch.int32, device=x.device)
    _align_local_kernel[(count,)](ids, local_start, sorted_ids, block_experts, n_padded, local_ids,
                              p, max_blocks, NE=N_EXPERTS, E=count, BS=block_size_m,
                              BS_P2=triton.next_power_of_2(block_size_m), CHUNK=ALIGN_CHUNK,
                              BLOCKS_CHUNK=min(512, triton.next_power_of_2(max_blocks)), num_warps=4)
    y = _experts(x, bank, state, sorted_ids, block_experts, n_padded, local_ids, topk_weights,
                 block_size_m, atomic)
    out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=x.device)
    _post_norm_sum_kernel[(num_tokens,)](y, state["norm_w"], topk_weights, local_ids, out, POST_NORM_EPS,
                                         TOPK=TOP_K, H=hidden, num_warps=8)
    return out


def _fast(data, state, block_size_m, atomic):
    return _moe_fast(data["hidden_states"], data["topk_ids"], data["topk_weights"], state["local_start"],
                     state, block_size_m, atomic)


def _run(plan, data, state):
    if plan[0] == "base":
        return _baseline(data, state)
    if plan[0] == "cuda":
        return _cuda(data, state, plan[1])
    return _fast(data, state, plan[1], plan[2])


# ---------------------------------------------------------------------------------------
# Our CUDA expert GEMMs (okc/moe_cuda/kernel.py)

# @bundle moe_cuda/kernel.py MOE_DEVICE_SRC MOE_HOST_SRC MOE_CUDA_SRC MOE_CUDA_CPP _MOE_MODULES moe_cuda_module _TARGETS _LEFTOVER _bit_source _lane_gather_index repack_expert_weights scales_supported _route_compact_kernel _gather_rows_kernel _splitk_silu_kernel _post_norm_parts_kernel _post_norm_rows_kernel build_cuda_bank ws_layout moe_cuda_stages moe_cuda_forward

CUDA_CONFIGS = {
    "s1": dict(mt=1, stages=4, ctas_per_sm=4, sab=True),         # v6 swap-AB (mma.sync), 8 tokens per item
    "s2": dict(mt=2, stages=4, ctas_per_sm=4, sab=True),         # v6, 16 tokens per item
    "s1d": dict(mt=1, stages=8, ctas_per_sm=2, sab=True),        # v6, deeper pipeline: few items per SM
    "s1k4": dict(mt=1, stages=4, ctas_per_sm=4, sab=True, kparts=4),   # v6, K split 4 ways (bs <= 8: too few items)
    "s1dw": dict(mt=1, stages=204, ctas_per_sm=2, sab=True),     # s1d with two-chunk stages (16 KB weight copies)
    "s2w": dict(mt=2, stages=202, ctas_per_sm=4, sab=True),      # s2 with two-chunk stages
    "t2": dict(mt=2, stages=304, ctas_per_sm=4, wg=True),        # v7 wgmma + TMA token tiles, 16 tokens
    "t4": dict(mt=4, stages=304, ctas_per_sm=4, wg=True),        # v7 wgmma + TMA, 32 tokens (2 CTAs/SM)
    "v8": dict(mt=8, stages=504, ctas_per_sm=2, wg=True),        # v7, 64-row chunks, wgmma width per chunk
}
CUDA_MARGIN = 0.98   # a CUDA plan must take <= 98% of the incumbent's time over the two timing routings
CUDA_EACH = 1.01     # ... and <= 101% on each of them


def _cuda(data, state, name):
    return moe_cuda_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"], state["local_start"],
                            state["cuda_bank"], state["norm_w"], POST_NORM_EPS, **CUDA_CONFIGS[name])


def _build_cuda(weights, start, count):
    """Compile and repack for the CUDA path; None (keep Marlin) on any failure."""
    try:
        t = time.time()
        moe_cuda_module()
        bank = build_cuda_bank(weights, start, count)
        if bank is None:
            print("moe cuda: scales outside the fast decode's range, CUDA path off")
            return None
        ws_layout(bank)
        for k in ("w13q", "w13s", "w2q", "w2s"):
            del bank[k]
        torch.cuda.synchronize()
        print(f"moe cuda: ready in {time.time() - t:.0f} s")
        return bank
    except Exception as e:
        print(f"moe cuda: unavailable ({type(e).__name__}: {str(e)[:200]})")
        return None


# ---------------------------------------------------------------------------------------
# Tuning


def _make_input(bs, start, count, device, gen):
    """Routing like the judge's: top-8 of uniform scores plus a heavy-tailed per-expert
    popularity, rotated so this GPU's experts are the busiest."""
    x = torch.randn((bs, 4096), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    z = torch.randn((N_EXPERTS,), generator=gen, device=device)
    u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // count).flatten(), minlength=N_EXPERTS // count)
    ids = (ids - count * pairs.argmax()) % N_EXPERTS + start
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS,
                   "local_start": start, "local_count": count, "top_k": TOP_K, "sharding": "ep8"},
    }


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_dequant(packed, sf, gs):
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    vals = torch.where((codes & 8) != 0, -grid[(codes & 7).long()], grid[(codes & 7).long()])
    k = vals.shape[-1]
    return (vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs).unsqueeze(-1)).view_as(vals)


def _reference(data, weights):
    x = data["hidden_states"].float()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    ids = data["topk_ids"].long()
    cfg = data["config"]
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    d_expert = cfg["d_expert"]
    w = weights["post_norm_w"].float()
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _nvfp4_dequant(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :d_expert]) * h[:, d_expert:]
        y = (inter @ _nvfp4_dequant(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T) * (1.0 + w)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out


def _close(out, ref):
    if out is None or out.shape != ref.shape or not torch.isfinite(out.float()).all():
        return False
    diff = out.float() - ref
    if float(diff.norm() / ref.norm().clamp(min=1e-12)) > 1e-2:
        return False
    rows = diff.norm(dim=-1) / torch.maximum(ref.norm(dim=-1), 0.1 * ref.norm(dim=-1).mean())
    return float(rows.max()) <= 5e-2


class _Bench:
    def __init__(self, device):
        self.flush = torch.empty(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device)

    def time(self, fn, reps=15):
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                fn()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        with torch.cuda.stream(self.stream):
            for i in range(reps):
                self.flush.zero_()
                starts[i].record()
                graph.replay()
                ends[i].record()
        ends[-1].synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        del graph
        return times[len(times) // 2]


def _tune_cuda(bucket, data, ref, small, small_ref, best, best_t, state, weights, bench, gen, start, count):
    """Adopts a CUDA configuration only if it is correct on the bucket's routing, a smaller odd
    batch and two more fresh routings, bitwise repeatable, and at most CUDA_MARGIN of the
    incumbent's time over two different routings (and not slower on either)."""
    extra = [_make_input(bucket, start, count, data["hidden_states"].device, gen) for _ in range(2)]
    extra_ref = [_reference(d, weights) for d in extra]
    for name in CUDA_CONFIGS:
        plan = ("cuda", name)
        try:
            out = _cuda(data, state, name)
            ok = _close(out, ref) and torch.equal(out, _cuda(data, state, name))
            ok = ok and (small is None or _close(_cuda(small, state, name), small_ref))
            ok = ok and all(_close(_cuda(d, state, name), r) for d, r in zip(extra, extra_ref))
            if not ok:
                print(f"moe bs{bucket} {plan}: failed the correctness gate, skipped")
                continue
            times = []
            for d in (data, extra[0]):
                tc = bench.time(lambda: _cuda(d, state, name), reps=31)
                ti = bench.time(lambda: _run(best, d, state), reps=31)
                times.append((tc, ti))
        except Exception as e:
            print(f"moe bs{bucket} {plan}: {type(e).__name__}: {str(e)[:160]}")
            continue
        # Faster on the two routings together by the margin, and not slower on either (one
        # near-tie routing no longer vetoes a clear win on the other: gate noise flipped bs 8).
        if (sum(tc for tc, _ in times) <= CUDA_MARGIN * sum(ti for _, ti in times)
                and all(tc <= CUDA_EACH * ti for tc, ti in times)):
            print(f"moe bs{bucket} {plan}: adopted over {best} "
                  + ", ".join(f"{tc * 1e3:.1f}/{ti * 1e3:.1f} us" for tc, ti in times))
            best, best_t = plan, times[0][0]
        else:
            print(f"moe bs{bucket} {plan}: not faster ("
                  + ", ".join(f"{tc * 1e3:.1f}/{ti * 1e3:.1f} us" for tc, ti in times) + ")")
    return best, best_t


def prepare(weights: Weights):
    t0 = time.time()
    device = weights["w13"].device
    start, count = 0, 32
    layer = _marlin_layer(weights, start, count)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
    state = {
        "bank": {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data,
            "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "count": count, "size_n": layer.intermediate_size_per_partition,
        },
        "local": (start, count),
        "local_start": torch.tensor([start], dtype=torch.int32, device=device),
        "expert_map": expert_map,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "plans": {},
    }
    del layer
    state["cuda_bank"] = _build_cuda(weights, start, count)
    bench = _Bench(device)
    gen = torch.Generator(device=device).manual_seed(4321)
    for bucket in BUCKETS:
        data = _make_input(bucket, start, count, device, gen)
        ref = _reference(data, weights)
        base_t = bench.time(lambda: _baseline(data, state))
        best, best_t = ("base",), base_t
        # A smaller, odd token count, to check padding and masking.
        small = _make_input(max(1, bucket - 3), start, count, device, gen) if bucket > 1 else None
        small_ref = _reference(small, weights) if small is not None else None
        sglang_bsm = _sglang_block_size(bucket, count)
        for bsm in BLOCK_SIZE_M_CANDIDATES:
            # Marlin's useful block sizes sit around the tokens per expert.
            if bsm > 16 and bsm > 2 * sglang_bsm:
                continue
            for atomic in (False, True):
                if time.time() - t0 > TUNE_BUDGET_S:
                    break
                plan = ("fast", bsm, atomic)
                try:
                    if not _close(_fast(data, state, bsm, atomic), ref):
                        print(f"moe bs{bucket} {plan}: wrong result, skipped")
                        continue
                    if small is not None and not _close(_fast(small, state, bsm, atomic), small_ref):
                        print(f"moe bs{bucket} {plan}: wrong at bs {small['config']['bs']}, skipped")
                        continue
                    t = bench.time(lambda: _fast(data, state, bsm, atomic))
                except Exception as e:  # a config that fails is skipped
                    print(f"moe bs{bucket} {plan}: {type(e).__name__}: {str(e)[:160]}")
                    continue
                if t < best_t:
                    best, best_t = plan, t
        if best[0] != "base":
            tb = bench.time(lambda: _baseline(data, state), reps=31)
            tw = bench.time(lambda: _run(best, data, state), reps=31)
            if tw >= tb:
                best, best_t = ("base",), tb
            else:
                best_t, base_t = tw, tb
        if state["cuda_bank"] is not None and time.time() - t0 < TUNE_BUDGET_S:
            best, best_t = _tune_cuda(bucket, data, ref, small, small_ref, best, best_t, state, weights, bench, gen,
                                      start, count)
        state["plans"][bucket] = best
        print(f"moe plan bs{bucket:<4d}: {best} {best_t * 1e3:.1f} us (sglang {base_t * 1e3:.1f} us)")
    del bench
    print(f"prepare: tuned in {time.time() - t0:.0f} s")
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    plan = ("base",)
    if (cfg["local_start"], cfg["local_count"]) == state["local"] and cfg["top_k"] == TOP_K:
        bs = data["hidden_states"].shape[0]
        for b in BUCKETS:
            if bs <= b:
                plan = state["plans"].get(b, plan)
                break
    if plan[0] == "base" and (cfg["local_start"], cfg["local_count"]) != state["local"]:
        raise ValueError("this submission holds experts 0-31 only")
    return _run(plan, data, state)
