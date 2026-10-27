# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: skinny BF16 GEMMs (and the FP32 router) as Triton
kernels tuned for the decode / verify token counts.

The weights are read once per call and dominate the time (memory bound up to m 32), so the
kernels stream them with enough CTAs to fill HBM: a tensor-core kernel (x @ W.T with tl.dot)
for m >= 16 and a CUDA-core GEMV for m <= 8, both with optional split-K that accumulates in
FP32 with atomics. The router keeps its FP32 weights for the GEMV, and the exact BF16 hi + lo
split (FP32 accumulation) for the tensor-core kernel.

prepare() picks, per (layer, token bucket), the fastest correct candidate by timing CUDA
graph replays with a cold L2, SGLang's own path included, so a candidate that loses or
fails never runs.
"""

import math
import time
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except Exception:  # pragma: no cover - outside the image
    _bf16_gemm_dispatch_impl = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
# Token buckets: a call with m tokens runs the plan tuned for the smallest bucket >= m.
BUCKETS = (1, 8, 16, 32, 128, 256)
ROUTER_FP32_MAX_TOKENS = 8
UNPOOL_SLICE = 1024
TUNE_BUDGET_S = 200.0


# ---------------------------------------------------------------------------------------
# Kernels


@triton.jit
def _gemm_dot_kernel(x_ptr, w_ptr, w2_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                     SPLIT_K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     TWO_W: tl.constexpr, ATOMIC: tl.constexpr):
    """out[M, N] (+)= x[M, K] @ W[N, K].T on tensor cores; W (+ W2 when TWO_W) is BF16."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    m_mask = rm[:, None] < M
    n_mask = rn[:, None] < N
    x_ptrs = x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :]
    w_ptrs = w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :]
    w2_ptrs = w2_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, k_per, BK):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0, eviction_policy="evict_first")
        acc += tl.dot(x, tl.trans(w))
        if TWO_W:
            w2 = tl.load(w2_ptrs, mask=n_mask, other=0.0, eviction_policy="evict_first")
            acc += tl.dot(x, tl.trans(w2))
            w2_ptrs += BK
        x_ptrs += BK
        w_ptrs += BK
    o_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :]
    o_mask = (rm[:, None] < M) & (rn[None, :] < N)
    if ATOMIC:
        tl.atomic_add(o_ptrs, acc, mask=o_mask, sem="relaxed")
    else:
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


@triton.jit
def _gemv_fma_kernel(x_ptr, w_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                     SPLIT_K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     ATOMIC: tl.constexpr):
    """out[M, N] (+)= x[M, K] @ W[N, K].T on CUDA cores, for a few tokens; W is BF16 or FP32."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    m_mask = rm[:, None] < M
    n_mask = rn[:, None] < N
    x_ptrs = x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :]
    w_ptrs = w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :]
    acc = tl.zeros((BM, BN, BK), dtype=tl.float32)
    for _ in range(0, k_per, BK):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
        x_ptrs += BK
        w_ptrs += BK
    res = tl.sum(acc, axis=2)
    o_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :]
    o_mask = (rm[:, None] < M) & (rn[None, :] < N)
    if ATOMIC:
        tl.atomic_add(o_ptrs, res, mask=o_mask, sem="relaxed")
    else:
        tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------------------
# Plans: a plan is a tuple describing one launch; run_plan executes it.


# ---------------------------------------------------------------------------------------
# CUDA weight stream for m <= 32 (okc/bf16_cuda/kernel.py)

# @bundle bf16_cuda/kernel.py BF16_SRC BF16_CPP _BF16_MOD bf16_module pack_bf16 bf16_stream_mm


def _run_plan(plan, x, state, layer, n):
    kind = plan[0]
    if kind == "base":
        return _baseline(x, state, layer, n)
    if kind == "cuda":   # ("cuda", S, grid)
        return bf16_stream_mm(state["bs_mod"], state[layer + "_pk"], x, n, plan[1], plan[2], state["bs_part"],
                              state["bs_cnt"])
    m, k = x.shape
    router = layer == "router"
    out_dtype = torch.float32 if router else torch.bfloat16
    if kind == "dot":
        _, bm, bn, bk, split, warps, stages = plan
        if router:
            w, w2 = state["router_hi"], state["router_lo"]
        else:
            w = w2 = state[layer]
    else:
        _, bm, bn, bk, split, warps, stages = plan
        w = state["router"] if router else state[layer]
        w2 = None
    atomic = split > 1
    if atomic:
        acc = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        acc = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), split)
    if kind == "dot":
        _gemm_dot_kernel[grid](x, w, w2, acc, m, n, k, x.stride(0), w.stride(0), acc.stride(0),
                               split, BM=bm, BN=bn, BK=bk, TWO_W=router, ATOMIC=atomic,
                               num_warps=warps, num_stages=stages)
    else:
        _gemv_fma_kernel[grid](x, w, acc, m, n, k, x.stride(0), w.stride(0), acc.stride(0),
                               split, BM=bm, BN=bn, BK=bk, ATOMIC=atomic,
                               num_warps=warps, num_stages=stages)
    if atomic and out_dtype != torch.float32:
        return acc.to(out_dtype)
    return acc


def _baseline(x, state, layer, n):
    """SGLang's path (baseline.py)."""
    if layer != "router":
        w = state[layer][:n]
        if _bf16_gemm_dispatch_impl is not None:
            return _bf16_gemm_dispatch_impl(x, w, None)
        return F.linear(x, w)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _pow2_at_least(v):
    return 1 << max(0, (v - 1).bit_length())


def _candidates(layer, m, n, k):
    """Launch plans worth timing for m tokens of this layer, cheapest-to-compile first."""
    sms = 132
    cands = []
    seen = set()
    if m <= 32 and layer != "router":   # the CUDA weight stream first
        nt = (m + 7) // 8
        for S in (2, 4, 8, 16, 32):   # persistent grid (1 x 512 threads per SM at 128 regs), K split S ways
            if k % (S * 16) == 0:
                cands.append(("cuda", S, sms))

    def add(plan):
        kind, bm, bn, bk, split = plan[:5]
        if plan in seen or k % (split * bk) or k // split < bk:
            return
        ctas = triton.cdiv(n, bn) * triton.cdiv(m, bm) * split
        if ctas < sms // 2 or ctas > 64 * sms:
            return
        seen.add(plan)
        cands.append(plan)

    def splits_for(tiles):
        out = []
        for s in (1, 2, 4, 8, 16):
            if tiles * s >= sms:
                out.append(s)
            if tiles * s >= 2 * sms:
                break
        return out or [16]

    if m <= 8:
        bm = _pow2_at_least(m)
        # CUDA-core GEMV: keep BM * BN * BK at 8K fp32 accumulators per 128 threads.
        for bn in (4, 8, 16):
            for bk in (256, 512, 1024):
                if bm * bn * bk > 8192 or bm * bn * bk < 2048:
                    continue
                tiles = triton.cdiv(n, bn)
                for s in splits_for(tiles):
                    add(("fma", bm, bn, bk, s, 4, 2))
    if m >= 2:
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        if m >= 128:
            bms = (64, 128)
        else:
            bms = (bm,)
        for bm in bms:
            for bn in ((16, 32, 64) if m <= 32 else (64, 128)):
                for bk in (128, 256):
                    if bm * bk + bn * bk > 64 * 256 + 64 * 256:
                        continue
                    tiles = triton.cdiv(n, bn) * triton.cdiv(m, bm)
                    for s in splits_for(tiles):
                        add(("dot", bm, bn, bk, s, 4, 4 if bk == 128 else 3))
    return cands


# ---------------------------------------------------------------------------------------
# Tuning


class _Bench:
    def __init__(self, device):
        # Larger than H200's L2 (50 MB), cleared before each replay as the judge does.
        self.flush = torch.empty(64 * 1024 * 1024 * 2 // 4, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device)

    def time(self, fn, reps=25):
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


def _make_x(m, k, device, gen):
    x = torch.randn((m, k), generator=gen, device=device)
    # A probe row: one power-of-two entry, so its output is an exact weight column.
    x[m // 2].zero_()
    x[m // 2, (m * 7919) % k] = 32.0
    return x.to(torch.bfloat16)


def _check(out, x, w_ref, router):
    if out is None or not torch.isfinite(out).all():
        return False
    ref = x.double() @ w_ref.double().T
    err = float((out.double() - ref).norm() / ref.norm().clamp(min=1e-30))
    probe = x.shape[0] // 2
    perr = float((out[probe].double() - ref[probe]).norm() / ref[probe].norm().clamp(min=1e-30))
    return err <= (2e-5 if router else 4e-3) and perr <= 2e-4


def prepare(weights: Weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    try:   # the CUDA weight stream: packed weights (same BF16 values), parts and counters
        state["bs_mod"] = bf16_module()
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            state[name + "_pk"] = pack_bf16(state[name])
        state["bs_part"] = torch.empty(64 * 512 * 512, dtype=torch.float32, device=router.device)
        state["bs_cnt"] = torch.zeros(512, dtype=torch.int32, device=router.device)
        print(f"bf16 stream: ready ({time.time() - t0:.0f} s)")
    except Exception as e:
        print(f"bf16 stream: unavailable ({type(e).__name__}: {str(e)[:200]})")
    plans = {}
    state["plans"] = plans
    device = router.device
    bench = _Bench(device)
    gen = torch.Generator(device=device).manual_seed(1234)
    # Tune the most common layers first so a blown budget costs least.
    for bucket in BUCKETS:
        for layer in LAYERS:
            n_full, k = weights[layer].shape
            n = UNPOOL_SLICE if (layer == "unpooler" and bucket <= 8) else n_full
            x = _make_x(bucket, k, device, gen)
            w_ref = state[layer][:n]
            base_t = bench.time(lambda: _baseline(x, state, layer, n))
            best, best_t = ("base",), base_t
            for plan in _candidates(layer, bucket, n, k):
                if plan[0] == "cuda" and "bs_mod" not in state:
                    continue
                if time.time() - t0 > TUNE_BUDGET_S:
                    break
                try:
                    out = _run_plan(plan, x, state, layer, n)
                    torch.cuda.current_stream().synchronize()
                    if not _check(out, x, w_ref, layer == "router"):
                        if plan[0] == "cuda":
                            ref = x.double() @ w_ref.double().T
                            print(f"cuda {layer} m{bucket} {plan}: wrong, rel err "
                                  f"{float((out.double() - ref).norm() / ref.norm()):.2e}")
                        continue
                    # Masking: a smaller, odd token count must be right too.
                    if bucket > 1:
                        xs = _make_x(bucket - 1, k, device, gen)
                        if not _check(_run_plan(plan, xs, state, layer, n), xs, w_ref, layer == "router"):
                            continue
                    t = bench.time(lambda: _run_plan(plan, x, state, layer, n))
                except Exception as e:  # a config that fails to compile or launch is skipped
                    print(f"skip {layer} m{bucket} {plan}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                if plan[0] == "cuda":
                    print(f"cuda {layer} m{bucket} {plan}: {t * 1e3:.1f} us (best {best_t * 1e3:.1f})")
                if t < best_t:
                    best, best_t = plan, t
            # Re-time the winner against the baseline to guard against a lucky sample.
            if best[0] != "base":
                tb = bench.time(lambda: _baseline(x, state, layer, n), reps=41)
                tw = bench.time(lambda: _run_plan(best, x, state, layer, n), reps=41)
                if tw >= tb:
                    best, best_t = ("base",), tb
                else:
                    best_t, base_t = tw, tb
            plans[(layer, bucket)] = best
            print(f"plan {layer:9s} m{bucket:<4d} n{n:<5d}: {best} {best_t * 1e3:.1f} us"
                  f" (sglang {base_t * 1e3:.1f} us)")
    del bench
    print(f"prepare: tuned in {time.time() - t0:.0f} s")
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    plan = ("base",)
    for b in BUCKETS:
        if m <= b:
            plan = state["plans"].get((layer, b), plan)
            break
    if plan[0] != "base" and not x.is_contiguous():
        x = x.contiguous()
    return _run_plan(plan, x, state, layer, n)
