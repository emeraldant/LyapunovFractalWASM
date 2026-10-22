# okc: problem=muse_attention_decode
"""Gated attention decode for Muse Spark Open in two Triton launches.

SGLang runs five or six kernels per call (QK-norm + RoPE, the cache store, the query's FP8
cast, FA3 and its split combine, the gated norm). At these sizes most of that is launch
latency, so here:

1. _attn_split_kernel, one program per (split of the keys, KV head, request): normalizes its
   group's queries from the fused projection output (RMSNorm, the 5.473 factor and
   interleaved RoPE in registers, kept BF16, like the reference), streams its share of the
   paged FP8 cache with an online softmax, and the last split also builds the block's new
   keys and values, stores them to the cache in FP8 and attends to them straight from
   registers. With one split it finishes the row itself (step 2's norm and gate).
2. _attn_combine_kernel merges the splits and applies RMSNorm x sigmoid(gate).

The split count depends only on shapes (batch, heads, window or the page table's capacity),
so the launch is CUDA-graph safe; each program reads the request's real length on device.
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
NUM_SMS = 132
_INV_FREQ = {}


def _inv_freq(theta, d, device):
    """RoPE frequencies computed exactly as SGLang's / the reference's (FP32)."""
    key = (theta, d, device)
    t = _INV_FREQ.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        else:
            t = torch.zeros(d // 2, dtype=torch.float, device=device)
        _INV_FREQ[key] = t = t.contiguous()
    return t


@triton.jit
def _load_heads(qkv_ptr, inv_freq_ptr, tok0, pos0, col0, rows, QKV_W, eps, factor,
                G: tl.constexpr, NROWS: tl.constexpr, HD: tl.constexpr, ROPE: tl.constexpr, NORM: tl.constexpr):
    """[R, HD] fp32 rows r = j * G + g: head (col0 / HD + g) of token tok0 + j (position pos0 + j),
    RMS-normalized and rotated (interleaved RoPE) when asked. Rows >= NROWS are zero."""
    d = tl.arange(0, HD)
    j = rows // G
    g = rows % G
    rmask = rows < NROWS
    base = qkv_ptr + (tok0 + j)[:, None].to(tl.int64) * QKV_W + col0 + g[:, None] * HD
    x = tl.load(base + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    if NORM:
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / HD + eps)
        x = x * (inv * factor)[:, None]
    if ROPE:
        # out[2i] = x[2i] cos - x[2i+1] sin, out[2i+1] = x[2i+1] cos + x[2i] sin
        xp = tl.load(base + (d ^ 1)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        if NORM:
            xp = xp * (inv * factor)[:, None]
        freq = tl.load(inv_freq_ptr + d // 2)
        ang = (pos0 + j).to(tl.float32)[:, None] * freq[None, :]
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        x = x * tl.cos(ang) + sign[None, :] * xp * tl.sin(ang)
    return x


@triton.jit
def _attn_split_kernel(
        qkv_ptr, inv_freq_ptr, seq_lens_ptr, page_table_ptr, cache_loc_ptr, k_cache_ptr, v_cache_ptr,
        k_scale_ptr, v_scale_ptr, out_ptr, ws_acc_ptr, ws_ml_ptr,
        QKV_W, PT_STRIDE, CHUNK, sm_scale_log2, qk_factor, eps,
        HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, L: tl.constexpr, G: tl.constexpr,
        R: tl.constexpr, RN: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
        SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, DIRECT: tl.constexpr):
    s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    seq = tl.load(seq_lens_ptr + b)
    rows = tl.arange(0, R)
    jrow = rows // G
    qpos = seq + jrow  # each row's query position
    d = tl.arange(0, HD)
    tok0 = b * L

    # Queries: RMSNorm x 5.473 (+ RoPE), rounded to BF16 as the reference does.
    q = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, h * G * HD, rows, QKV_W, eps, qk_factor,
                    G, G * L, HD, ROPE, True).to(tl.bfloat16)

    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    qk_scale = sm_scale_log2 * k_scale  # scores in log2 units

    # This split's share of the cached keys [lo, seq).
    if WINDOW > 0:
        first = tl.maximum(seq - WINDOW + 1, 0)
    else:
        first = seq * 0
    lo = first + s * CHUNK
    hi = tl.minimum(lo + CHUNK, seq)

    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, HD), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(lo, hi, BLOCK_N):
        kpos = start + offs_n
        kmask = kpos < hi
        page = tl.load(page_table_ptr + b.to(tl.int64) * PT_STRIDE + kpos // PAGE, mask=kmask, other=0)
        row = page.to(tl.int64) * PAGE + kpos % PAGE
        kv_off = (row * HKV + h)[:, None] * HD + d[None, :]
        k = tl.load(k_cache_ptr + kv_off, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(v_cache_ptr + kv_off, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
        valid = kmask[None, :]
        if WINDOW > 0:
            valid = valid & (kpos[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(valid, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    if s == SPLITS - 1:
        # The block's own L keys and values: normalized (+ RoPE), FP8 like the cache, stored to
        # their rows, and attended from registers (causal inside the block).
        nrows = tl.arange(0, RN)
        nmask = nrows < L
        kn = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, HQ * HD + h * HD, nrows, QKV_W, eps, 1.0,
                         1, L, HD, ROPE, True)
        vn = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, (HQ + HKV) * HD + h * HD, nrows, QKV_W, eps, 1.0,
                         1, L, HD, False, False)
        kq = (kn / k_scale).to(k_cache_ptr.dtype.element_ty)
        vq = (vn / v_scale).to(v_cache_ptr.dtype.element_ty)
        loc = tl.load(cache_loc_ptr + tok0 + nrows, mask=nmask, other=0).to(tl.int64)
        new_off = (loc * HKV + h)[:, None] * HD + d[None, :]
        tl.store(k_cache_ptr + new_off, kq, mask=nmask[:, None])
        tl.store(v_cache_ptr + new_off, vq, mask=nmask[:, None])
        k = kq.to(tl.bfloat16)
        v = vq.to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
        valid = nmask[None, :] & (nrows[None, :] <= jrow[:, None])
        sc = tl.where(valid, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    rmask = rows < G * L
    if DIRECT:
        o = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None] * v_scale
        _norm_gate_store(o, qkv_ptr, out_ptr, tok0, h, rows, rmask, QKV_W, eps, HQ, HKV, HD, G)
    else:
        base = ((b * HKV + h) * SPLITS + s).to(tl.int64) * R
        tl.store(ws_acc_ptr + (base + rows)[:, None] * HD + d[None, :], acc)
        tl.store(ws_ml_ptr + (base + rows) * 2, m_i)
        tl.store(ws_ml_ptr + (base + rows) * 2 + 1, l_i)


@triton.jit
def _norm_gate_store(o, qkv_ptr, out_ptr, tok0, h, rows, rmask, QKV_W, eps,
                     HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, G: tl.constexpr):
    """out[token, head] = RMSNorm(o) * sigmoid(gate) for rows r = j * G + g of KV head h."""
    d = tl.arange(0, HD)
    j = rows // G
    head = h * G + rows % G
    inv = tl.rsqrt(tl.sum(o * o, axis=1) / HD + eps)
    tok = (tok0 + j).to(tl.int64)
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * HD + head[:, None] * HD + d[None, :],
                   mask=rmask[:, None], other=0.0).to(tl.float32)
    y = o * inv[:, None] * tl.sigmoid(gate)
    tl.store(out_ptr + tok[:, None] * (HQ * HD) + head[:, None] * HD + d[None, :], y.to(tl.bfloat16),
             mask=rmask[:, None])


@triton.jit
def _attn_combine_kernel(qkv_ptr, out_ptr, ws_acc_ptr, ws_ml_ptr, v_scale_ptr, QKV_W, eps,
                         HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, L: tl.constexpr,
                         G: tl.constexpr, R: tl.constexpr, SPLITS: tl.constexpr):
    h = tl.program_id(0)
    b = tl.program_id(1)
    rows = tl.arange(0, R)
    d = tl.arange(0, HD)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, HD), tl.float32)
    for s in range(SPLITS):
        base = ((b * HKV + h) * SPLITS + s).to(tl.int64) * R
        m_s = tl.load(ws_ml_ptr + (base + rows) * 2)
        l_s = tl.load(ws_ml_ptr + (base + rows) * 2 + 1)
        a_s = tl.load(ws_acc_ptr + (base + rows)[:, None] * HD + d[None, :])
        m_new = tl.maximum(m_i, m_s)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        beta = tl.exp2(m_s - m_safe)
        l_i = l_i * alpha + l_s * beta
        acc = acc * alpha[:, None] + a_s * beta[:, None]
        m_i = m_new
    o = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None] * tl.load(v_scale_ptr)
    _norm_gate_store(o, qkv_ptr, out_ptr, b * L, h, rows, rows < G * L, QKV_W, eps, HQ, HKV, HD, G)


def _plan(batch, hkv, span, length):
    """(splits, block_n): enough programs to keep the GPU's memory system busy, with at
    least a few key blocks per split."""
    block_n = 64
    want = 2 * NUM_SMS
    splits = max(1, min(triton.cdiv(want, batch * hkv), triton.cdiv(span, 2 * block_n)))
    return splits, block_n


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, hd = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    page = cfg["page_size"]
    window = cfg["window"]
    qkv = data["qkv"]
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    device = qkv.device
    page_table = data["page_table"]
    capacity = page_table.shape[1] * page
    span = min(window, capacity) if window else capacity
    splits, block_n = _plan(batch, hkv, span, length)
    chunk = triton.cdiv(span, splits)
    chunk = triton.cdiv(chunk, block_n) * block_n
    splits = triton.cdiv(span, chunk)
    rows = max(16, triton.next_power_of_2(group * length))
    rows_new = max(16, triton.next_power_of_2(length))
    out = torch.empty((batch, length, hq * hd), device=device, dtype=torch.bfloat16)
    direct = splits == 1
    if direct:
        ws_acc = ws_ml = out  # unused
    else:
        ws_acc = torch.empty((batch, hkv, splits, rows, hd), device=device, dtype=torch.float32)
        ws_ml = torch.empty((batch, hkv, splits, rows, 2), device=device, dtype=torch.float32)
    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"], hd, device)
    qkv_w = qkv.shape[-1]
    _attn_split_kernel[(splits, hkv, batch)](
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], out, ws_acc, ws_ml,
        qkv_w, page_table.stride(0), chunk, cfg["sm_scale"] * LOG2E, cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, HD=hd, L=length, G=group, R=rows, RN=rows_new, PAGE=page, WINDOW=window, ROPE=rope,
        SPLITS=splits, BLOCK_N=block_n, DIRECT=direct, num_warps=4, num_stages=2)
    if not direct:
        _attn_combine_kernel[(hkv, batch)](
            qkv, out, ws_acc, ws_ml, data["v_scale"], qkv_w, RMS_EPS,
            HQ=hq, HKV=hkv, HD=hd, L=length, G=group, R=rows, SPLITS=splits, num_warps=4)
    return out
