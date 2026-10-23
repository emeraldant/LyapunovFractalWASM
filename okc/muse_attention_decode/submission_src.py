# okc: problem=muse_attention_decode
"""Gated attention decode for Muse Spark Open: our Triton kernels on decode steps (v1 on
sliding-window layers, v3 on full-attention layers), SGLang's FA3 path on verify blocks
(see kernel_fn).

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
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


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


def _fused_attention(qkv, data, cfg):
    """Attention between the QKV and output projections: qkv [T, (2Hq + 2Hkv) * 64] BF16 (the
    fused projection output, request-major), returns [T, Hq * 64] BF16. Writes the new keys
    and values into data's caches."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, hd = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    page = cfg["page_size"]
    window = cfg["window"]
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
    out = torch.empty((batch * length, hq * hd), device=device, dtype=torch.bfloat16)
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


# ---------------------------------------------------------------------------------------
# v3: memory-first. The probe run (8433b3e3) measured random 128 B rows at 4.03 TB/s, the
# same as streaming, while FA3 reaches 1.3-1.7 TB/s on full layers and v1 1.1 TB/s (each
# step waited on its page-table load, then its KV load, with few keys in flight). Here one
# program covers a key range for a pair of KV heads (each key's 128 B K and V rows are read
# by one program, and page indices once), the grid is sized to ~4 programs per SM, and a
# separate combine reduces the splits per query head in parallel.


@triton.jit
def _attn3_step(q, k, v, valid, m_i, l_i, acc, qk_scale):
    """One online-softmax step: scores of q [R, HD] against k [N, HD], masked by valid [R, N]."""
    sc = tl.dot(q, tl.trans(k)) * qk_scale
    sc = tl.where(valid, sc, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(sc, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_safe)
    p = tl.exp2(sc - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l_i, acc


@triton.jit
def _attn3_new_keys(qkv_ptr, inv_freq_ptr, cache_loc_ptr, k_cache_ptr, v_cache_ptr, q, m_i, l_i, acc,
                    tok0, seq, h, jrow, QKV_W, eps, k_scale, v_scale, qk_scale,
                    HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, L: tl.constexpr, RN: tl.constexpr,
                    ROPE: tl.constexpr):
    """The block's own L keys and values for KV head h: normalized (+ RoPE), stored to the
    cache in FP8, and attended from registers (causal inside the block)."""
    d = tl.arange(0, HD)
    nrows = tl.arange(0, RN)
    nmask = nrows < L
    kn = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, HQ * HD + h * HD, nrows, QKV_W, eps, 1.0,
                     1, L, HD, ROPE, True)
    vn = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, (HQ + HKV) * HD + h * HD, nrows, QKV_W, eps, 1.0,
                     1, L, HD, False, False)
    kq = (kn / k_scale).to(k_cache_ptr.dtype.element_ty)
    vq = (vn / v_scale).to(v_cache_ptr.dtype.element_ty)
    loc = tl.load(cache_loc_ptr + tok0 + nrows, mask=nmask, other=0).to(tl.int64)
    off = (loc * HKV + h)[:, None] * HD + d[None, :]
    tl.store(k_cache_ptr + off, kq, mask=nmask[:, None])
    tl.store(v_cache_ptr + off, vq, mask=nmask[:, None])
    valid = nmask[None, :] & (nrows[None, :] <= jrow[:, None])
    return _attn3_step(q, kq.to(tl.bfloat16), vq.to(tl.bfloat16), valid, m_i, l_i, acc, qk_scale)


@triton.jit
def _attn3_split_kernel(
        qkv_ptr, inv_freq_ptr, seq_lens_ptr, page_table_ptr, cache_loc_ptr, k_cache_ptr, v_cache_ptr,
        k_scale_ptr, v_scale_ptr, ws_acc_ptr, ws_ml_ptr,
        QKV_W, PT_STRIDE, CHUNK, sm_scale_log2, qk_factor, eps,
        HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, L: tl.constexpr, G: tl.constexpr,
        R: tl.constexpr, RN: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
        SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, HP: tl.constexpr):
    s = tl.program_id(0)
    h0 = tl.program_id(1) * HP
    h1 = h0 + HP - 1  # == h0 when HP == 1 (its work is skipped)
    b = tl.program_id(2)
    seq = tl.load(seq_lens_ptr + b)
    rows = tl.arange(0, R)
    jrow = rows // G
    qpos = seq + jrow
    d = tl.arange(0, HD)
    tok0 = b * L
    q0 = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, h0 * G * HD, rows, QKV_W, eps, qk_factor,
                     G, G * L, HD, ROPE, True).to(tl.bfloat16)
    q1 = _load_heads(qkv_ptr, inv_freq_ptr, tok0, seq, h1 * G * HD, rows, QKV_W, eps, qk_factor,
                     G, G * L, HD, ROPE, True).to(tl.bfloat16)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    qk_scale = sm_scale_log2 * k_scale
    if WINDOW > 0:
        first = tl.maximum(seq - WINDOW + 1, 0)
    else:
        first = seq * 0
    lo = first + s * CHUNK
    hi = tl.minimum(lo + CHUNK, seq)

    m0 = tl.full((R,), float("-inf"), tl.float32)
    l0 = tl.zeros((R,), tl.float32)
    a0 = tl.zeros((R, HD), tl.float32)
    m1 = tl.full((R,), float("-inf"), tl.float32)
    l1 = tl.zeros((R,), tl.float32)
    a1 = tl.zeros((R, HD), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    pt_row = page_table_ptr + b.to(tl.int64) * PT_STRIDE
    for start in range(lo, hi, BLOCK_N):
        kpos = start + offs_n
        kmask = kpos < hi
        page = tl.load(pt_row + kpos // PAGE, mask=kmask, other=0)
        base = (page.to(tl.int64) * PAGE + kpos % PAGE) * HKV
        off0 = (base + h0)[:, None] * HD + d[None, :]
        off1 = off0 + HD
        k0 = tl.load(k_cache_ptr + off0, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
        v0 = tl.load(v_cache_ptr + off0, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
        valid = kmask[None, :]
        if WINDOW > 0:
            valid = valid & (kpos[None, :] > qpos[:, None] - WINDOW)
        m0, l0, a0 = _attn3_step(q0, k0, v0, valid, m0, l0, a0, qk_scale)
        if HP == 2:
            k1 = tl.load(k_cache_ptr + off1, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
            v1 = tl.load(v_cache_ptr + off1, mask=kmask[:, None], other=0.0).to(tl.bfloat16)
            m1, l1, a1 = _attn3_step(q1, k1, v1, valid, m1, l1, a1, qk_scale)

    if s == SPLITS - 1:
        m0, l0, a0 = _attn3_new_keys(qkv_ptr, inv_freq_ptr, cache_loc_ptr, k_cache_ptr, v_cache_ptr, q0, m0, l0, a0,
                                     tok0, seq, h0, jrow, QKV_W, eps, k_scale, v_scale, qk_scale,
                                     HQ, HKV, HD, L, RN, ROPE)
        if HP == 2:
            m1, l1, a1 = _attn3_new_keys(qkv_ptr, inv_freq_ptr, cache_loc_ptr, k_cache_ptr, v_cache_ptr, q1, m1, l1,
                                         a1, tok0, seq, h1, jrow, QKV_W, eps, k_scale, v_scale, qk_scale,
                                         HQ, HKV, HD, L, RN, ROPE)
    for hh in tl.static_range(HP):
        if hh == 0:
            m_i, l_i, acc, h = m0, l0, a0, h0
        else:
            m_i, l_i, acc, h = m1, l1, a1, h1
        base_ws = ((b * HKV + h) * SPLITS + s).to(tl.int64) * R
        tl.store(ws_acc_ptr + (base_ws + rows)[:, None] * HD + d[None, :], acc)
        tl.store(ws_ml_ptr + (base_ws + rows) * 2, m_i)
        tl.store(ws_ml_ptr + (base_ws + rows) * 2 + 1, l_i)


@triton.jit
def _attn3_combine_kernel(qkv_ptr, out_ptr, ws_acc_ptr, ws_ml_ptr, v_scale_ptr, QKV_W, eps,
                          HQ: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, L: tl.constexpr,
                          G: tl.constexpr, R: tl.constexpr, SPLITS: tl.constexpr, SC: tl.constexpr):
    """One program per (query head, request, block token): merges that row's splits in
    vectorized chunks of SC, then RMSNorm x sigmoid(gate)."""
    hq = tl.program_id(0)
    b = tl.program_id(1)
    j = tl.program_id(2)
    h = hq // G
    r = j * G + hq % G
    d = tl.arange(0, HD)
    sc = tl.arange(0, SC)
    m_run = tl.full((1,), float("-inf"), tl.float32)
    l_run = tl.zeros((1,), tl.float32)
    acc = tl.zeros((HD,), tl.float32)
    for c0 in range(0, SPLITS, SC):
        si = c0 + sc
        smask = si < SPLITS
        idx = ((b * HKV + h) * SPLITS + si).to(tl.int64) * R + r
        m_s = tl.load(ws_ml_ptr + idx * 2, mask=smask, other=float("-inf"))
        l_s = tl.load(ws_ml_ptr + idx * 2 + 1, mask=smask, other=0.0)
        a_s = tl.load(ws_acc_ptr + idx[:, None] * HD + d[None, :], mask=smask[:, None], other=0.0)
        m_new = tl.maximum(m_run, tl.max(m_s, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_run - m_safe)
        beta = tl.exp2(m_s - m_safe)
        l_run = l_run * alpha + tl.sum(l_s * beta, axis=0)
        acc = acc * alpha + tl.sum(a_s * beta[:, None], axis=0)
        m_run = m_new
    o = acc / tl.where(l_run == 0.0, 1.0, l_run) * tl.load(v_scale_ptr)
    inv = tl.rsqrt(tl.sum(o * o, axis=0) / HD + eps)
    tok = (b * L + j).to(tl.int64)
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * HD + hq * HD + d).to(tl.float32)
    tl.store(out_ptr + tok * (HQ * HD) + hq * HD + d, (o * inv * tl.sigmoid(gate)).to(tl.bfloat16))


def _attn3(qkv, data, cfg, block_n=64, warps=4, stages=3, per_sm=4, min_chunk=256, hp=2):
    """v3 attention: qkv [T, W] -> [T, Hq * 64]; writes the new cache entries."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, hd = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    page, window = cfg["page_size"], cfg["window"]
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    device = qkv.device
    page_table = data["page_table"]
    capacity = page_table.shape[1] * page
    span = min(window, capacity) if window else capacity
    if hkv % hp:
        hp = 1
    pairs = hkv // hp
    splits = max(1, min(triton.cdiv(per_sm * NUM_SMS, batch * pairs), triton.cdiv(span, min_chunk)))
    chunk = triton.cdiv(triton.cdiv(span, splits), block_n) * block_n
    splits = triton.cdiv(span, chunk)
    rows = max(16, triton.next_power_of_2(group * length))
    rows_new = max(16, triton.next_power_of_2(length))
    ws_acc = torch.empty((batch, hkv, splits, rows, hd), device=device, dtype=torch.float32)
    ws_ml = torch.empty((batch, hkv, splits, rows, 2), device=device, dtype=torch.float32)
    out = torch.empty((batch * length, hq * hd), device=device, dtype=torch.bfloat16)
    inv_freq = _inv_freq(cfg["rope_theta"], hd, device)
    qkv_w = qkv.shape[-1]
    _attn3_split_kernel[(splits, pairs, batch)](
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], ws_acc, ws_ml,
        qkv_w, page_table.stride(0), chunk, cfg["sm_scale"] * LOG2E, cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, HD=hd, L=length, G=group, R=rows, RN=rows_new, PAGE=page, WINDOW=window,
        ROPE=bool(cfg["rope_theta"]), SPLITS=splits, BLOCK_N=block_n, HP=hp, num_warps=warps,
        num_stages=stages)
    _attn3_combine_kernel[(hq, batch, length)](
        qkv, out, ws_acc, ws_ml, data["v_scale"], qkv_w, RMS_EPS,
        HQ=hq, HKV=hkv, HD=hd, L=length, G=group, R=rows, SPLITS=splits,
        SC=min(64, triton.next_power_of_2(splits)), num_warps=4)
    return out


# ---------------------------------------------------------------------------------------
# SGLang's path (baseline.py), for the shapes where FA3 is faster: measured on the judge's
# H200 (check run 6047873c), the fused kernel wins decode steps on sliding layers (18.3 vs
# 25.5 µs at B 1, 23.3 vs 32.1 at B 16, 28.8 vs 38.8 at B 32) and loses verify blocks and
# full-attention layers, where FA3 streams the scattered cache better.

FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _sglang_attention(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


# ---------------------------------------------------------------------------------------
# v4: CUDA, one launch per decode step (okc/attn_cuda/kernel.py, bundled below).

# @bundle attn_cuda/kernel.py ATTN_DEVICE_SRC ATTN_HOST_SRC ATTN_CUDA_SRC ATTN_CUDA_CPP _ATTN_MODULES _ATTN_COUNTERS _ATTN_EMPTY _no_prof attn_cuda_module attn_cuda_supported attn_cuda_decode


def _attn_v4(data, n_ctas=396, stages=3, min_keys=256, prof=None):
    cfg = data["config"]
    batch = cfg["batch"]
    qkv = data["qkv"].reshape(batch, -1)
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    inv_freq = _inv_freq(cfg["rope_theta"], cfg["head_dim"], qkv.device)
    out = attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=n_ctas, stages=stages, min_keys=min_keys, prof=prof)
    return out.view(batch, 1, -1)


def _kernel_v3(data):
    """The previous routing (sim run 3c2e9a7a): sliding decode -> v1, full decode -> v3,
    verify blocks -> FA3."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    if length != 1 or cfg["num_kv_heads"] % 2:
        return _sglang_attention(data)
    qkv = data["qkv"].reshape(batch, -1)
    if cfg["window"]:
        return _fused_attention(qkv, data, cfg).view(batch, 1, -1)
    return _attn3(qkv, data, cfg, block_n=64, warps=4, stages=3, per_sm=4).view(batch, 1, -1)


def kernel_fn(data: Inputs) -> torch.Tensor:
    """Decode steps -> v4 (CUDA); verify blocks -> FA3 for now."""
    cfg = data["config"]
    if attn_cuda_supported(cfg, data["page_table"]):
        return _attn_v4(data)
    return _kernel_v3(data)
