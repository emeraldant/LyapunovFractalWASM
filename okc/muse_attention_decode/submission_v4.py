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

ATTN_DEVICE_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define ATT_TILE 64
#define ATT_INF __int_as_float(0x7f800000)
#define ATT_MAXB 64
#define ATT_STAGE_BYTES (8192 * 2 + 2048 + 512)

struct AttnParams {
  const __nv_bfloat16* qkv; const float* inv_freq; const int32_t* seq_lens; const int32_t* page_table;
  const int64_t* cache_loc; uint8_t* k_cache; uint8_t* v_cache; const float* k_scale; const float* v_scale;
  __nv_bfloat16* out; float* ws_acc; float* ws_ml; float* ws2_acc; float* ws2_ml; int32_t* counters; long long* prof;
  int qkv_w, pt_stride, B, HQ, HKV, page_shift, window, rope, n_ctas, max_groups, L;
  float qk_factor, sm_scale_log2, eps;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t pair16) {
  uint32_t r;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}\n" : "=r"(r) : "r"(pair16));
  return r;
}
__device__ __forceinline__ uint32_t cvt_lo(uint32_t w) { return e4m3x2_f16x2(w & 0xFFFFu); }
__device__ __forceinline__ uint32_t cvt_hi(uint32_t w) { return e4m3x2_f16x2(w >> 16); }

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_f16(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// sin, cos of an fp32 angle (up to ~1e6 rad): Cody-Waite reduction by 2 pi in two parts, then
// the fast intrinsics on [-pi, pi] (abs error ~1e-6, far below the BF16 rounding that follows).
__device__ __forceinline__ void att_sincos(float a, float& sn, float& cn) {
  const float k = rintf(a * 0.15915494309189535f);
  float r = fmaf(-k, 6.28318548202514648f, a);
  r = fmaf(-k, -1.7484555314695172e-07f, r);
  __sincosf(r, &sn, &cn);
}

// 16 B chunk swizzle of a 128 B key row: conflict-free for the K reads (rows 2m, 2m+1 x four
// chunks of one head) and the V reads (rows 2t, t = 0..3, x two chunks).
__device__ __forceinline__ int att_swz(int key) {
  return ((((key >> 1) & 3) ^ ((key & 1) << 1)) << 1);
}

// The CTA owning virtual key u of T split over n CTAs: start(i) = floor(T i / n).
__device__ __forceinline__ int att_cta_of(long long u, long long T, int n) {
  return (int)(((u + 1) * n - 1) / T);
}

#define ATT_GROUP 16

// Optional phase stamps (sim only): prof[cta][k] = clock64 at event k (62: globaltimer at the end,
// 63: at the start).
#define ATT_STAMP(k) do { if (p.prof != nullptr && threadIdx.x == 0) \
  p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + (k)] = clock64(); } while (0)
__device__ __forceinline__ long long att_gtime() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

// Merges `count` partials (m, l, acc) at slots slot0 .. slot0 + count - 1 for this thread's
// (head h, row r, 8 d values from 8 dc): loads four partials ahead of their merges.
__device__ __forceinline__ void att_merge(const float* ws_acc, const float* ws_ml, size_t slot0, int count, int h,
                                          int r, int dc, float& M, float& L, float o[8]) {
  for (int k0 = 0; k0 < count; k0 += 4) {
    float2 ml[4];
    float4 a0[4], a1[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t slot = slot0 + min(k0 + u, count - 1);
      ml[u] = __ldcg(reinterpret_cast<const float2*>(ws_ml) + (slot * 2 + h) * 8 + r);
      const float4* wa = reinterpret_cast<const float4*>(ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc);
      a0[u] = __ldcg(wa); a1[u] = __ldcg(wa + 1);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (k0 + u < count) {
        float mn = fmaxf(M, ml[u].x);
        float fa = ex2(M - mn), fb = ex2(ml[u].x - mn);
        M = mn;
        L = L * fa + ml[u].y * fb;
        o[0] = o[0] * fa + a0[u].x * fb; o[1] = o[1] * fa + a0[u].y * fb; o[2] = o[2] * fa + a0[u].z * fb;
        o[3] = o[3] * fa + a0[u].w * fb; o[4] = o[4] * fa + a1[u].x * fb; o[5] = o[5] * fa + a1[u].y * fb;
        o[6] = o[6] * fa + a1[u].z * fb; o[7] = o[7] * fa + a1[u].w * fb;
      }
    }
  }
}

__device__ __forceinline__ void att_store_partial(float* ws_acc, float* ws_ml, size_t slot, int h, int r, int dc,
                                                  float M, float L, const float o[8]) {
  float* wa = ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc;
  reinterpret_cast<float4*>(wa)[0] = make_float4(o[0], o[1], o[2], o[3]);
  reinterpret_cast<float4*>(wa)[1] = make_float4(o[4], o[5], o[6], o[7]);
  if (dc == 0) reinterpret_cast<float2*>(ws_ml)[(slot * 2 + h) * 8 + r] = make_float2(M, L);
}

template <int STAGES>
__global__ void __launch_bounds__(128) attn_dec_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  // per stage: K [64][128] | V [64][128] | Q [16][64] bf16 | new K, V [2][2][64] bf16
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];
  __shared__ __align__(16) __nv_bfloat16 s_q0[16 * 64];
  __shared__ float s_knew[2][64];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ;
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 63] = att_gtime();
  ATT_STAMP(0);

  // Lengths: each request's attended cached keys n_b (its window) and the prefix of v_b = n_b + 1.
  if (warp == 0) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      int b = b0 + lane;
      int len = b < B ? p.seq_lens[b] : 0;
      int n = p.window > 0 ? min(len, p.window - 1) : len;
      int v = b < B ? n + 1 : 0;
      int x = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (b < B) { s_pref[b] = carry + x - v; s_first[b] = len - n; s_n[b] = n; }
      carry += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) s_pref[B] = carry;
  }
  __syncthreads();
  const long long T = s_pref[B];
  const int N = (int)min((long long)p.n_ctas, T);
  const int cta = blockIdx.x;
  ATT_STAMP(1);
  if (cta >= N) return;
  const long long r0 = T * cta / N, r1 = T * (cta + 1) / N;

  // This CTA's segments: (request, cached range [u0, uc), holds the new token, tiles); warp 0,
  // a lane per request.
  if (warp == 0) {
    int ns = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long lo = 0, hi = 0;
      bool hit = false;
      if (b < B) { lo = s_pref[b]; hi = s_pref[b + 1]; hit = hi > r0 && lo < r1; }
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int i = ns + __popc(m & ((1u << lane) - 1u));
        const int u0 = (int)max(r0 - lo, 0LL), u1 = (int)min(r1 - lo, hi - lo);
        const int n = s_n[b], uc = min(u1, n);
        s_sb[i] = b; s_su0[i] = u0; s_suc[i] = max(uc, u0); s_snew[i] = u1 > n;
        s_stiles[i] = uc > u0 ? (uc - u0 + ATT_TILE - 1) / ATT_TILE : 1;
      }
      ns += __popc(m);
    }
    if (lane == 0) s_nseg = ns;
  }
  __syncthreads();
  ATT_STAMP(2);
  const int nseg = s_nseg;
  int ntiles = 0;
  for (int s = 0; s < nseg; ++s) ntiles += s_stiles[s];

  const uint32_t smem_u32 = (uint32_t)__cvta_generic_to_shared(smem);
  const int pmask = (1 << p.page_shift) - 1;
  const size_t row_bytes = (size_t)HKV * 64;
  // Gather roles, coalesced: lanes 8r..8r+7 copy the eight 16 B granules of one 128 B row, so a
  // warp instruction moves four whole rows (a row per thread made every instruction touch 32
  // rows with 16 B each: DRAM served scattered sectors at ~0.8 TB/s). Thread rows: K and V of
  // keys rs + 16q, q = 0..3.
  const int rs = tid >> 3, gr = tid & 7;

  // Tile iterators (segment, tile within it): page-entry copies (STAGES - 1 tiles ahead of the
  // gathers, into an 8-tile smem ring by 4 B cp.async: nothing waits on them in registers),
  // gathers, compute. One commit group per tile: its gather plus a later tile's page entries.
  int ip_s = 0, ip_k = 0, ip_t = 0, is_s = 0, is_k = 0, is_t = 0;
  auto advance = [&](int& s, int& k, int& tt) { ++tt; if (s < nseg && ++k == s_stiles[s]) { ++s; k = 0; } };
  auto copy_pages = [&]() {
    if (tid < ATT_TILE && ip_s < nseg) {
      const int b = s_sb[ip_s];
      const int u = s_su0[ip_s] + ip_k * ATT_TILE + tid;
      const bool ok = u < s_suc[ip_s];
      const int pos = s_first[b] + (ok ? u : 0);
      const int32_t* src = p.page_table + (size_t)b * p.pt_stride + (pos >> p.page_shift);
      const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_pg[ip_t & 7][tid]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 4 : 0));
    }
    advance(ip_s, ip_k, ip_t);
  };
  auto gather = [&](int stage) {
    if (is_s < nseg) {
      const int s = is_s, k = is_k, b = s_sb[s];
      const uint32_t st = smem_u32 + stage * ATT_STAGE_BYTES;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int key = rs + 16 * q;
        const int u = s_su0[s] + k * ATT_TILE + key;
        const bool ok = u < s_suc[s];
        const int pos = s_first[b] + u;
        const long long row = ok ? ((long long)s_pg[is_t & 7][key] << p.page_shift) + (pos & pmask) : 0;
        const size_t go = row * row_bytes + pair * 128 + gr * 16;
        const uint32_t dst = st + key * 128 + ((gr ^ att_swz(key)) << 4);
        cp_async16(dst, p.k_cache + go, ok ? 16u : 0u);
        cp_async16(dst + 8192, p.v_cache + go, ok ? 16u : 0u);
      }
      if (k == 0 && s > 0) {  // a later request's 16 query heads of this pair (2 KB); its gate rows to L2
        const uint8_t* q = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + pair * 16 * 64);
        cp_async16(st + 16384 + tid * 16, q + tid * 16, 16u);
        if (tid < 16) {
          const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + pair * 16 * 64);
          asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + tid * 128));
        }
      }
      if (k == s_stiles[s] - 1 && s_snew[s] && tid < 32) {  // the new token's K and V rows (bf16)
        const int kv = tid >> 4, c = tid & 15;
        const uint8_t* src2 = reinterpret_cast<const uint8_t*>(
            p.qkv + (size_t)b * p.qkv_w + (HQ + kv * HKV) * 64 + pair * 128);
        cp_async16(st + 16384 + 2048 + kv * 256 + c * 16, src2 + c * 16, 16u);
      }
    }
    advance(is_s, is_k, is_t);
  };

  // Prologue: the first tiles' page entries and the first request's queries in one round trip,
  // then the first STAGES - 1 gathers; the queries are prepared while those are in flight.
  for (int m = 0; m < STAGES - 1; ++m) copy_pages();
  if (nseg > 0) {
    const int b = s_sb[0];
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + pair * 16 * 64);
    cp_async16((uint32_t)__cvta_generic_to_shared(s_q0) + tid * 16, q + tid * 16, 16u);
    if (tid < 16) {
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + pair * 16 * 64);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + tid * 128));
    }
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int m = 0; m < STAGES - 1; ++m) { gather(m); copy_pages(); cp_commit(); }

  ATT_STAMP(3);
  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale;

  uint32_t qa[2][4][2];
  float m_r[2], l_r[2], acc[2][4][4];
  // Queries: RMSNorm x factor (+ RoPE at the token's position), rounded to BF16, as f16 A
  // fragments: k-step c holds d = 16t + 4c + {0,1} (a0) and {2,3} (a2) of head row g.
  auto prep_q = [&](const __nv_bfloat16* qs, int b) {
    const int pos = s_first[b] + s_n[b];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float x[16];
      const uint4* src = reinterpret_cast<const uint4*>(qs + (8 * h + g) * 64 + 16 * t);
      uint4 w0 = src[0], w1 = src[1];
      const __nv_bfloat162* hw0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
      const __nv_bfloat162* hw1 = reinterpret_cast<const __nv_bfloat162*>(&w1);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f0 = __bfloat1622float2(hw0[i]), f1 = __bfloat1622float2(hw1[i]);
        x[2 * i] = f0.x; x[2 * i + 1] = f0.y; x[8 + 2 * i] = f1.x; x[8 + 2 * i + 1] = f1.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      float sc = rsqrtf(ss / 64.f + p.eps) * p.qk_factor;
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] *= sc;
      if (p.rope) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float sn, cn;
          att_sincos((float)pos * __ldg(p.inv_freq + 8 * t + i), sn, cn);
          float e = x[2 * i], o = x[2 * i + 1];
          x[2 * i] = e * cn - o * sn;
          x[2 * i + 1] = o * cn + e * sn;
        }
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        qa[h][c][0] = pack_h2(x[4 * c], x[4 * c + 1]);
        qa[h][c][1] = pack_h2(x[4 * c + 2], x[4 * c + 3]);
      }
      m_r[h] = -ATT_INF; l_r[h] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { acc[h][i][0] = acc[h][i][1] = acc[h][i][2] = acc[h][i][3] = 0.f; }
    }
  };
  if (nseg > 0) prep_q(s_q0, s_sb[0]);

  int cs = 0, ck = 0;  // compute iterator
  for (int j = 0; j < ntiles; ++j) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (j < 16) ATT_STAMP(4 + 2 * j);
    gather((j + STAGES - 1) % STAGES);
    copy_pages();
    cp_commit();

    const uint8_t* st = smem + (j % STAGES) * ATT_STAGE_BYTES;
    const int b = s_sb[cs];
    if (ck == 0 && cs > 0) prep_q(reinterpret_cast<const __nv_bfloat16*>(st + 16384), b);

    const int nvalid = min(s_suc[cs] - (s_su0[cs] + ck * ATT_TILE) - 16 * warp, 16);
    if (nvalid > 0) {
      const uint8_t* sK = st + (16 * warp) * 128;
      const uint8_t* sV = st + 8192 + (16 * warp) * 128;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        // S = Q K^T: two n8 tiles (keys g, g + 8 of this warp's 16)
        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          int key = 8 * nt + g;
          uint4 kw = *reinterpret_cast<const uint4*>(sK + key * 128 + (((4 * h + t) ^ att_swz(key)) << 4));
          uint32_t kws[4] = {kw.x, kw.y, kw.z, kw.w};
          s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
          for (int c = 0; c < 4; ++c)
            mma_f16(s[nt], qa[h][c][0], 0u, qa[h][c][1], 0u, cvt_lo(kws[c]), cvt_hi(kws[c]));
        }
        // online softmax on row g (keys 2t, 2t+1 of each n tile), log2 units
        float x0 = 2 * t < nvalid ? s[0][0] * c_log2 : -ATT_INF;
        float x1 = 2 * t + 1 < nvalid ? s[0][1] * c_log2 : -ATT_INF;
        float x2 = 8 + 2 * t < nvalid ? s[1][0] * c_log2 : -ATT_INF;
        float x3 = 9 + 2 * t < nvalid ? s[1][1] * c_log2 : -ATT_INF;
        float mt = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
        float mn = fmaxf(m_r[h], mt);
        float alpha = ex2(m_r[h] - mn);
        m_r[h] = mn;
        float p0 = ex2(x0 - mn), p1 = ex2(x1 - mn), p2 = ex2(x2 - mn), p3 = ex2(x3 - mn);
        l_r[h] = l_r[h] * alpha + (p0 + p1) + (p2 + p3);
        float al_a = __shfl_sync(0xffffffffu, alpha, 8 * t);
        float al_b = __shfl_sync(0xffffffffu, alpha, 8 * t + 4);
        uint32_t pb0 = pack_h2(p0, p1), pb1 = pack_h2(p2, p3);
        // O^T += V^T P^T: m tile i holds d = 8g + i (rows g) and 8g + 4 + i (rows g + 8)
        const int vo = ((4 * h + (g >> 1)) << 4) | ((g & 1) << 3);
        uint2 va = *reinterpret_cast<const uint2*>(sV + (2 * t) * 128 + (vo ^ (att_swz(2 * t) << 4)));
        uint2 vb = *reinterpret_cast<const uint2*>(sV + (2 * t + 1) * 128 + (vo ^ (att_swz(2 * t + 1) << 4)));
        uint2 vc = *reinterpret_cast<const uint2*>(sV + (2 * t + 8) * 128 + (vo ^ (att_swz(2 * t + 8) << 4)));
        uint2 vd = *reinterpret_cast<const uint2*>(sV + (2 * t + 9) * 128 + (vo ^ (att_swz(2 * t + 9) << 4)));
        uint32_t lo01 = __byte_perm(va.x, vb.x, 0x5140), lo23 = __byte_perm(va.x, vb.x, 0x7362);
        uint32_t hi01 = __byte_perm(va.y, vb.y, 0x5140), hi23 = __byte_perm(va.y, vb.y, 0x7362);
        uint32_t clo01 = __byte_perm(vc.x, vd.x, 0x5140), clo23 = __byte_perm(vc.x, vd.x, 0x7362);
        uint32_t chi01 = __byte_perm(vc.y, vd.y, 0x5140), chi23 = __byte_perm(vc.y, vd.y, 0x7362);
        uint32_t A0[4] = {cvt_lo(lo01), cvt_hi(lo01), cvt_lo(lo23), cvt_hi(lo23)};
        uint32_t A1[4] = {cvt_lo(hi01), cvt_hi(hi01), cvt_lo(hi23), cvt_hi(hi23)};
        uint32_t A2[4] = {cvt_lo(clo01), cvt_hi(clo01), cvt_lo(clo23), cvt_hi(clo23)};
        uint32_t A3[4] = {cvt_lo(chi01), cvt_hi(chi01), cvt_lo(chi23), cvt_hi(chi23)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[h][i][0] *= al_a; acc[h][i][1] *= al_b; acc[h][i][2] *= al_a; acc[h][i][3] *= al_b;
          mma_f16(acc[h][i], A0[i], A1[i], A2[i], A3[i], pb0, pb1);
        }
      }
    }

    if (j < 16) ATT_STAMP(5 + 2 * j);
    if (ck == s_stiles[cs] - 1) {
      ATT_STAMP(40);
      // ---- end of this request's share: the new token, then this CTA's partial ----
      if (s_snew[cs] && warp == 0) {
        const __nv_bfloat16* kn = reinterpret_cast<const __nv_bfloat16*>(st + 16384 + 2048);
        const int pos = s_first[b] + s_n[b];
        const long long loc = p.cache_loc[b];
        // lane: head lane >> 4, d = 4 (lane & 15) .. + 3
        const int hh = lane >> 4, d0 = 4 * (lane & 15);
        float k4[4], v4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          k4[i] = __bfloat162float(kn[hh * 64 + d0 + i]);
          v4[i] = __bfloat162float(kn[128 + hh * 64 + d0 + i]);
        }
        float ss = k4[0] * k4[0] + k4[1] * k4[1] + k4[2] * k4[2] + k4[3] * k4[3];
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        float inv = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) k4[i] *= inv;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            float sn, cn;
            att_sincos((float)pos * __ldg(p.inv_freq + d0 / 2 + i), sn, cn);
            float e = k4[2 * i], o = k4[2 * i + 1];
            k4[2 * i] = e * cn - o * sn;
            k4[2 * i + 1] = o * cn + e * sn;
          }
        }
        uint32_t kq = 0, vq = 0;
        float kd[4], vdq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          __nv_fp8_e4m3 a(k4[i] / k_scale), c(v4[i] / v_scale);
          kq |= (uint32_t)a.__x << (8 * i);
          vq |= (uint32_t)c.__x << (8 * i);
          kd[i] = (float)a; vdq[i] = (float)c;
        }
        size_t off = ((size_t)loc * HKV + 2 * pair + hh) * 64 + d0;
        *reinterpret_cast<uint32_t*>(p.k_cache + off) = kq;
        *reinterpret_cast<uint32_t*>(p.v_cache + off) = vq;
#pragma unroll
        for (int i = 0; i < 4; ++i) { s_knew[hh][d0 + i] = kd[i]; }
        __syncwarp();
        // scores of rows g (query heads 8h + g) against the new key: q holds d = 16t + 4c + {0..3}
        float sc[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a = 0.f;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float2 q01 = __half22float2(*reinterpret_cast<__half2*>(&qa[h][c][0]));
            float2 q23 = __half22float2(*reinterpret_cast<__half2*>(&qa[h][c][1]));
            const float* kk = &s_knew[h][16 * t + 4 * c];
            a += q01.x * kk[0] + q01.y * kk[1] + q23.x * kk[2] + q23.y * kk[3];
          }
          a += __shfl_xor_sync(0xffffffffu, a, 1);
          a += __shfl_xor_sync(0xffffffffu, a, 2);
          sc[h] = a * c_log2;
        }
        __syncwarp();
        // reuse s_knew for the new value (dequantized FP8, v_scale applied at the end)
#pragma unroll
        for (int i = 0; i < 4; ++i) s_knew[hh][d0 + i] = vdq[i];
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float mn = fmaxf(m_r[h], sc[h]);
          float alpha = ex2(m_r[h] - mn), pn = ex2(sc[h] - mn);
          m_r[h] = mn;
          l_r[h] = l_r[h] * alpha + (t == 0 ? pn : 0.f);
          float al_a = __shfl_sync(0xffffffffu, alpha, 8 * t), al_b = __shfl_sync(0xffffffffu, alpha, 8 * t + 4);
          float pa = __shfl_sync(0xffffffffu, pn, 8 * t), pbb = __shfl_sync(0xffffffffu, pn, 8 * t + 4);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float v0 = s_knew[h][8 * g + i], v1 = s_knew[h][8 * g + 4 + i];
            acc[h][i][0] = acc[h][i][0] * al_a + pa * v0;
            acc[h][i][1] = acc[h][i][1] * al_b + pbb * v0;
            acc[h][i][2] = acc[h][i][2] * al_a + pa * v1;
            acc[h][i][3] = acc[h][i][3] * al_b + pbb * v1;
          }
        }
      }
      // warps -> scratch [warp][h][row][66]: m, l, acc[64], in this tile's stage (16.9 of its
      // 18.9 KB; the stage is reissued only after the next iteration's barrier)
      __syncthreads();
      float* s_scr = reinterpret_cast<float*>(smem + (j % STAGES) * ATT_STAGE_BYTES);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float l = l_r[h];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        float* sw = s_scr + ((warp * 2 + h) * 8) * 66;
        if (t == 0) { sw[g * 66] = m_r[h]; sw[g * 66 + 1] = l; }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sw[(2 * t) * 66 + 2 + 8 * g + i] = acc[h][i][0];
          sw[(2 * t + 1) * 66 + 2 + 8 * g + i] = acc[h][i][1];
          sw[(2 * t) * 66 + 2 + 8 * g + 4 + i] = acc[h][i][2];
          sw[(2 * t + 1) * 66 + 2 + 8 * g + 4 + i] = acc[h][i][3];
        }
      }
      __syncthreads();
      {
        const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
        float M = -ATT_INF;
#pragma unroll
        for (int w = 0; w < 4; ++w) M = fmaxf(M, s_scr[((w * 2 + h) * 8 + r) * 66]);
        float L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float* sw = s_scr + ((w * 2 + h) * 8 + r) * 66;
          float f = ex2(sw[0] - M);
          L += sw[1] * f;
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] += sw[2 + 8 * dc + i] * f;
        }
        const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
        att_store_partial(p.ws_acc, p.ws_ml, slot, h, r, dc, M, L, o);
      }
      __threadfence();
      __syncthreads();
      // Merge: the last of each group of up to 16 partials merges its group; the last group
      // (or, with one group, the last partial) merges into the output.
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      const int c = i_last - i_first + 1, ng = (c + ATT_GROUP - 1) / ATT_GROUP;
      const int gi = (cta - i_first) / ATT_GROUP, gs = min(ATT_GROUP, c - gi * ATT_GROUP);
      int* cnt_final = p.counters + b * gridDim.y + pair;
      int* cnt_group = p.counters + B * gridDim.y + (size_t)(b * gridDim.y + pair) * p.max_groups + gi;
      const size_t g_slot0 = ((size_t)pair * B + b) * p.max_groups;
      if (tid == 0) {
        int mode = 0;
        if (ng == 1) {
          if (atomicAdd(cnt_final, 1) == c - 1) { *cnt_final = 0; mode = 2; }
        } else if (atomicAdd(cnt_group, 1) == gs - 1) {
          *cnt_group = 0; mode = 1;
        }
        s_last = mode;
      }
      __syncthreads();
      ATT_STAMP(41);
      const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
      if (s_last == 1) {
        __threadfence();
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        att_merge(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + gi * ATT_GROUP + b, gs, h, r, dc, M, L, o);
        att_store_partial(p.ws2_acc, p.ws2_ml, g_slot0 + gi, h, r, dc, M, L, o);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          int mode = 0;
          if (atomicAdd(cnt_final, 1) == ng - 1) { *cnt_final = 0; mode = 3; }
          s_last = mode;
        }
        __syncthreads();
      }
      if (s_last >= 2) {
        __threadfence();
        const int qh = (2 * pair + h) * 8 + r;
        const uint4 gw = *reinterpret_cast<const uint4*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + qh * 64 + 8 * dc);
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (s_last == 2) att_merge(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, r, dc, M, L, o);
        else att_merge(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, r, dc, M, L, o);
        float sc = v_scale / L, ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[i] *= sc; ss += o[i] * o[i]; }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        ss += __shfl_xor_sync(0xffffffffu, ss, 4);
        float inv = rsqrtf(ss / 64.f + p.eps);
        const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw);
        uint4 ow;
        __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 gf = __bfloat1622float2(gh[i]);
          float y0 = o[2 * i] * inv / (1.f + __expf(-gf.x));
          float y1 = o[2 * i + 1] * inv / (1.f + __expf(-gf.y));
          oh[i] = __floats2bfloat162_rn(y0, y1);
        }
        *reinterpret_cast<uint4*>(p.out + (size_t)b * HQ * 64 + qh * 64 + 8 * dc) = ow;
      }
      __syncthreads();
      ATT_STAMP(42);
      ++cs; ck = 0;
    } else {
      ++ck;
    }
  }
  cp_wait<0>();
  ATT_STAMP(61);
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 62] = att_gtime();
}
"""


ATTN_VER_SRC = r"""
// ---------------------------------------------------------------------------------------
// Verify blocks (L = 2..8 new tokens per request, causal inside the block): the same work split,
// gather and merge as attn_dec_kernel, with 8 L rows per KV head. Warp w owns KV head w >> 1
// and rows 32 (w & 1) .. + 31 (4 tokens x 8 query heads) over all of a tile's keys, so its
// online-softmax state is final for its rows (no cross-warp merge). S = Q K^T in E4M3 (queries
// rounded to E4M3 as FA3 does, the cached K bytes straight from shared memory as the B
// operand: mma m16n8k32); O^T = V^T P^T in f16 as in the decode kernel. The block's own keys
// are normalized, stored, and attended as one extra 16-key step with the causal mask.

__device__ __forceinline__ void mma_e4m3(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t att_e4m3x4(float a, float b, float c, float d) {
  __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Partials of the verify kernel: acc [slot][2 heads][64 d][64 rows], ml [slot][2][64 rows][2].
__device__ __forceinline__ void att_merge_v(const float* ws_acc, const float* ws_ml, size_t slot0, int count, int h,
                                            int row, float& M, float& Lsum, float o[64]) {
  for (int k = 0; k < count; ++k) {
    const size_t slot = slot0 + k;
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(ws_ml) + (slot * 2 + h) * 64 + row);
    const float* wa = ws_acc + (slot * 2 + h) * 4096 + row;
    const float mn = fmaxf(M, ml.x);
    const float ms = mn == -ATT_INF ? 0.f : mn;
    const float fa = ex2(M - ms), fb = ex2(ml.x - ms);
    M = mn;
    Lsum = Lsum * fa + ml.y * fb;
#pragma unroll
    for (int d0 = 0; d0 < 64; d0 += 16) {
      float a[16];
#pragma unroll
      for (int d = 0; d < 16; ++d) a[d] = __ldcg(wa + (d0 + d) * 64);
#pragma unroll
      for (int d = 0; d < 16; ++d) o[d0 + d] = o[d0 + d] * fa + a[d] * fb;
    }
  }
}

__device__ __forceinline__ void att_store_partial_v(float* ws_acc, float* ws_ml, size_t slot, int h, int row,
                                                    float M, float Lsum, const float o[64]) {
  float* wa = ws_acc + (slot * 2 + h) * 4096 + row;
#pragma unroll
  for (int d = 0; d < 64; ++d) wa[d * 64] = o[d];
  reinterpret_cast<float2*>(ws_ml)[(slot * 2 + h) * 64 + row] = make_float2(M, Lsum);
}

template <int STAGES>
__global__ void __launch_bounds__(128, 3) attn_ver_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];   // per stage: K [64][128] | V [64][128]
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ, L = p.L;
  const int hw = warp >> 1, rh = warp & 1;   // this warp's KV head (of the pair) and row half
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 63] = att_gtime();
  ATT_STAMP(0);

  if (warp == 0) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      int b = b0 + lane;
      int len = b < B ? p.seq_lens[b] : 0;
      int n = p.window > 0 ? min(len, p.window - 1) : len;
      int v = b < B ? n + 1 : 0;
      int x = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (b < B) { s_pref[b] = carry + x - v; s_first[b] = len - n; s_n[b] = n; }
      carry += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) s_pref[B] = carry;
  }
  __syncthreads();
  const long long T = s_pref[B];
  const int N = (int)min((long long)p.n_ctas, T);
  const int cta = blockIdx.x;
  ATT_STAMP(1);
  if (cta >= N) return;
  const long long r0 = T * cta / N, r1 = T * (cta + 1) / N;

  if (warp == 0) {
    int ns = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long lo = 0, hi = 0;
      bool hit = false;
      if (b < B) { lo = s_pref[b]; hi = s_pref[b + 1]; hit = hi > r0 && lo < r1; }
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int i = ns + __popc(m & ((1u << lane) - 1u));
        const int u0 = (int)max(r0 - lo, 0LL), u1 = (int)min(r1 - lo, hi - lo);
        const int n = s_n[b], uc = min(u1, n);
        s_sb[i] = b; s_su0[i] = u0; s_suc[i] = max(uc, u0); s_snew[i] = u1 > n;
        s_stiles[i] = uc > u0 ? (uc - u0 + ATT_TILE - 1) / ATT_TILE : 1;
      }
      ns += __popc(m);
    }
    if (lane == 0) s_nseg = ns;
  }
  __syncthreads();
  ATT_STAMP(2);
  const int nseg = s_nseg;
  int ntiles = 0;
  for (int s = 0; s < nseg; ++s) ntiles += s_stiles[s];

  const uint32_t smem_u32 = (uint32_t)__cvta_generic_to_shared(smem);
  const int pmask = (1 << p.page_shift) - 1;
  const size_t row_bytes = (size_t)HKV * 64;
  const int rs = tid >> 3, gr = tid & 7;
  const size_t W = p.qkv_w;

  int ip_s = 0, ip_k = 0, ip_t = 0, is_s = 0, is_k = 0, is_t = 0;
  auto advance = [&](int& s, int& k, int& tt) { ++tt; if (s < nseg && ++k == s_stiles[s]) { ++s; k = 0; } };
  auto copy_pages = [&]() {
    if (tid < ATT_TILE && ip_s < nseg) {
      const int b = s_sb[ip_s];
      const int u = s_su0[ip_s] + ip_k * ATT_TILE + tid;
      const bool ok = u < s_suc[ip_s];
      const int pos = s_first[b] + (ok ? u : 0);
      const int32_t* src = p.page_table + (size_t)b * p.pt_stride + (pos >> p.page_shift);
      const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_pg[ip_t & 7][tid]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 4 : 0));
    }
    advance(ip_s, ip_k, ip_t);
  };
  // The request's gate rows and new K / V rows to L2 when its first tile is gathered.
  auto prefetch_req = [&](int b) {
    if (tid < 16 * L) {
      const int j = tid >> 4, c = tid & 15;
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + 2 * HKV) * 64 + pair * 1024);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + c * 128));
      if (c < 4) {
        const uint8_t* kp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + (c >> 1) * HKV) * 64 + pair * 128);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(kp + (c & 1) * 128));
      }
    }
  };
  auto gather = [&](int stage) {
    if (is_s < nseg) {
      const int s = is_s, k = is_k, b = s_sb[s];
      const uint32_t st = smem_u32 + stage * 16384;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int key = rs + 16 * q;
        const int u = s_su0[s] + k * ATT_TILE + key;
        const bool ok = u < s_suc[s];
        const int pos = s_first[b] + u;
        const long long row = ok ? ((long long)s_pg[is_t & 7][key] << p.page_shift) + (pos & pmask) : 0;
        const size_t go = row * row_bytes + pair * 128 + gr * 16;
        const uint32_t dst = st + key * 128 + ((gr ^ att_swz(key)) << 4);
        cp_async16(dst, p.k_cache + go, ok ? 16u : 0u);
        cp_async16(dst + 8192, p.v_cache + go, ok ? 16u : 0u);
      }
      if (k == 0 && s > 0) prefetch_req(b);
    }
    advance(is_s, is_k, is_t);
  };

  for (int m = 0; m < STAGES - 1; ++m) copy_pages();
  if (nseg > 0) prefetch_req(s_sb[0]);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int m = 0; m < STAGES - 1; ++m) { gather(m); copy_pages(); cp_commit(); }
  ATT_STAMP(3);

  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale * p.qk_factor;  // queries are E4M3 of q / factor

  // Per thread: rows (mi, hf) = warp rows 16 mi + 8 hf + g: token 4 rh + 2 mi + hf, query head g.
  uint32_t qw[2][2][4];
  float m_r[2][2], l_r[2][2], o[4][4][4];
  int seq = 0;
  auto prep_q = [&](int b) {
    seq = s_first[b] + s_n[b];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int j = 4 * rh + 2 * mi + hf;
        float x[16];
        if (j < L) {
          const uint4* src = reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + j) * W + (pair * 16 + 8 * hw + g) * 64 + 16 * t);
          uint4 w0 = __ldg(src), w1 = __ldg(src + 1);
          const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
          const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&w1);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float2 f0 = __bfloat1622float2(h0[i]), f1 = __bfloat1622float2(h1[i]);
            x[2 * i] = f0.x; x[2 * i + 1] = f0.y; x[8 + 2 * i] = f1.x; x[8 + 2 * i + 1] = f1.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) x[i] = 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float sc = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] *= sc;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float sn, cn;
            att_sincos((float)(seq + j) * __ldg(p.inv_freq + 8 * t + i), sn, cn);
            float e = x[2 * i], od = x[2 * i + 1];
            x[2 * i] = e * cn - od * sn;
            x[2 * i + 1] = od * cn + e * sn;
          }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) qw[mi][hf][c] = att_e4m3x4(x[4 * c], x[4 * c + 1], x[4 * c + 2], x[4 * c + 3]);
        m_r[mi][hf] = -ATT_INF; l_r[mi][hf] = 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) o[i][ni][0] = o[i][ni][1] = o[i][ni][2] = o[i][ni][3] = 0.f;
  };
  if (nseg > 0) prep_q(s_sb[0]);

  // One 16-key step of this warp over keys kb .. kb + 15 of a K / V tile. nvalid: valid keys;
  // mode 1: the block's own keys (causal); wpos >= 0: absolute position of key kb, for the
  // window's lower edge (row j sees positions > seq + j - window).
  auto substep = [&](const uint8_t* sK, const uint8_t* sV, int kb, int nvalid, int mode, int wpos) {
    float s[2][2][4];
    uint4 kw[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int key = kb + 8 * nt + g;
      kw[nt] = *reinterpret_cast<const uint4*>(sK + key * 128 + (((4 * hw + t) ^ att_swz(key)) << 4));
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        s[mi][nt][0] = s[mi][nt][1] = s[mi][nt][2] = s[mi][nt][3] = 0.f;
        mma_e4m3(s[mi][nt], qw[mi][0][0], qw[mi][1][0], qw[mi][0][1], qw[mi][1][1], kw[nt].x, kw[nt].y);
        mma_e4m3(s[mi][nt], qw[mi][0][2], qw[mi][1][2], qw[mi][0][3], qw[mi][1][3], kw[nt].z, kw[nt].w);
      }
    float al[2][2];
    bool grow = false;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int j = 4 * rh + 2 * mi + hf;
        float mt = -ATT_INF;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int k = 8 * nt + 2 * t + e;
            bool ok = k < nvalid;
            if (mode == 1) ok = ok && k <= j;
            if (wpos >= 0) ok = ok && wpos + k > seq + j - p.window;
            const float x = ok ? s[mi][nt][2 * hf + e] * c_log2 : -ATT_INF;
            s[mi][nt][2 * hf + e] = x;
            mt = fmaxf(mt, x);
          }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
        const float mn = fmaxf(m_r[mi][hf], mt);
        const float ms = mn == -ATT_INF ? 0.f : mn;
        al[mi][hf] = ex2(m_r[mi][hf] - ms);
        grow |= mn > m_r[mi][hf];
        m_r[mi][hf] = mn;
        float ls = 0.f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float pv = ex2(s[mi][nt][2 * hf + e] - ms);
            s[mi][nt][2 * hf + e] = pv;
            ls += pv;
          }
        l_r[mi][hf] = l_r[mi][hf] * al[mi][hf] + ls;
      }
    if (__any_sync(0xffffffffu, grow)) {
      // O^T column (row) 8 ni + 2t (+1) belongs to S row g = 2t (+1) of (mi, hf) = (ni >> 1, ni & 1)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const float a0 = __shfl_sync(0xffffffffu, al[ni >> 1][ni & 1], 8 * t);
        const float a1 = __shfl_sync(0xffffffffu, al[ni >> 1][ni & 1], 8 * t + 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) { o[i][ni][0] *= a0; o[i][ni][1] *= a1; o[i][ni][2] *= a0; o[i][ni][3] *= a1; }
      }
    }
    uint32_t pb0[4], pb1[4];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int mi = ni >> 1, hf = ni & 1;
      pb0[ni] = pack_h2(s[mi][0][2 * hf], s[mi][0][2 * hf + 1]);
      pb1[ni] = pack_h2(s[mi][1][2 * hf], s[mi][1][2 * hf + 1]);
    }
    const int vo = ((4 * hw + (g >> 1)) << 4) | ((g & 1) << 3);
    const uint8_t* sVb = sV + kb * 128;
    uint2 va = *reinterpret_cast<const uint2*>(sVb + (2 * t) * 128 + (vo ^ (att_swz(2 * t) << 4)));
    uint2 vb = *reinterpret_cast<const uint2*>(sVb + (2 * t + 1) * 128 + (vo ^ (att_swz(2 * t + 1) << 4)));
    uint2 vc = *reinterpret_cast<const uint2*>(sVb + (2 * t + 8) * 128 + (vo ^ (att_swz(2 * t + 8) << 4)));
    uint2 vd = *reinterpret_cast<const uint2*>(sVb + (2 * t + 9) * 128 + (vo ^ (att_swz(2 * t + 9) << 4)));
    uint32_t lo01 = __byte_perm(va.x, vb.x, 0x5140), lo23 = __byte_perm(va.x, vb.x, 0x7362);
    uint32_t hi01 = __byte_perm(va.y, vb.y, 0x5140), hi23 = __byte_perm(va.y, vb.y, 0x7362);
    uint32_t clo01 = __byte_perm(vc.x, vd.x, 0x5140), clo23 = __byte_perm(vc.x, vd.x, 0x7362);
    uint32_t chi01 = __byte_perm(vc.y, vd.y, 0x5140), chi23 = __byte_perm(vc.y, vd.y, 0x7362);
    uint32_t A0[4] = {cvt_lo(lo01), cvt_hi(lo01), cvt_lo(lo23), cvt_hi(lo23)};
    uint32_t A1[4] = {cvt_lo(hi01), cvt_hi(hi01), cvt_lo(hi23), cvt_hi(hi23)};
    uint32_t A2[4] = {cvt_lo(clo01), cvt_hi(clo01), cvt_lo(clo23), cvt_hi(clo23)};
    uint32_t A3[4] = {cvt_lo(chi01), cvt_hi(chi01), cvt_lo(chi23), cvt_hi(chi23)};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) mma_f16(o[i][ni], A0[i], A1[i], A2[i], A3[i], pb0[ni], pb1[ni]);
  };

  int cs = 0, ck = 0;
  for (int j = 0; j < ntiles; ++j) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (j < 16) ATT_STAMP(4 + 2 * j);
    gather((j + STAGES - 1) % STAGES);
    copy_pages();
    cp_commit();

    uint8_t* st = smem + (j % STAGES) * 16384;
    const int b = s_sb[cs];
    if (ck == 0 && cs > 0) prep_q(b);
    const int u0 = s_su0[cs] + ck * ATT_TILE;
    const int nv_tile = s_suc[cs] - u0;
    const int pos0 = s_first[b] + u0;
    const int wpos = (p.window > 0 && pos0 < seq + L - p.window) ? pos0 : -1;
#pragma unroll 1
    for (int sub = 0; sub < 4; ++sub) {
      const int nv = min(nv_tile - 16 * sub, 16);
      if (nv > 0) substep(st, st + 8192, 16 * sub, nv, 0, wpos >= 0 ? wpos + 16 * sub : -1);
    }
    if (j < 16) ATT_STAMP(5 + 2 * j);

    if (ck == s_stiles[cs] - 1) {
      ATT_STAMP(40);
      if (s_snew[cs]) {
        // The block's own keys: normalized (+ RoPE), FP8, stored to the cache and to rows 0..15
        // of this (consumed) stage, then one causal step. Thread: token tid >> 4, head
        // (tid >> 3) & 1, d 8 (tid & 7) .. + 7.
        __syncthreads();
        const int jt = tid >> 4, hh = (tid >> 3) & 1, dc = tid & 7;
        float kx[8], vx[8];
        if (jt < L) {
          const uint4 kr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + 2 * pair + hh) * 64 + 8 * dc));
          const uint4 vr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + HKV + 2 * pair + hh) * 64 + 8 * dc));
          const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kr);
          const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vr);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float2 a = __bfloat1622float2(kh[i]), c = __bfloat1622float2(vh[i]);
            kx[2 * i] = a.x; kx[2 * i + 1] = a.y; vx[2 * i] = c.x; vx[2 * i + 1] = c.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) { kx[i] = 0.f; vx[i] = 0.f; }
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += kx[i] * kx[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        ss += __shfl_xor_sync(0xffffffffu, ss, 4);
        const float inv = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) kx[i] *= inv;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float sn, cn;
            att_sincos((float)(seq + jt) * __ldg(p.inv_freq + 4 * dc + i), sn, cn);
            float e = kx[2 * i], od = kx[2 * i + 1];
            kx[2 * i] = e * cn - od * sn;
            kx[2 * i + 1] = od * cn + e * sn;
          }
        }
        const float ik = 1.f / k_scale, iv = 1.f / v_scale;
        uint2 kq, vq;
        kq.x = att_e4m3x4(kx[0] * ik, kx[1] * ik, kx[2] * ik, kx[3] * ik);
        kq.y = att_e4m3x4(kx[4] * ik, kx[5] * ik, kx[6] * ik, kx[7] * ik);
        vq.x = att_e4m3x4(vx[0] * iv, vx[1] * iv, vx[2] * iv, vx[3] * iv);
        vq.y = att_e4m3x4(vx[4] * iv, vx[5] * iv, vx[6] * iv, vx[7] * iv);
        if (jt < L) {
          const long long loc = p.cache_loc[(size_t)b * L + jt];
          const size_t off = ((size_t)loc * HKV + 2 * pair + hh) * 64 + 8 * dc;
          *reinterpret_cast<uint2*>(p.k_cache + off) = kq;
          *reinterpret_cast<uint2*>(p.v_cache + off) = vq;
        } else {
          kq = make_uint2(0u, 0u); vq = make_uint2(0u, 0u);
        }
        // rows jt (0..7) and jt + 8 (zero) of the stage
        const int so = ((((4 * hh + (dc >> 1)) ^ att_swz(jt)) << 4) | ((dc & 1) << 3));
        *reinterpret_cast<uint2*>(st + jt * 128 + so) = kq;
        *reinterpret_cast<uint2*>(st + 8192 + jt * 128 + so) = vq;
        const int so8 = ((((4 * hh + (dc >> 1)) ^ att_swz(jt + 8)) << 4) | ((dc & 1) << 3));
        *reinterpret_cast<uint2*>(st + (jt + 8) * 128 + so8) = make_uint2(0u, 0u);
        *reinterpret_cast<uint2*>(st + 8192 + (jt + 8) * 128 + so8) = make_uint2(0u, 0u);
        __syncthreads();
        substep(st, st + 8192, 0, L, 1, -1);
      }
      // this warp's rows -> partial slot (cta + b): acc [d][row], ml [row]
      const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
      {
        float* wa = p.ws_acc + (slot * 2 + hw) * 4096 + 32 * rh;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int rr = 8 * ni + 2 * t;
            *reinterpret_cast<float2*>(wa + (8 * g + i) * 64 + rr) = make_float2(o[i][ni][0], o[i][ni][1]);
            *reinterpret_cast<float2*>(wa + (8 * g + 4 + i) * 64 + rr) = make_float2(o[i][ni][2], o[i][ni][3]);
          }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            float l = l_r[mi][hf];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            if (t == 0)
              reinterpret_cast<float2*>(p.ws_ml)[(slot * 2 + hw) * 64 + 32 * rh + 16 * mi + 8 * hf + g] = make_float2(m_r[mi][hf], l);
          }
      }
      __threadfence();
      __syncthreads();
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      const int c = i_last - i_first + 1, ng = (c + ATT_GROUP - 1) / ATT_GROUP;
      const int gi = (cta - i_first) / ATT_GROUP, gs = min(ATT_GROUP, c - gi * ATT_GROUP);
      int* cnt_final = p.counters + b * gridDim.y + pair;
      int* cnt_group = p.counters + B * gridDim.y + (size_t)(b * gridDim.y + pair) * p.max_groups + gi;
      const size_t g_slot0 = ((size_t)pair * B + b) * p.max_groups;
      if (tid == 0) {
        int mode = 0;
        if (ng == 1) {
          if (atomicAdd(cnt_final, 1) == c - 1) { *cnt_final = 0; mode = 2; }
        } else if (atomicAdd(cnt_group, 1) == gs - 1) {
          *cnt_group = 0; mode = 1;
        }
        s_last = mode;
      }
      __syncthreads();
      ATT_STAMP(41);
      const int h = tid >> 6, row = tid & 63;
      if (s_last == 1) {
        __threadfence();
        float M = -ATT_INF, Ls = 0.f, oo[64];
#pragma unroll
        for (int d = 0; d < 64; ++d) oo[d] = 0.f;
        att_merge_v(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + gi * ATT_GROUP + b, gs, h, row, M, Ls, oo);
        att_store_partial_v(p.ws2_acc, p.ws2_ml, g_slot0 + gi, h, row, M, Ls, oo);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          int mode = 0;
          if (atomicAdd(cnt_final, 1) == ng - 1) { *cnt_final = 0; mode = 3; }
          s_last = mode;
        }
        __syncthreads();
      }
      if (s_last >= 2) {
        __threadfence();
        float M = -ATT_INF, Ls = 0.f, oo[64];
#pragma unroll
        for (int d = 0; d < 64; ++d) oo[d] = 0.f;
        if (s_last == 2) att_merge_v(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, row, M, Ls, oo);
        else att_merge_v(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, row, M, Ls, oo);
        const int jt = row >> 3, qh = (2 * pair + h) * 8 + (row & 7);
        if (jt < L) {
          const float sc = v_scale / Ls;
          float ss = 0.f;
#pragma unroll
          for (int d = 0; d < 64; ++d) { oo[d] *= sc; ss += oo[d] * oo[d]; }
          const float inv = rsqrtf(ss / 64.f + p.eps);
          const size_t tok = (size_t)b * L + jt;
          const uint4* gp = reinterpret_cast<const uint4*>(p.qkv + tok * W + (HQ + 2 * HKV) * 64 + qh * 64);
          uint4* op = reinterpret_cast<uint4*>(p.out + tok * HQ * 64 + qh * 64);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint4 gw = gp[q];
            const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw);
            uint4 ow;
            __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 gf = __bfloat1622float2(gh[i]);
              oh[i] = __floats2bfloat162_rn(oo[8 * q + 2 * i] * inv / (1.f + __expf(-gf.x)),
                                            oo[8 * q + 2 * i + 1] * inv / (1.f + __expf(-gf.y)));
            }
            op[q] = ow;
          }
        }
      }
      __syncthreads();
      ATT_STAMP(42);
      ++cs; ck = 0;
    } else {
      ++ck;
    }
  }
  cp_wait<0>();
  ATT_STAMP(61);
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 62] = att_gtime();
}
"""


ATTN_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <int S>
static void launch_dec(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * ATT_STAGE_BYTES;
  auto kern = attn_dec_kernel<S>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

template <int S>
static void launch_ver(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * 16384;
  auto kern = attn_ver_kernel<S>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages,
              int64_t L) {
  AttnParams p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.inv_freq = inv_freq.data_ptr<float>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.page_table = page_table.data_ptr<int32_t>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_ml = ws_ml.data_ptr<float>();
  p.ws2_acc = ws2_acc.data_ptr<float>();
  p.ws2_ml = ws2_ml.data_ptr<float>();
  p.counters = counters.data_ptr<int32_t>();
  p.max_groups = (int)max_groups;
  p.prof = prof.numel() > 0 ? reinterpret_cast<long long*>(prof.data_ptr()) : nullptr;
  p.qkv_w = (int)qkv.size(-1);
  p.pt_stride = (int)page_table.stride(0);
  p.B = (int)B; p.HQ = (int)HQ; p.HKV = (int)HKV; p.page_shift = (int)page_shift; p.window = (int)window;
  p.rope = (int)rope; p.n_ctas = (int)n_ctas; p.L = (int)L;
  p.qk_factor = (float)qk_factor; p.sm_scale_log2 = (float)sm_scale_log2; p.eps = (float)eps;
  int pairs = (int)HKV / 2;
  if (L > 1) {
    if (stages == 3) launch_ver<3>(p, (int)n_ctas, pairs);
    else if (stages == 4) launch_ver<4>(p, (int)n_ctas, pairs);
    else TORCH_CHECK(false, "attn_dec: unsupported stages for L > 1");
    return;
  }
  if (stages == 3) launch_dec<3>(p, (int)n_ctas, pairs);
  else if (stages == 4) launch_dec<4>(p, (int)n_ctas, pairs);
  else if (stages == 5) launch_dec<5>(p, (int)n_ctas, pairs);
  else if (stages == 6) launch_dec<6>(p, (int)n_ctas, pairs);
  else TORCH_CHECK(false, "attn_dec: unsupported stages");
}
"""


ATTN_CUDA_SRC = ATTN_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n",
                                      "#include <ATen/cuda/CUDAContext.h>\n" + ATTN_DEVICE_SRC + ATTN_VER_SRC, 1)


ATTN_CUDA_CPP = """
void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages,
              int64_t L);
"""


_ATTN_MODULES = {}


_ATTN_COUNTERS = {}


_ATTN_EMPTY = {}


def _no_prof(device):
    t = _ATTN_EMPTY.get(device)
    if t is None:
        t = _ATTN_EMPTY[device] = torch.empty(0, dtype=torch.int64, device=device)
    return t


def attn_cuda_module():
    if "m" not in _ATTN_MODULES:
        from torch.utils.cpp_extension import load_inline
        _ATTN_MODULES["m"] = load_inline(
            name="okc_attn_dec", cpp_sources=ATTN_CUDA_CPP, cuda_sources=ATTN_CUDA_SRC, functions=["attn_dec"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _ATTN_MODULES["m"]


def attn_cuda_supported(cfg, page_table):
    """Shapes the CUDA path covers (others take the Triton / SGLang paths)."""
    page = cfg["page_size"]
    return (1 <= cfg["q_len"] <= 8 and cfg["head_dim"] == 64 and cfg["num_kv_heads"] % 2 == 0
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["batch"] <= 64
            and page & (page - 1) == 0)


def attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=264, stages=4, min_keys=256, prof=None):
    """qkv [B * L, W] bf16 -> [B * L, Hq * 64] bf16; writes the new cache entries. n_ctas: CTAs
    per KV-head pair at most (fewer when the batch has fewer than min_keys keys per CTA)."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    pairs = hkv // 2
    page = cfg["page_size"]
    window = cfg["window"]
    page_table = data["page_table"]
    capacity = page_table.shape[1] * page
    span = min(window, capacity) if window else capacity
    n = max(1, min(max(1, n_ctas // pairs), -(-batch * span // min_keys)))
    device = qkv.device
    max_groups = n // 16 + 2
    key = (device, batch * pairs * (1 + max_groups))
    counters = _ATTN_COUNTERS.get(key)
    if counters is None:
        counters = _ATTN_COUNTERS[key] = torch.zeros(key[1], dtype=torch.int32, device=device)
    slots = pairs * (n + batch)
    rows = 8 if length == 1 else 64
    acc_shape = (2, 8, 64) if length == 1 else (2, 64, 64)
    ws_acc = torch.empty((slots,) + acc_shape, dtype=torch.float32, device=device)
    ws_ml = torch.empty((slots, 2, rows, 2), dtype=torch.float32, device=device)
    ws2_acc = torch.empty((pairs * batch * max_groups,) + acc_shape, dtype=torch.float32, device=device)
    ws2_ml = torch.empty((pairs * batch * max_groups, 2, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch * length, hq * 64), dtype=torch.bfloat16, device=device)
    attn_cuda_module().attn_dec(
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], out, ws_acc, ws_ml, ws2_acc, ws2_ml, counters,
        prof if prof is not None else _no_prof(device), max_groups, batch, hq, hkv, page.bit_length() - 1,
        window, int(bool(cfg["rope_theta"])), cfg["qk_scale_factor"], cfg["sm_scale"] * 1.4426950408889634, 1e-5,
        n, stages, length)
    return out




def _attn_v4(data, n_ctas=396, stages=3, min_keys=256, prof=None):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = data["qkv"].reshape(batch * length, -1)
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    inv_freq = _inv_freq(cfg["rope_theta"], cfg["head_dim"], qkv.device)
    out = attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=n_ctas, stages=stages, min_keys=min_keys, prof=prof)
    return out.view(batch, length, -1)


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
