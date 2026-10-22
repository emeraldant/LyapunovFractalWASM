"""Memory-first NVFP4 MoE experts for decode-sized batches (CUDA via load_inline + Triton).

Why: Marlin streams the experts at ~2 TB/s (judge profiles); HBM gives 4.06 TB/s. The
NVFP4 unpack runs on the 64-lane INT pipe, so it must stay ~2-3 INT ops per weight and the
unpacked values must land directly in mma.sync register fragments (mma_probe.py checks the
building blocks on the judge's GPU: 1.4e-7 relative error).

Pipeline per call (4 launches + 1 memset):
  1. _route_compact_kernel (Triton): local routes sorted by expert (compact, no padding),
     per-expert offsets, the active-expert list, and route <-> row maps.
  2. moe_gemm<1>: h = x @ W13^T, fused silu(gate) * up -> inter [rows, 4096] BF16.
  3. moe_gemm<2>: y = inter @ W2^T -> [rows, 4096] BF16.
  4. _post_norm_rows_kernel (Triton): out[t] = sum_k s[t,k] * RMSNorm(y[row(t,k)] * w).

Layouts (built once in prepare, lossless: same E2M1 codes and E4M3 scales):
  weights: [expert][n8 group][k chunk of 128][lane 32][4 x u32], each lane's 16 bytes holding
    its m16n8k16 B-fragment codes for the chunk's 8 k16 steps (Marlin's nibble order);
    GEMM1 groups alternate gate / up so a warp's group pair gives silu(gate) * up directly.
  scales: [expert][n8 group][k chunk][row 8][step 8] E4M3 bytes.
"""
import torch
import triton
import triton.language as tl

MOE_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define NG 4          // n8 groups per CTA
#define KW 8          // warps per CTA, each a K slice
#define KTOT 4096
#define CHUNKS_PER_WARP (KTOT / 128 / KW)   // 4

__device__ __forceinline__ void unpack8(uint32_t q, uint32_t out[4]) {
  const uint32_t M = 0x70007000u, S = 0x80008000u;
  uint32_t a = q;
  out[0] = (a & S) | ((a & M) >> 6);
  a <<= 4;
  out[1] = (a & S) | ((a & M) >> 6);
  uint32_t b = q << 8;
  out[2] = (b & S) | ((b & M) >> 6);
  b <<= 4;
  out[3] = (b & S) | ((b & M) >> 6);
}

// E4M3 byte -> bf16x2 (both halves) = scale * 2^118 (normal scales; prepare checks).
__device__ __forceinline__ uint32_t scale2(uint32_t s) {
  uint32_t bits = (((s & 0x80u) << 8) | ((s & 0x7Fu) << 4)) + (238u << 7);
  return bits | (bits << 16);
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_stream16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ldg_stream8(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// MODE 1: inter[row, 8p + c] = silu(h_gate) * h_up, groups (gate p0, up p0, gate p1, up p1).
// MODE 2: y[row, 8g + c] = h, groups g = 4 nb .. 4 nb + 3.
// a_rows: GEMM1 -> x [M, 4096] indexed by token (route / 8); GEMM2 -> inter [rows, 4096].
template <int MODE, int MT>
__global__ void __launch_bounds__(256) moe_gemm_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint32_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, const __nv_bfloat16* __restrict__ zero_row,
    __nv_bfloat16* __restrict__ out, int n_blocks, int m_chunks) {
  extern __shared__ float red[];   // [KW][MT][NG][32 lanes][4]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = lane / 4, c = (lane % 4) * 2;
  const int n_active = *n_active_ptr;
  const int total = n_active * n_blocks * m_chunks;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int mc = item % m_chunks;
    const int nb = (item / m_chunks) % n_blocks;
    const int e = active[item / (m_chunks * n_blocks)];
    const int row0 = exp_start[e] + mc * MT * 16;
    const int rows_e = exp_start[e + 1] - row0;
    if (rows_e <= 0) continue;
    const int tiles = min(MT, (rows_e + 15) / 16);
    const float g = gs[e] * 256.0f;   // global scale x 2^8 (the B values carry 2^-8)

    // A-fragment row pointers for this lane: rows r and r + 8 of each tile.
    const uint32_t* arow[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int lr = mt * 16 + r + 8 * h;
        const __nv_bfloat16* p = zero_row;
        if (lr < rows_e) {
          int src = MODE == 1 ? row_route[row0 + lr] / 8 : row0 + lr;
          p = a_src + (size_t)src * KTOT;
        }
        arow[mt][h] = reinterpret_cast<const uint32_t*>(p);
      }
    }
    float acc[MT][NG][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int ng = 0; ng < NG; ++ng)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][ng][i] = 0.f;

    // Group base offsets (in chunks) for this expert and n-block.
    // Groups per expert: MODE 1 has 1024 (512 gate/up pairs, interleaved), MODE 2 has 512.
    size_t gbase[NG];
#pragma unroll
    for (int ng = 0; ng < NG; ++ng) {
      size_t grp = MODE == 1 ? (size_t)(2 * (2 * nb + ng / 2) + (ng % 2)) : (size_t)(4 * nb + ng);
      gbase[ng] = ((size_t)e * (MODE == 1 ? 1024 : 512) + grp) * (KTOT / 128);
    }
    const int kc0 = warp * CHUNKS_PER_WARP;
    uint4 wv[NG];
    uint2 sv[NG];
#pragma unroll
    for (int ng = 0; ng < NG; ++ng) {
      size_t ch = gbase[ng] + kc0;
      wv[ng] = ldg_stream16(wq + (ch * 32 + lane) * 4);
      sv[ng] = ldg_stream8(ws + ch * 64 + r * 8);
    }
#pragma unroll 1
    for (int i = 0; i < CHUNKS_PER_WARP; ++i) {
      uint4 cw[NG];
      uint2 cs[NG];
#pragma unroll
      for (int ng = 0; ng < NG; ++ng) { cw[ng] = wv[ng]; cs[ng] = sv[ng]; }
      if (i + 1 < CHUNKS_PER_WARP) {
#pragma unroll
        for (int ng = 0; ng < NG; ++ng) {
          size_t ch = gbase[ng] + kc0 + i + 1;
          wv[ng] = ldg_stream16(wq + (ch * 32 + lane) * 4);
          sv[ng] = ldg_stream8(ws + ch * 64 + r * 8);
        }
      }
      const int kbase = (kc0 + i) * 128;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        uint32_t v[NG][4];
#pragma unroll
        for (int ng = 0; ng < NG; ++ng) {
          uint32_t q = w == 0 ? cw[ng].x : w == 1 ? cw[ng].y : w == 2 ? cw[ng].z : cw[ng].w;
          unpack8(q, v[ng]);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int step = 2 * w + half;
          const int k0 = kbase + step * 16;
          uint32_t b0[NG], b1[NG];
#pragma unroll
          for (int ng = 0; ng < NG; ++ng) {
            uint32_t sb = (step < 4 ? (cs[ng].x >> (8 * step)) : (cs[ng].y >> (8 * (step - 4)))) & 0xFFu;
            uint32_t s2 = scale2(sb);
            b0[ng] = mul2(v[ng][2 * half], s2);
            b1[ng] = mul2(v[ng][2 * half + 1], s2);
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            if (mt < tiles) {
              uint32_t a[4];
              a[0] = arow[mt][0][(k0 + c) / 2];
              a[1] = arow[mt][1][(k0 + c) / 2];
              a[2] = arow[mt][0][(k0 + c + 8) / 2];
              a[3] = arow[mt][1][(k0 + c + 8) / 2];
#pragma unroll
              for (int ng = 0; ng < NG; ++ng) mma16816(acc[mt][ng], a, b0[ng], b1[ng]);
            }
          }
        }
      }
    }
    // Reduce the KW warps' partial sums through shared memory.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int ng = 0; ng < NG; ++ng)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          red[((((warp * MT + mt) * NG + ng) * 32 + lane) * 4) + i] = acc[mt][ng][i];
    __syncthreads();
    // Each thread finishes (mt, lane, i) outputs for a pair of groups (MODE 1) or groups.
    const int per_warp = MT * NG * 32 * 4;
    for (int idx = threadIdx.x; idx < MT * 32 * 4 * (MODE == 1 ? NG / 2 : NG); idx += blockDim.x) {
      int i = idx % 4, ln = (idx / 4) % 32;
      int rest = idx / 128;
      int mt, ng;
      if (MODE == 1) { int pr = rest % (NG / 2); mt = rest / (NG / 2); ng = 2 * pr; }
      else { ng = rest % NG; mt = rest / NG; }
      if (mt >= tiles) continue;
      int orow = mt * 16 + ln / 4 + (i >= 2 ? 8 : 0);
      int ocol = (ln % 4) * 2 + (i & 1);
      if (orow >= rows_e) continue;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        s0 += red[kw * per_warp + ((mt * NG + ng) * 32 + ln) * 4 + i];
        if (MODE == 1) s1 += red[kw * per_warp + ((mt * NG + ng + 1) * 32 + ln) * 4 + i];
      }
      float val;
      int n;
      if (MODE == 1) {
        float gate = s0 * g, up = s1 * g;
        val = gate / (1.f + __expf(-gate)) * up;
        n = 8 * (2 * nb + ng / 2) + ocol;
      } else {
        val = s0 * g;
        n = 8 * (4 * nb + ng) + ocol;
      }
      out[(size_t)(row0 + orow) * KTOT + n] = __float2bfloat16(val);
    }
    __syncthreads();
  }
}

template <int MODE, int MT>
void launch(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
            torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor zero_row,
            torch::Tensor out, int n_blocks, int m_chunks, int grid) {
  size_t smem = (size_t)KW * MT * NG * 32 * 4 * sizeof(float);
  auto k = moe_gemm_kernel<MODE, MT>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), row_route.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), ws.data_ptr<uint8_t>(), gs.data_ptr<float>(),
      active.data_ptr<int32_t>(), n_active.data_ptr<int32_t>(), exp_start.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(zero_row.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, m_chunks);
}

void moe_gemm(int64_t mode, int64_t mt, torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq,
              torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor zero_row, torch::Tensor out, int64_t n_blocks,
              int64_t m_chunks, int64_t grid) {
#define CASE(M_, T_) if (mode == M_ && mt == T_) { launch<M_, T_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, zero_row, out, n_blocks, m_chunks, grid); return; }
  CASE(1, 1) CASE(1, 2) CASE(1, 4) CASE(2, 1) CASE(2, 2) CASE(2, 4)
  TORCH_CHECK(false, "moe_gemm: unsupported mode / m tiles");
}
"""

MOE_CUDA_CPP = """
void moe_gemm(int64_t mode, int64_t mt, torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq,
              torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor zero_row, torch::Tensor out, int64_t n_blocks,
              int64_t m_chunks, int64_t grid);
"""

_MOE_MODULE = []


def moe_cuda_module():
    if not _MOE_MODULE:
        from torch.utils.cpp_extension import load_inline
        _MOE_MODULE.append(load_inline(
            name="okc_moe_gemm", cpp_sources=MOE_CUDA_CPP, cuda_sources=MOE_CUDA_SRC, functions=["moe_gemm"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False))
    return _MOE_MODULE[0]


# ---------------------------------------------------------------------------------------
# Repacking (prepare, once)

# Nibble position in a lane's word for (half = step % 2, b register, bf16x2 element);
# words hold steps (2w, 2w + 1). Matches unpack8's outputs (mma_probe.py).
_NIBBLE_POS = {(0, 0, 0): 12, (0, 0, 1): 28, (0, 1, 0): 8, (0, 1, 1): 24,
               (1, 0, 0): 4, (1, 0, 1): 20, (1, 1, 0): 0, (1, 1, 1): 16}


def _lane_gather_index(device):
    """[32 lanes, 4 words, 8 nibbles] -> (row in group, k in chunk) and bit position."""
    rows, ks, shifts = [], [], []
    for lane in range(32):
        r, c = lane // 4, (lane % 4) * 2
        for w in range(4):
            for (half, bi, e), pos in sorted(_NIBBLE_POS.items(), key=lambda kv: kv[1]):
                step = 2 * w + half
                rows.append(r)
                ks.append(step * 16 + c + 8 * bi + e)
                shifts.append(pos)
    t = lambda v: torch.tensor(v, dtype=torch.int64, device=device).view(32, 4, 8)
    return t(rows), t(ks), t(shifts)


def repack_expert_weights(packed, sf, pair_gate_up):
    """packed uint8 [N, 2048] (two E2M1 codes per byte, low nibble first), sf E4M3 [N, 256]
    for one expert -> (wq int32 [groups, 32 chunks, 32 lanes, 4], ws uint8 [groups, 32, 8, 8]).
    pair_gate_up: groups alternate gate n8 group p, up n8 group p (N = 8192), else plain."""
    n = packed.shape[0]
    device = packed.device
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(n, -1)       # [N, 4096] uint8
    if pair_gate_up:
        half = n // 2
        order = torch.stack((torch.arange(0, half, device=device).view(-1, 8),
                             torch.arange(half, n, device=device).view(-1, 8)), dim=1).reshape(-1)
        codes, sf = codes[order], sf[order]
    groups = n // 8
    rows, ks, shifts = _lane_gather_index(device)
    c = codes.view(groups, 8, 32, 128).permute(0, 2, 1, 3)                         # [g, chunk, row, k]
    picked = c[:, :, rows, ks].to(torch.int64)                                      # [g, chunk, 32, 4, 8]
    words = (picked << shifts).sum(-1)                                              # [g, chunk, 32, 4]
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    ws = sf.view(torch.uint8).view(groups, 8, 32, 8).permute(0, 2, 1, 3).contiguous()  # [g, chunk, row, step]
    return words.contiguous(), ws


def scales_supported(sf):
    """The scale bit trick (exponent add) is exact only for normal E4M3 scales: exponent
    field 1..15. Zero or subnormal scales would decode wrong, so those weights keep Marlin."""
    b = sf.view(torch.uint8) & 0x7F
    return bool(((b >> 3) > 0).all())


# ---------------------------------------------------------------------------------------
# Routing and the post-expert norm (Triton)


@triton.jit
def _route_compact_kernel(ids_ptr, local_start_ptr, row_route_ptr, pos_of_route_ptr, exp_start_ptr,
                          active_ptr, n_active_ptr, P, NE: tl.constexpr, E: tl.constexpr, CHUNK: tl.constexpr):
    """One program per local expert e: places e's routes (in route order) at rows
    exp_start[e] .. exp_start[e+1]; row_route[row] = route p (= token * 8 + slot) and
    pos_of_route[p] = row (-1 for non-local routes, written by program 0). Program 0 also
    writes exp_start [E + 1] and the active-expert list."""
    e = tl.program_id(0)
    experts = tl.arange(0, E)
    offs = tl.arange(0, CHUNK)
    local_start = tl.load(local_start_ptr)
    counts = tl.zeros((E,), tl.int32)
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        loc = (tl.load(ids_ptr + p, mask=p < P, other=0) - local_start + NE) % NE
        loc = tl.where((p < P) & (loc < E), loc, E)
        counts += tl.sum((loc[:, None] == experts[None, :]).to(tl.int32), axis=0)
    ends = tl.cumsum(counts, axis=0)
    starts = ends - counts
    my_start = tl.sum(tl.where(experts == e, starts, 0), axis=0)
    running = my_start
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        in_range = p < P
        loc = (tl.load(ids_ptr + p, mask=in_range, other=0) - local_start + NE) % NE
        mine = (loc == e) & in_range
        rank = tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(row_route_ptr + running + rank, p, mask=mine)
        tl.store(pos_of_route_ptr + p, running + rank, mask=mine)
        running += tl.sum(mine.to(tl.int32), axis=0)
        if e == 0:
            tl.store(pos_of_route_ptr + p, -1, mask=in_range & (loc >= E))
    if e == 0:
        tl.store(exp_start_ptr + experts, starts)
        tl.store(exp_start_ptr + E, tl.sum(counts, axis=0))
        live = (counts > 0).to(tl.int32)
        slot = tl.cumsum(live, axis=0) - 1
        tl.store(active_ptr + slot, experts, mask=live > 0)
        tl.store(n_active_ptr, tl.sum(live, axis=0))


@triton.jit
def _post_norm_rows_kernel(y_ptr, w_ptr, s_ptr, pos_ptr, out_ptr, eps, TOPK: tl.constexpr, H: tl.constexpr):
    """out[t] = sum over slots k with a local row of s[t, k] * RMSNorm(y[row] * w)."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.load(pos_ptr + t * TOPK + k)
        if row >= 0:
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * w
            inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (inv * tl.load(s_ptr + t * TOPK + k))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------------------
# Forward


def build_cuda_bank(weights, start=0, count=32):
    """Repack the local experts for moe_gemm (prepare). Returns None if unsupported."""
    sl = slice(start, start + count)
    if not (scales_supported(weights["w13_sf"][sl]) and scales_supported(weights["w2_sf"][sl])):
        return None
    w13q, w13s, w2q, w2s = [], [], [], []
    for e in range(start, start + count):
        q, s = repack_expert_weights(weights["w13"][e], weights["w13_sf"][e], True)
        w13q.append(q); w13s.append(s)
        q, s = repack_expert_weights(weights["w2"][e], weights["w2_sf"][e], False)
        w2q.append(q); w2s.append(s)
    device = weights["w13"].device
    return {
        "w13q": torch.stack(w13q).contiguous(), "w13s": torch.stack(w13s).contiguous(),
        "w2q": torch.stack(w2q).contiguous(), "w2s": torch.stack(w2s).contiguous(),
        "w13_gs": weights["w13_gs"][sl].float().contiguous(), "w2_gs": weights["w2_gs"][sl].float().contiguous(),
        "zero_row": torch.zeros(4096, dtype=torch.bfloat16, device=device),
        "count": count,
    }


def moe_cuda_forward(x, ids, topk_weights, local_start, bank, norm_w, eps, mt=1, ctas_per_sm=2, topk=8):
    """x [M, 4096] BF16, ids [M, 8] global expert ids, topk_weights [M, 8] FP32, local_start
    int32 [1] (device) -> [M, 4096] BF16: this GPU's experts' weighted, normed sum."""
    mod = moe_cuda_module()
    x = x.contiguous()
    m, hidden = x.shape
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    p = m * topk
    count = bank["count"]
    device = x.device
    row_route = torch.empty((p,), dtype=torch.int32, device=device)
    pos_of_route = torch.empty((p,), dtype=torch.int32, device=device)
    exp_start = torch.empty((count + 1,), dtype=torch.int32, device=device)
    active = torch.empty((count,), dtype=torch.int32, device=device)
    n_active = torch.empty((1,), dtype=torch.int32, device=device)
    _route_compact_kernel[(count,)](ids, local_start, row_route, pos_of_route, exp_start, active, n_active, p,
                                    NE=256, E=count, CHUNK=256, num_warps=4)
    rows = p  # upper bound on local rows
    inter = torch.empty((rows, 4096), dtype=torch.bfloat16, device=device)
    y = torch.empty((rows, 4096), dtype=torch.bfloat16, device=device)
    m_chunks = triton.cdiv(min(m, rows), mt * 16)
    grid = 132 * ctas_per_sm
    mod.moe_gemm(1, mt, x, row_route, bank["w13q"], bank["w13s"], bank["w13_gs"], active, n_active, exp_start,
                 bank["zero_row"], inter, 4096 // 16, m_chunks, grid)
    mod.moe_gemm(2, mt, inter, row_route, bank["w2q"], bank["w2s"], bank["w2_gs"], active, n_active, exp_start,
                 bank["zero_row"], y, 4096 // 32, m_chunks, grid)
    out = torch.empty((m, hidden), dtype=x.dtype, device=device)
    _post_norm_rows_kernel[(m,)](y, norm_w, topk_weights, pos_of_route, out, eps, TOPK=topk, H=hidden, num_warps=8)
    return out
