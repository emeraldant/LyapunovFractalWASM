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

#define KTOT 4096

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

// v2 (v1 reached 1.7 TB/s at bs 1 and 0.35 TB/s at bs 256: one chunk in flight per warp,
// per-step global gathers of the activations, and a K-split reduction per item).
//
// CTA = 8 warps; a work item is (active expert, block of 8 warps' groups, token chunk). Each
// warp owns NGW n8 groups over the whole K (MODE 1: a gate group and its up group, so
// silu(gate) * up stays in registers; MODE 2: one group). A STAGES-deep cp.async pipeline
// moves, per 128-k chunk, the item's activation tile (MT x 16 rows, XOR-swizzled 16-byte
// segments, read with ldmatrix) and each warp's weight and scale bytes into shared memory.
template <int MODE, int MT, int NGW, int STAGES>
__global__ void __launch_bounds__(256) moe_gemm_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint32_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks, int m_chunks) {
  constexpr int A_STAGE = MT * 16 * 128 * 2;          // bytes
  constexpr int W_STAGE = 8 * NGW * 512;
  constexpr int S_STAGE = 8 * NGW * 64;
  constexpr int NCHUNK = KTOT / 128;
  constexpr int GROUPS = MODE == 1 ? 1024 : 512;      // n8 groups per expert
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sW = smem + STAGES * A_STAGE;
  uint8_t* sS = sW + STAGES * W_STAGE;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
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
    const float gsc = gs[e] * 256.0f;

    // This thread's activation segments: MT of them per stage (row = seg / 16, 16 B each).
    const uint8_t* a_row[MT];
    int a_ok[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      int seg = tid + i * 256;
      int lr = seg / 16;
      a_ok[i] = lr < rows_e && lr < tiles * 16;
      int src = 0;
      if (a_ok[i]) src = MODE == 1 ? row_route[row0 + lr] / 8 : row0 + lr;
      a_row[i] = reinterpret_cast<const uint8_t*>(a_src + (size_t)src * KTOT);
    }
    // This warp's weight groups.
    size_t gch[NGW];
#pragma unroll
    for (int j = 0; j < NGW; ++j) {
      int grp = MODE == 1 ? 2 * (nb * 8 + warp) + j : nb * 8 + warp;
      gch[j] = ((size_t)e * GROUPS + grp) * NCHUNK;
    }
    auto load_stage = [&](int kc, int st) {
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        int seg = tid + i * 256;
        int lr = seg / 16, cs = seg % 16;
        if (lr < tiles * 16) {
          uint32_t dst = (uint32_t)__cvta_generic_to_shared(sA + st * A_STAGE + lr * 256 + ((cs ^ (lr & 7)) * 16));
          const uint8_t* src = a_row[i] + kc * 256 + cs * 16;
          int bytes = a_ok[i] ? 16 : 0;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes));
        }
      }
#pragma unroll
      for (int j = 0; j < NGW; ++j) {
        size_t ch = gch[j] + kc;
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(sW + st * W_STAGE + ((warp * NGW + j) * 32 + lane) * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(wq + (ch * 32 + lane) * 4));
        if (lane < 4) {
          uint32_t sdst = (uint32_t)__cvta_generic_to_shared(sS + st * S_STAGE + (warp * NGW + j) * 64 + lane * 16);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sdst), "l"(ws + ch * 64 + lane * 16));
        }
      }
    };
    float acc[MT][NGW][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < NGW; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][j][i] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      load_stage(s, s);
      asm volatile("cp.async.commit_group;\n" ::);
    }
#pragma unroll 1
    for (int kc = 0; kc < NCHUNK; ++kc) {
      asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
      __syncthreads();
      if (kc + STAGES - 1 < NCHUNK) load_stage(kc + STAGES - 1, (kc + STAGES - 1) % STAGES);
      asm volatile("cp.async.commit_group;\n" ::);
      const int st = kc % STAGES;
      uint4 wv[NGW];
      uint2 sv[NGW];
#pragma unroll
      for (int j = 0; j < NGW; ++j) {
        wv[j] = *reinterpret_cast<const uint4*>(sW + st * W_STAGE + ((warp * NGW + j) * 32 + lane) * 16);
        sv[j] = *reinterpret_cast<const uint2*>(sS + st * S_STAGE + (warp * NGW + j) * 64 + r * 8);
      }
      const uint8_t* a_stage = sA + st * A_STAGE;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        uint32_t v[NGW][4];
#pragma unroll
        for (int j = 0; j < NGW; ++j) {
          uint32_t q = w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w;
          unpack8(q, v[j]);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int step = 2 * w + half;
          uint32_t b0[NGW], b1[NGW];
#pragma unroll
          for (int j = 0; j < NGW; ++j) {
            uint32_t sb = (step < 4 ? (sv[j].x >> (8 * step)) : (sv[j].y >> (8 * (step - 4)))) & 0xFFu;
            uint32_t s2 = scale2(sb);
            b0[j] = mul2(v[j][2 * half], s2);
            b1[j] = mul2(v[j][2 * half + 1], s2);
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            if (mt < tiles) {
              int row = mt * 16 + (lane % 16);
              int cs = step * 2 + lane / 16;
              uint32_t addr = (uint32_t)__cvta_generic_to_shared(a_stage + row * 256 + ((cs ^ (row & 7)) * 16));
              uint32_t a[4];
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                           : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
#pragma unroll
              for (int j = 0; j < NGW; ++j) mma16816(acc[mt][j], a, b0[j], b1[j]);
            }
          }
        }
      }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // Epilogue straight from the accumulators: c0, c1 row r cols c, c+1; c2, c3 row r + 8.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= tiles) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int orow = mt * 16 + r + 8 * h;
        if (orow >= rows_e) continue;
        float v0, v1;
        int n;
        if (MODE == 1) {
          float g0 = acc[mt][0][2 * h] * gsc, g1 = acc[mt][0][2 * h + 1] * gsc;
          float u0 = acc[mt][NGW - 1][2 * h] * gsc, u1 = acc[mt][NGW - 1][2 * h + 1] * gsc;
          v0 = g0 / (1.f + __expf(-g0)) * u0;
          v1 = g1 / (1.f + __expf(-g1)) * u1;
          n = 8 * (nb * 8 + warp) + c;
        } else {
          v0 = acc[mt][0][2 * h] * gsc;
          v1 = acc[mt][0][2 * h + 1] * gsc;
          n = 8 * (nb * 8 + warp) + c;
        }
        __nv_bfloat162 pr = __floats2bfloat162_rn(v0, v1);
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(row0 + orow) * KTOT + n) = pr;
      }
    }
  }
}

template <int MODE, int MT, int STAGES>
void launch(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
            torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
            int n_blocks, int m_chunks, int grid) {
  constexpr int NGW = MODE == 1 ? 2 : 1;
  size_t smem = (size_t)STAGES * (MT * 16 * 128 * 2 + 8 * NGW * 512 + 8 * NGW * 64);
  auto k = moe_gemm_kernel<MODE, MT, NGW, STAGES>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), row_route.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), ws.data_ptr<uint8_t>(), gs.data_ptr<float>(),
      active.data_ptr<int32_t>(), n_active.data_ptr<int32_t>(), exp_start.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, m_chunks);
}

void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t m_chunks, int64_t grid) {
#define CASE(M_, T_, S_) if (mode == M_ && mt == T_ && stages == S_) { launch<M_, T_, S_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, m_chunks, grid); return; }
  CASE(1, 1, 4) CASE(1, 2, 4) CASE(1, 4, 4) CASE(2, 1, 4) CASE(2, 2, 4) CASE(2, 4, 4)
  CASE(1, 1, 6) CASE(2, 1, 6)
  TORCH_CHECK(false, "moe_gemm: unsupported mode / m tiles / stages");
}
"""

MOE_CUDA_CPP = """
void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t m_chunks, int64_t grid);
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
        "count": count,
    }


def moe_cuda_forward(x, ids, topk_weights, local_start, bank, norm_w, eps, mt=1, ctas_per_sm=2, topk=8,
                     stages=4):
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
    # 64 work items per active expert per GEMM: 8 warps x (MODE 1: a gate/up pair, MODE 2: an n8 group).
    mod.moe_gemm(1, mt, stages, x, row_route, bank["w13q"], bank["w13s"], bank["w13_gs"], active, n_active,
                 exp_start, inter, 512 // 8, m_chunks, grid)
    mod.moe_gemm(2, mt, stages, inter, row_route, bank["w2q"], bank["w2s"], bank["w2_gs"], active, n_active,
                 exp_start, y, 512 // 8, m_chunks, grid)
    out = torch.empty((m, hidden), dtype=x.dtype, device=device)
    _post_norm_rows_kernel[(m,)](y, norm_w, topk_weights, pos_of_route, out, eps, TOPK=topk, H=hidden, num_warps=8)
    return out
