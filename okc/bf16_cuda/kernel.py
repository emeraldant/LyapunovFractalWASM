"""Skinny BF16 GEMM (m <= 32 tokens) for the dense projections: out[m, n] = x[m, k] @ W[n, k].T.

Memory bound (the weights are read once), so it is a weight stream: W is prepacked (once, lossless:
same BF16 values) into mma.sync m16n8k16 A-fragment order, [16-row tile][k16 step][lane] x 16 B, and
each warp streams its fragments through a shared-memory ring filled by TMA bulk copies (deep enough
to cover DRAM latency at full bandwidth). The tokens are the B operand (n8 tiles, up to 4), staged
once per CTA (its K slice) in shared memory. K is split S ways so there are enough (tile, slice) units for every
warp. Slices meet as fp32 parts; each tile's last-arriving slice sums them in slice order
(deterministic: per-lane release fences, as the MoE GEMV) and writes BF16. One launch, no memset.
"""
import torch

BF16_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 bs_ldg(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void bs_mma(float c[4], uint4 a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bs_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bs_bar_init(uint32_t bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar) : "memory");
}
// weights are read once: evict_first, so the stream recycles its own L2 lines instead of evicting
// (and writing back) whatever else is resident
__device__ __forceinline__ void bs_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
  asm volatile("{\n .reg .b64 pol;\n createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
               " cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], pol;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bs_bulk_only(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bs_bar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n BS_WAIT%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra BS_WAIT%=;\n}\n"
               :: "r"(bar), "r"(parity) : "memory");
}

// NT: n8 token tiles (1-4). C: k16 steps per chunk (one bulk copy of C x 512 B); R: chunks in flight
// per warp. A CTA owns one K slice (s = blockIdx % S): it stages x[:, slice] in shared memory once
// (B fragments are then conflict-free LDS.32, no L2 latency in the loop), and its warps take 16-row
// tiles grid-stride over the slice's CTAs. Each warp streams its tiles' weights through its own
// shared-memory ring with TMA bulk copies (lane 0 issues, one mbarrier per stage): R*C*512 B per warp
// in flight, enough to cover DRAM latency at full bandwidth.
template <int NT, int C, int R>
__global__ void __launch_bounds__(384, 1) bf16_stream_kernel(const uint4* __restrict__ wq, const __nv_bfloat16* __restrict__ x,
                                                             __nv_bfloat16* __restrict__ out, float* __restrict__ part,
                                                             int* __restrict__ cnt, int m, int n, int k, int S) {
  extern __shared__ __align__(128) unsigned char bs_sm[];
  const int ncb = gridDim.x / S;
  if (blockIdx.x >= ncb * S) return;
  const int s = blockIdx.x % S, cb = blockIdx.x / S;
  const int kS = k / S, steps = kS / 16, tiles = n / 16, ktot_steps = k / 16, XS = kS + 8;
  const int warp = threadIdx.x / 32, nw = blockDim.x / 32;
  const int lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
  const int t0 = cb * nw + warp, TS = ncb * nw;
  const int my_tiles = t0 < tiles ? (tiles - 1 - t0) / TS + 1 : 0;
  const int chunks = my_tiles * steps / C;
  const uint4* ring = reinterpret_cast<const uint4*>(bs_sm + (size_t)warp * R * C * 512);   // [R][C][32 lanes]
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(bs_sm + (size_t)nw * R * C * 512);  // [m][XS]
  const uint32_t ring_s = bs_smem(ring);
  const uint32_t bar_s = bs_smem(bs_sm + (size_t)nw * R * C * 512 + (size_t)m * XS * 2) + warp * R * 8;
  // weights first: start the stream before staging x
  int pt = t0, pst = 0;
  const uint4* pw = wq + ((size_t)t0 * ktot_steps + s * steps) * 32;
  auto issue = [&](int r) {
    if (lane == 0) bs_bulk(ring_s + r * C * 512, pw, C * 512, bar_s + 8 * r);
    pw += C * 32;
    pst += C;
    if (pst == steps) {
      pst = 0;
      pt += TS;
      pw = wq + ((size_t)pt * ktot_steps + s * steps) * 32;
    }
  };
  if (chunks > 0) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) bs_bar_init(bar_s + 8 * r);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncwarp();
#pragma unroll 1
    for (int r = 0; r < R && r < chunks; ++r) issue(r);
  }
  // stage x[:, slice] by bulk copies, one per token row, on one CTA barrier (rows padded by 8
  // elements: the 8 g-rows x 4 c-words of a B fragment then hit 32 distinct banks)
  const uint32_t xbar = bs_smem(bs_sm + (size_t)nw * R * C * 512 + (size_t)m * XS * 2) + nw * R * 8;
  if (threadIdx.x == 0) {
    bs_bar_init(xbar);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(xbar), "r"(m * kS * 2) : "memory");
    __syncwarp();
    for (int row = lane; row < m; row += 32)
      bs_bulk_only(bs_smem(xs + (size_t)row * XS), x + (size_t)row * k + (size_t)s * kS, kS * 2, xbar);
  }
  bs_bar_wait(xbar, 0);   // every warp waits, so no CTA exits with a copy into its shared memory in flight
  if (chunks == 0) return;
  const __nv_bfloat16* xrow[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) xrow[j] = xs + (size_t)min(8 * j + g, m - 1) * XS + 2 * c;
  float acc[NT][4];
  int st = 0, t = t0;
  int r = 0;
  uint32_t parity = 0;
#pragma unroll 1
  for (int ci = 0; ci < chunks; ++ci) {
    bs_bar_wait(bar_s + 8 * r, parity);
    const uint4* chunk = ring + r * C * 32 + lane;
#pragma unroll
    for (int p = 0; p < C; ++p) {
      if (st == 0) {
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      }
      const uint4 a = chunk[p * 32];
      const int kk = st * 16;
#pragma unroll
      for (int j = 0; j < NT; ++j)
        bs_mma(acc[j], a, *reinterpret_cast<const uint32_t*>(xrow[j] + kk), *reinterpret_cast<const uint32_t*>(xrow[j] + kk + 8));
      if (st == steps - 1) {   // tile done: write (S == 1) or meet the other slices
        if (S == 1) {
#pragma unroll
          for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int row = 16 * t + g + (i >= 2 ? 8 : 0), tok = 8 * j + 2 * c + (i & 1);
              if (tok < m) out[(size_t)tok * n + row] = __float2bfloat16_rn(acc[j][i]);
            }
        } else {
          float* pp = part + ((size_t)s * tiles + t) * (16 * 32);   // [slice][tile][row 16][token 32]
#pragma unroll
          for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) pp[(g + (i >= 2 ? 8 : 0)) * 32 + 8 * j + 2 * c + (i & 1)] = acc[j][i];
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          __syncwarp();
          int last = 0;
          if (lane == 0) {
            const int old = atomicAdd(cnt + t, 1);
            last = old == S - 1;
            if (last) cnt[t] = 0;
          }
          last = __shfl_sync(0xffffffffu, last, 0);
          if (last) {
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
            const int row = lane % 16;
            for (int tok = lane / 16; tok < min(m, NT * 8); tok += 2) {
              float v = 0.f;
              for (int qq = 0; qq < S; ++qq) v += __ldcg(part + ((size_t)qq * tiles + t) * (16 * 32) + row * 32 + tok);
              out[(size_t)tok * n + 16 * t + row] = __float2bfloat16_rn(v);
            }
          }
        }
      }
      if (++st == steps) { st = 0; t += TS; }
    }
    // stage r is free once every lane has read it: order those generic reads before the async refill
    __syncwarp();
    if (ci + R < chunks) {
      if (lane == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      issue(r);
    }
    if (++r == R) { r = 0; parity ^= 1u; }
  }
}

void bf16_stream(torch::Tensor wq, torch::Tensor x, torch::Tensor out, torch::Tensor part, torch::Tensor cnt,
                 int64_t n, int64_t S, int64_t grid) {
  constexpr int C = 4, R = 4, NW = 12;
  const int m = (int)x.size(0), k = (int)x.size(1);
  const int smem = NW * R * C * 512 + m * (int)(k / S + 8) * 2 + NW * R * 8 + 8;
  TORCH_CHECK(m >= 1 && m <= 32 && k % (S * 16 * C) == 0 && n % 16 == 0 && grid >= S && smem <= 227 * 1024,
              "bf16_stream: shape");
  const int nt = (m + 7) / 8;
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* w = reinterpret_cast<const uint4*>(wq.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* pp = part.data_ptr<float>();
  int* c = cnt.data_ptr<int32_t>();
#define BS_L(NT_) { \
    static bool attr = false; \
    if (!attr) { cudaFuncSetAttribute(bf16_stream_kernel<NT_, C, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024); attr = true; } \
    bf16_stream_kernel<NT_, C, R><<<(int)grid, NW * 32, smem, st>>>(w, xp, o, pp, c, m, (int)n, k, (int)S); return; }
  if (nt == 1) BS_L(1)
  if (nt == 2) BS_L(2)
  if (nt == 3) BS_L(3)
  BS_L(4)
}
"""
BF16_CPP = "void bf16_stream(torch::Tensor wq, torch::Tensor x, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t n, int64_t S, int64_t grid);\n"

_BF16_MOD = []


def bf16_module():
    if not _BF16_MOD:
        from torch.utils.cpp_extension import load_inline
        _BF16_MOD.append(load_inline(name="okc_bf16_stream", cpp_sources=BF16_CPP, cuda_sources=BF16_SRC,
                                     functions=["bf16_stream"],
                                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False))
    return _BF16_MOD[0]


def pack_bf16(w):
    """W [n, k] BF16 -> [n / 16 tiles, k / 16 steps, 32 lanes, 8] BF16 in m16n8k16 A-fragment order:
    lane (g, c) holds (row g, k 2c..2c+1), (row g + 8, same k), (row g, k 2c+8..+9), (row g + 8, ...)."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)   # [tile, row half h, g, step, k half kh, c, pair]
    # fragment register order: a0 (h0, kh0), a1 (h1, kh0), a2 (h0, kh1), a3 (h1, kh1)
    t = t.permute(0, 3, 2, 5, 4, 1, 6)            # [tile, step, g, c, kh, h, pair]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def bf16_stream_mm(mod, wq, x, n, S, grid, part, cnt):
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    mod.bf16_stream(wq, x, out, part, cnt, n, S, grid)
    return out
