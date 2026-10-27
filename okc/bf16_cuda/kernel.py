"""Skinny BF16 GEMM (m <= 32 tokens) for the dense projections: out[m, n] = x[m, k] @ W[n, k].T.

Memory bound (the weights are read once), so it is a weight stream: W is prepacked (once, lossless:
same BF16 values) into mma.sync m16n8k16 A-fragment order, [16-row tile][k16 step][lane] x 16 B, and
each lane loads its fragment with one LDG.128 straight into registers, P steps ahead (no shared
memory ring, no barriers). The tokens are the B operand (n8 tiles, up to 4), staged per K slice in
shared memory. K is split S ways so every SM streams; a CTA owns one slice, its warps take 16-row
tiles. Slices meet as fp32 parts; each tile's last-arriving slice sums them in slice order
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
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// NT: n8 token tiles (1-4); P: k16 steps of weights in flight per lane. Persistent: units (16-row
// tile, K slice) go to warps grid-stride, and each warp's ring of P loads runs on across its units,
// so the next unit's first steps are in flight under the current one's last. The tokens (B
// fragments) are read from global memory through L1 (every warp of the SM reads the same few rows).
template <int NT, int P>
__global__ void __launch_bounds__(512) bf16_stream_kernel(const uint4* __restrict__ wq, const __nv_bfloat16* __restrict__ x,
                                                          __nv_bfloat16* __restrict__ out, float* __restrict__ part,
                                                          int* __restrict__ cnt, int m, int n, int k, int S) {
  const int kS = k / S, steps = kS / 16, tiles = n / 16, ktot_steps = k / 16;
  const int lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
  const int gw = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, tw = gridDim.x * (blockDim.x / 32);
  const int units = tiles * S;
  const int my_units = gw < units ? (units - 1 - gw) / tw + 1 : 0;
  const int total = my_units * steps;   // this warp's k16 steps over all its units
  auto addr = [&](int q) {   // weight fragment of the warp's q-th step
    const int u = gw + (q / steps) * tw, st = q % steps;
    const int t = u / S, s = u % S;
    return wq + ((size_t)t * ktot_steps + s * steps + st) * 32 + lane;
  };
  uint4 ring[P];
#pragma unroll
  for (int p = 0; p < P; ++p) ring[p] = p < total ? bs_ldg(addr(p)) : make_uint4(0u, 0u, 0u, 0u);
  // token rows: B fragment pointers (rows >= m read row 0 and are never stored)
  const __nv_bfloat16* xrow[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) xrow[j] = x + (size_t)min(8 * j + g, m - 1) * k + 2 * c;
  float acc[NT][4];
  for (int q0 = 0; q0 < total; q0 += P) {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const int q = q0 + p;
      if (q < total) {
        const int st = q % steps, u = gw + (q / steps) * tw;
        const int t = u / S, s = u % S;
        if (st == 0) {
#pragma unroll
          for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        }
        const uint4 a = ring[p];
        if (q + P < total) ring[p] = bs_ldg(addr(q + P));
        const int kk = (s * steps + st) * 16;
#pragma unroll
        for (int j = 0; j < NT; ++j)
          bs_mma(acc[j], a, __ldg(reinterpret_cast<const uint32_t*>(xrow[j] + kk)),
                 __ldg(reinterpret_cast<const uint32_t*>(xrow[j] + kk + 8)));
        if (st == steps - 1) {   // unit done: write (S == 1) or meet the other slices
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
      }
    }
  }
}

void bf16_stream(torch::Tensor wq, torch::Tensor x, torch::Tensor out, torch::Tensor part, torch::Tensor cnt,
                 int64_t n, int64_t S, int64_t grid) {
  const int m = (int)x.size(0), k = (int)x.size(1);
  TORCH_CHECK(m >= 1 && m <= 32 && k % (S * 16) == 0 && n % 16 == 0, "bf16_stream: shape");
  const int nt = (m + 7) / 8;
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* w = reinterpret_cast<const uint4*>(wq.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* pp = part.data_ptr<float>();
  int* c = cnt.data_ptr<int32_t>();
#define BS_L(NT_) { bf16_stream_kernel<NT_, 8><<<(int)grid, 512, 0, st>>>(w, xp, o, pp, c, m, (int)n, k, (int)S); return; }
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
