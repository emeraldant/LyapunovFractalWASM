# Small-batch streaming probe: expert weights loaded by LDG.128 straight into mma.sync A
# fragments (swap-AB: 16 weight rows x 8 tokens per m16n8k16), no shared-memory ring and no
# barriers; a D-deep register ring of k128 blocks per warp keeps loads in flight under the
# decode + MMA. Activations (8 tokens x 4096 bf16) sit in shared memory, read as B fragments.
# Real decode (unpack8 + scale2 + HMUL2) and MMA, so the time is what a kernel built this way
# would see. Compared with pure LDG streaming of the same bytes and with our GEMM stage times
# (bs 1 GEMM1 2 experts 37.7 MB ~17.6-19 us, GEMM2 18.9 MB ~11-12.6 us; bs 8 GEMM1 189 MB ~59 us).
# Timed like the judge (SimBench: graph replay after the 256 MB zero-fill flush).
import torch

_LDG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void unpack8(uint32_t q, uint32_t out[4]) {
  const uint32_t M = 0x81C081C0u;
  out[0] = q & M;
  out[1] = __funnelshift_l(q, q, 10) & M;
  out[2] = __funnelshift_l(q, q, 13) & M;
  out[3] = ((q & 0x001B001Bu) * 0x4108u) & M;
}
template <int I>
__device__ __forceinline__ uint32_t scale2(uint32_t word) {
  uint32_t t = __byte_perm(word, 0u, 0x4040u | I | (I << 8));
  return t * 16u + ((238u << 7) | (238u << 23));
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
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_stream32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

#define XSTRIDE (4096 * 2 + 16)   // bytes per token row in shared memory (padded: conflict-free B loads)

// Layout: weights [tile][k128 block j][half h][lane] uint4 (lane's codes for steps 4h..4h+3),
// scales [tile][j][lane] u32 (rows g, g+8 for steps 2c, 2c+1). A unit = (tile, K slice of KB
// blocks); warps take units grid-stride. Partial sums -> part[(slice * tiles + tile) * 128 + ...].
template <int D, bool IL>
__global__ void __launch_bounds__(512) ldg_mma(const uint4* __restrict__ w, const uint32_t* __restrict__ sc,
                                               const __nv_bfloat16* __restrict__ x, float* __restrict__ part,
                                               int tiles, int kb, int nslices) {
  extern __shared__ __align__(16) uint8_t xs[];
  for (int i = threadIdx.x; i < 8 * 512; i += blockDim.x) {
    const int t = i / 512, o = i % 512;
    *reinterpret_cast<uint4*>(xs + t * XSTRIDE + o * 16) = reinterpret_cast<const uint4*>(x)[i];
  }
  __syncthreads();
  const int lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
  const int nw = blockDim.x / 32;
  const int gw = blockIdx.x * nw + threadIdx.x / 32, tw = gridDim.x * nw;
  const int units = tiles * nslices;
  const uint8_t* xg = xs + g * XSTRIDE + c * 4;
  for (int u = gw; u < units; u += tw) {
    const int tile = u / nslices, sl = u % nslices;
    const int j0 = sl * kb;
    // IL: block j of every tile adjacent ([j][tile]), so warps at the same j read one window.
    const size_t bstride = IL ? (size_t)tiles : 1;
    const size_t b0 = IL ? (size_t)j0 * tiles + tile : (size_t)tile * 32 + j0;
    const uint4* wp = w + b0 * 64 + lane;
    const uint32_t* sp = sc + b0 * 32 + lane;
    uint4 rw[D][2];
    uint32_t rs[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (d < kb) {
        rw[d][0] = ldg_stream(wp + d * bstride * 64);
        rw[d][1] = ldg_stream(wp + d * bstride * 64 + 32);
        rs[d] = ldg_stream32(sp + d * bstride * 32);
      }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < kb; j += D) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
        if (j + d < kb) {
          const uint4 q0 = rw[d][0], q1 = rw[d][1];
          const uint32_t s0 = rs[d];
          if (j + d + D < kb) {
            rw[d][0] = ldg_stream(wp + (j + d + D) * bstride * 64);
            rw[d][1] = ldg_stream(wp + (j + d + D) * bstride * 64 + 32);
            rs[d] = ldg_stream32(sp + (j + d + D) * bstride * 32);
          }
          uint32_t sw[4];
#pragma unroll
          for (int cc = 0; cc < 4; ++cc) sw[cc] = __shfl_sync(0xffffffffu, s0, (lane & ~3) | cc);
          const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
          const uint8_t* xk = xg + (size_t)(j0 + j + d) * 256;
#pragma unroll
          for (int st = 0; st < 8; ++st) {
            uint32_t a[4];
            unpack8(words[st], a);
            const uint32_t sg = (st & 1) ? scale2<2>(sw[st / 2]) : scale2<0>(sw[st / 2]);
            const uint32_t sg8 = (st & 1) ? scale2<3>(sw[st / 2]) : scale2<1>(sw[st / 2]);
            a[0] = mul2(a[0], sg);
            a[2] = mul2(a[2], sg);
            a[1] = mul2(a[1], sg8);
            a[3] = mul2(a[3], sg8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xk + st * 32);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xk + st * 32 + 16);
            mma16816(acc, a, b0, b1);
          }
        }
      }
    }
    float* o = part + ((size_t)sl * tiles + tile) * 128;
    *reinterpret_cast<float2*>(o + g * 8 + 2 * c) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2*>(o + (g + 8) * 8 + 2 * c) = make_float2(acc[2], acc[3]);
  }
}

__global__ void ldg_plain(const uint4* __restrict__ src, size_t n, uint32_t* sink) {
  uint32_t x = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    uint4 a = __ldcs(src + i), b = __ldcs(src + i + stride), c = __ldcs(src + i + 2 * stride), d = __ldcs(src + i + 3 * stride);
    x ^= a.x ^ a.w ^ b.y ^ c.z ^ d.x ^ d.w;
  }
  for (; i < n; i += stride) x ^= __ldcs(src + i).y;
  if (x == 0x12345678u) sink[0] = x;
}

// Control: each warp streams its own contiguous region (as a unit does), D uint4 per lane in flight.
template <int D>
__global__ void ldg_region(const uint4* __restrict__ src, size_t per_warp16, int nregions, uint32_t* sink) {
  const int lane = threadIdx.x % 32;
  const int gw = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, tw = gridDim.x * (blockDim.x / 32);
  uint32_t x = 0;
  for (int r = gw; r < nregions; r += tw) {
    const uint4* p = src + (size_t)r * per_warp16 + lane;
    for (size_t i = 0; i < per_warp16; i += 32 * D) {
      uint4 v[D];
#pragma unroll
      for (int d = 0; d < D; ++d) v[d] = __ldcs(p + i + d * 32);
#pragma unroll
      for (int d = 0; d < D; ++d) x ^= v[d].x ^ v[d].w;
    }
  }
  if (x == 0x12345678u) sink[0] = x;
}


// bs 1 GEMV on CUDA cores (no MMA): lane = one weight row of a 32-row group; per k128 block a lane
// loads its 64 B of codes (4 LDG.128, layout [group][block][q][lane]) + 8 scale bytes, unpacks with
// unpack8, rescales by a constant (HMUL2: undo the 2^-126 placement), multiplies by the activation
// pairs (one broadcast LDS.128 per word) with HFMA2 into a bf16x2 sum per 16-value scale block, then
// one fp32 FMA by the block's scale. (Timing probe: bit order / denormals are not checked here.)
template <int D>
__global__ void __launch_bounds__(512) ldg_gemv(const uint4* __restrict__ w, const uint2* __restrict__ sc,
                                                const __nv_bfloat16* __restrict__ x, float* __restrict__ part,
                                                int groups, int kb, int nslices) {
  __shared__ __align__(16) uint8_t xs[8192];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int nw = blockDim.x / 32;
  const int gw = blockIdx.x * nw + threadIdx.x / 32, tw = gridDim.x * nw;
  const int units = groups * nslices;
  const uint32_t K2 = 0x7E807E80u;   // bf16x2 (2^126, 2^126)
  for (int u = gw; u < units; u += tw) {
    const int g = u / nslices, sl = u % nslices;
    const int j0 = sl * kb;
    const uint4* wp = w + ((size_t)g * 32 + j0) * 128 + lane;
    const uint2* sp = sc + ((size_t)g * 32 + j0) * 32 + lane;
    uint4 rw[D][4];
    uint2 rs[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
      if (d < kb) {
#pragma unroll
        for (int q = 0; q < 4; ++q) rw[d][q] = ldg_stream(wp + d * 128 + q * 32);
        rs[d] = *reinterpret_cast<const uint2*>(sp + d * 32);
      }
    float acc = 0.f;
    for (int j = 0; j < kb; j += D) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
        if (j + d < kb) {
          uint4 q4[4] = {rw[d][0], rw[d][1], rw[d][2], rw[d][3]};
          const uint2 s2 = rs[d];
          if (j + d + D < kb) {
#pragma unroll
            for (int q = 0; q < 4; ++q) rw[d][q] = ldg_stream(wp + (j + d + D) * 128 + q * 32);
            rs[d] = *reinterpret_cast<const uint2*>(sp + (j + d + D) * 32);
          }
          const uint8_t* xk = xs + (size_t)(j0 + j + d) * 256;
#pragma unroll
          for (int b = 0; b < 8; ++b) {   // scale block b: words 2b, 2b + 1
            __nv_bfloat162 s2acc = __floats2bfloat162_rn(0.f, 0.f);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int wi = 2 * b + h;
              const uint4 qq = q4[wi / 4];
              const uint32_t word = (wi % 4) == 0 ? qq.x : (wi % 4) == 1 ? qq.y : (wi % 4) == 2 ? qq.z : qq.w;
              uint32_t v[4];
              unpack8(word, v);
              const uint4 xv = *reinterpret_cast<const uint4*>(xk + wi * 16);
              const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const uint32_t a = mul2(v[i], K2);
                s2acc = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&xw[i]), s2acc);
              }
            }
            const uint32_t sb = ((b < 4 ? s2.x : s2.y) >> (8 * (b % 4))) & 0x7Fu;
            const float scf = __uint_as_float((sb << 20) + (120u << 23));
            acc = fmaf(scf, __low2float(s2acc) + __high2float(s2acc), acc);
          }
        }
      }
    }
    part[(size_t)sl * groups * 32 + g * 32 + lane] = acc;
  }
}

static torch::Tensor g_w, g_s, g_x, g_part, g_sink;
void ldg_setup(int64_t max_tiles) {
  g_w = torch::randint(0, 256, {max_tiles * 32 * 64 * 16}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_s = torch::randint(40, 80, {max_tiles * 32 * 32 * 4}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_x = torch::randn({8, 4096}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  g_part = torch::zeros({16 * max_tiles * 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  g_sink = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  const int smem = 8 * XSTRIDE;
  cudaFuncSetAttribute(ldg_mma<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(ldg_mma<4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(ldg_mma<6, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(ldg_mma<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(ldg_mma<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
// mode 0: ldg_mma; mode 1: plain LDG streaming of the same weight + scale bytes; mode 2: ldg_mma
// interleaved layout; mode 3: per-warp regions of tiles * 32 / nslices blocks (d uint4 in flight).
void ldg_launch(int64_t mode, int64_t d, int64_t tiles, int64_t nslices, int64_t cps, int64_t threads) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int grid = 132 * (int)cps;
  if (mode == 1) {
    const size_t n = (size_t)tiles * 32 * 64 * 9 / 8;
    ldg_plain<<<grid, (int)threads, 0, st>>>(reinterpret_cast<const uint4*>(g_w.data_ptr()), n,
                                             reinterpret_cast<uint32_t*>(g_sink.data_ptr()));
    return;
  }
  if (mode == 4) {   // GEMV: groups of 32 rows = tiles / 2
    const int kb = 32 / (int)nslices;
    auto* w4 = reinterpret_cast<const uint4*>(g_w.data_ptr());
    auto* s2 = reinterpret_cast<const uint2*>(g_s.data_ptr());
    auto* xx = reinterpret_cast<const __nv_bfloat16*>(g_x.data_ptr());
    float* pp = g_part.data_ptr<float>();
    if (d == 1) ldg_gemv<1><<<grid, (int)threads, 0, st>>>(w4, s2, xx, pp, (int)(tiles / 2), kb, (int)nslices);
    if (d == 2) ldg_gemv<2><<<grid, (int)threads, 0, st>>>(w4, s2, xx, pp, (int)(tiles / 2), kb, (int)nslices);
    return;
  }
  if (mode == 3) {
    const size_t per = (size_t)(32 / nslices) * 72;   // uint4 per region: kb blocks x 1152 B
    const int nreg = (int)(tiles * nslices);
    auto* src = reinterpret_cast<const uint4*>(g_w.data_ptr());
    auto* sk = reinterpret_cast<uint32_t*>(g_sink.data_ptr());
    if (d == 2) ldg_region<2><<<grid, (int)threads, 0, st>>>(src, per, nreg, sk);
    if (d == 4) ldg_region<4><<<grid, (int)threads, 0, st>>>(src, per, nreg, sk);
    return;
  }
  const int smem = 8 * XSTRIDE;
  const uint4* w = reinterpret_cast<const uint4*>(g_w.data_ptr());
  const uint32_t* s = reinterpret_cast<const uint32_t*>(g_s.data_ptr());
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(g_x.data_ptr());
  float* p = g_part.data_ptr<float>();
  const int kb = 32 / (int)nslices;
  if (mode == 2) {
    if (d == 2) ldg_mma<2, true><<<grid, (int)threads, smem, st>>>(w, s, x, p, (int)tiles, kb, (int)nslices);
    if (d == 4) ldg_mma<4, true><<<grid, (int)threads, smem, st>>>(w, s, x, p, (int)tiles, kb, (int)nslices);
    return;
  }
  if (d == 2) ldg_mma<2, false><<<grid, (int)threads, smem, st>>>(w, s, x, p, (int)tiles, kb, (int)nslices);
  if (d == 4) ldg_mma<4, false><<<grid, (int)threads, smem, st>>>(w, s, x, p, (int)tiles, kb, (int)nslices);
  if (d == 6) ldg_mma<6, false><<<grid, (int)threads, smem, st>>>(w, s, x, p, (int)tiles, kb, (int)nslices);
}
"""
_LDG_CPP = ("void ldg_setup(int64_t max_tiles);\n"
            "void ldg_launch(int64_t mode, int64_t d, int64_t tiles, int64_t nslices, int64_t cps, int64_t threads);\n")


def run_ldg_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="okc_ldg_probe", cpp_sources=_LDG_CPP, cuda_sources=_LDG_SRC,
                      functions=["ldg_setup", "ldg_launch"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"ldg probe: compiled in {time.time() - t0:.0f} s", flush=True)
    bench = SimBench()
    tiny = torch.zeros(1, device="cuda")
    # (label, 16-row tiles): bs 1 GEMM1 = 2 experts x 8192 rows; bs 1 GEMM2 = 2 x 4096; bs 8 GEMM1 = 10 x 8192.
    cases = (("bs1 G1 2x8192", 1024), ("bs1 G2 2x4096", 512), ("bs8 G1 10x8192", 5120), ("bs8 G2 10x4096", 2560))
    mod.ldg_setup(5120)
    def t(fn, pairs=6):
        return bench.compare(fn, lambda: tiny.add_(1), pairs=pairs)[0]
    cases = cases[:2]
    for label, tiles in cases:
        nbytes = tiles * 32 * 64 * 16 * 9 // 8
        line = [f"plain {t(lambda: mod.ldg_launch(1, 0, tiles, 1, 4, 512), 8):.1f}",
                f"mma D2 s4 2x512 {t(lambda: mod.ldg_launch(0, 2, tiles, 4, 2, 512)):.1f}",
                f"mma D2 s8 2x512 {t(lambda: mod.ldg_launch(0, 2, tiles, 8, 2, 512)):.1f}"]
        res = []
        for d in (1, 2):
            for ns in (2, 4, 8, 16):
                for cps, th in ((1, 512), (2, 512), (4, 256)):
                    res.append((t(lambda d=d, ns=ns, cps=cps, th=th: mod.ldg_launch(4, d, tiles, ns, cps, th)), f"gemv D{d} s{ns} {cps}x{th}"))
        res.sort()
        line += [f"{n} {u:.1f}" for u, n in res[:5]]
        print(f"ldg3 {label} {nbytes / 1e6:.1f} MB (us): " + "; ".join(line), flush=True)
