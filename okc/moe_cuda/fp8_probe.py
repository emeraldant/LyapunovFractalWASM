# FP8 inner-loop probe: is decoding NVFP4 to e4m3 and multiplying with mma.sync m16n8k32 e4m3
# (activations as an exact hi + lo FP8 pair in the two k halves, the block's weights duplicated in
# both: one k32 MMA = one 16-k scale block, the scale applied to the fp32 accumulator) cheaper per
# weight than our bf16 loop (unpack8 + scale + HMUL2 + m16n8k16)? Registers only (no memory), many
# warps, clocks per 8 weights per warp; plus the bare MMA rates. Timed in a plain launch (no flush).
import torch

_FP8_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void unpack8(uint32_t q, uint32_t out[4]) {
  const uint32_t M = 0x81C081C0u;
  out[0] = q & M;
  out[1] = __funnelshift_l(q, q, 10) & M;
  out[2] = __funnelshift_l(q, q, 13) & M;
  out[3] = ((q & 0x001B001Bu) * 0x4108u) & M;
}
__device__ __forceinline__ uint32_t scale_even(uint32_t w) {
  uint32_t r;
  asm("mad.lo.u32 %0, %1, 16, %2;" : "=r"(r) : "r"(w & 0x00FF00FFu), "r"((238u << 7) | (238u << 23)));
  return r;
}
__device__ __forceinline__ uint32_t scale_odd(uint32_t w) {
  uint32_t r;
  asm("mad.hi.u32 %0, %1, 268435456, %2;" : "=r"(r) : "r"(w & 0xFF00FF00u), "r"((238u << 7) | (238u << 23)));
  return r;
}
template <int H>
__device__ __forceinline__ uint32_t mul2h(uint32_t a, uint32_t s) {
  __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), H == 0 ? __low2bfloat162(sv) : __high2bfloat162(sv));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D = A * B (C = 0): A = (a0, a1, a0, a1) (the block's weights in both k halves), B = (hi, lo).
__device__ __forceinline__ void mma_e4m3(float d[4], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%4,%5}, {%6,%7}, {%8,%8,%8,%8};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(0.f));
}
__device__ __forceinline__ void mma_e4m3_acc(float c[4], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// MODE 0: bf16 loop (per 8 weights: unpack8, scales, 4 HMUL2, one m16n8k16). Per iteration a warp
//   handles 16 words = 16 octets per lane (two "chunk words" x 8), like v6's inner loop.
// MODE 1: e4m3 loop: per two words (16 weights / lane): 2 masks + 2 x (2 rotations + 2 masks)
//   -> 4 e4m3 registers; per octet one k32 MMA into a zeroed temp and 4 FFMA with fp32 scales
//   (bf16x2 scale pair from scale_even, split with one LOP3 + one shift).
// MODE 2: bare bf16 MMA (16 per iteration); MODE 3: bare e4m3 k32 MMA (16 per iteration).
// MODE 4: e4m3 loop with the scale folded per two blocks (an upper bound: no per-block FFMA).
template <int MODE>
__global__ void __launch_bounds__(256) loop_rate(const uint32_t* __restrict__ src, float* sink, int iters) {
  uint32_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = src[(threadIdx.x * 16 + i) & 4095];
  uint32_t sw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) sw[i] = src[(threadIdx.x * 4 + i + 77) & 4095] & 0x3F3F3F3Fu;
  uint32_t b0 = 0x3f803f80u ^ threadIdx.x, b1 = 0x3f003f00u ^ threadIdx.x;
  float acc[2][4] = {};
  for (int it = 0; it < iters; ++it) {
    const uint32_t salt = (uint32_t)it * 0x9E3779B9u;
    if (MODE == 0) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        uint32_t v[4];
        unpack8(w[i] ^ salt, v);
        const uint32_t sp = (i & 1) ? scale_odd(sw[(i / 2) & 3]) : scale_even(sw[(i / 2) & 3]);
        uint32_t a[4];
        a[0] = mul2h<0>(v[0], sp); a[1] = mul2h<1>(v[1], sp); a[2] = mul2h<0>(v[2], sp); a[3] = mul2h<1>(v[3], sp);
        mma_bf16(acc[i & 1], a, b0, b1);
      }
    } else if (MODE == 1 || MODE == 4) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        const uint32_t q0 = w[i] ^ salt, q1 = w[i + 1] ^ salt;
        const uint32_t M = 0x9C9C9C9Cu, X = 0x8C8C8C8Cu, Y = 0x10101010u;
        uint32_t e[4];
        e[0] = q0 & M;
        e[1] = q1 & M;
        e[2] = (__funnelshift_l(q0, q0, 1) & X) | (__funnelshift_l(q1, q1, 4) & Y);
        e[3] = (__funnelshift_l(q1, q1, 1) & X) | (__funnelshift_l(q0, q0, 4) & Y);
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // two octets per word pair: (e0, e2) and (e1, e3)
          if (MODE == 1) {
            float d[4];
            mma_e4m3(d, e[h], e[2 + h], b0, b1);
            const uint32_t sp = h ? scale_odd(sw[(i / 2) & 3]) : scale_even(sw[(i / 2) & 3]);
            const float sg = __uint_as_float(sp << 16), sg8 = __uint_as_float(sp & 0xFFFF0000u);
            acc[h][0] = fmaf(sg, d[0], acc[h][0]);
            acc[h][1] = fmaf(sg, d[1], acc[h][1]);
            acc[h][2] = fmaf(sg8, d[2], acc[h][2]);
            acc[h][3] = fmaf(sg8, d[3], acc[h][3]);
          } else {
            mma_e4m3_acc(acc[h], e[h], e[2 + h], b0, b1);
          }
        }
      }
    } else if (MODE == 2) {
      uint32_t a[4] = {w[0] ^ salt, w[1], w[2], w[3]};
#pragma unroll
      for (int i = 0; i < 16; ++i) mma_bf16(acc[i & 1], a, b0, b1);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) mma_e4m3_acc(acc[i & 1], w[0] ^ salt, w[1], b0, b1);
    }
  }
  float s = 0.f;
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int j = 0; j < 4; ++j) s += acc[h][j];
  if (s == 1234.5f) sink[0] = s;
}

// Returns ms for one launch of grid x 256 threads, iters iterations.
double loop_run(int64_t mode, int64_t grid, int64_t iters) {
  static torch::Tensor src, sink;
  if (!src.defined()) {
    src = torch::randint(0, 1 << 30, {4096}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    sink = torch::zeros({4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }
  const uint32_t* s = reinterpret_cast<const uint32_t*>(src.data_ptr());
  float* k = sink.data_ptr<float>();
  auto launch = [&]() {
    if (mode == 0) loop_rate<0><<<grid, 256>>>(s, k, (int)iters);
    if (mode == 1) loop_rate<1><<<grid, 256>>>(s, k, (int)iters);
    if (mode == 2) loop_rate<2><<<grid, 256>>>(s, k, (int)iters);
    if (mode == 3) loop_rate<3><<<grid, 256>>>(s, k, (int)iters);
    if (mode == 4) loop_rate<4><<<grid, 256>>>(s, k, (int)iters);
  };
  launch();
  cudaDeviceSynchronize();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    best = ms < best ? ms : best;
  }
  return best;
}
"""
_FP8_CPP = "double loop_run(int64_t mode, int64_t grid, int64_t iters);\n"


def run_fp8_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="okc_fp8_probe", cpp_sources=_FP8_CPP, cuda_sources=_FP8_SRC, functions=["loop_run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"fp8 probe: compiled in {time.time() - t0:.0f} s", flush=True)
    iters = 4000
    names = {0: "bf16 loop (unpack8+scale+HMUL2+k16)", 1: "e4m3 loop (dup-k32 + FFMA scale)",
             2: "bare bf16 m16n8k16", 3: "bare e4m3 m16n8k32", 4: "e4m3 loop, no scale (bound)"}
    for mode in (0, 1, 4, 2, 3):
        for cps in (2, 4, 6, 8):   # CTAs of 8 warps per SM
            grid = 132 * cps
            ms = mod.loop_run(mode, grid, iters)
            # 16 octets (or MMAs) per warp per iteration; SM-clocks at 1.755 GHz (the loaded clock)
            octets = grid * 8 * iters * 16
            clk = ms * 1e-3 * 1.755e9 * 132
            print(f"fp8 probe {names[mode]:<38} {cps} CTAs/SM: {ms:.3f} ms, {clk / (octets * 256):.5f} SM-clk per weight"
                  f" ({clk / octets:.2f} SM-clk per warp-octet)", flush=True)
