# Upper bound for a TRT-GEMV-style expert kernel on the judge's H200: weights streamed by LDG.128
# straight into registers (no shared memory, no barriers), our 8-op FP4 decode + block-scale
# multiply, then T tokens of FMAs per decoded bf16x2 pair (activations held in registers: perfect
# reuse assumed, so this is an upper bound). ACC 0: bf16x2 HFMA2 accumulators (TRT's choice);
# ACC 1: fp32 FFMA accumulators (two per pair, bf16 -> f32 by shift). Timed like the judge: CUDA
# graph replay right after the 256 MB zero-fill flush, replay overhead subtracted (SimBench).
import torch

_GEMV_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void unpack8g(uint32_t q, uint32_t out[4]) {
  const uint32_t M = 0x81C081C0u;
  out[0] = q & M;
  out[1] = __funnelshift_l(q, q, 10) & M;
  out[2] = __funnelshift_l(q, q, 13) & M;
  out[3] = ((q & 0x001B001Bu) * 0x4108u) & M;
}

template <int T, int ACC, int U>
__global__ void __launch_bounds__(256) gemv_stream(const uint4* __restrict__ w, const uint32_t* __restrict__ sc,
                                                  size_t n16, float* sink) {
  // Each thread: U uint4 loads in flight per step (32 weights each), grid-stride over n16.
  uint32_t xa[T];
#pragma unroll
  for (int t = 0; t < T; ++t) xa[t] = 0x3f803f80u + t;
  __nv_bfloat162 hacc[T];
  float facc[T][2];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    hacc[t] = __floats2bfloat162_rn(0.f, 0.f);
    facc[t][0] = facc[t][1] = 0.f;
  }
  const size_t stride = (size_t)gridDim.x * blockDim.x * U;
  for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * U; i < n16; i += stride) {
    uint4 q[U];
    uint32_t s[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      q[u] = __ldcs(w + i + u);
      s[u] = __ldcs(sc + (i + u) / 2);   // one 32-bit scale word per 2 x 32 weights (2 blocks of 16 per uint4 half)
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint32_t words[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t v[4];
        unpack8g(words[j], v);
        const uint32_t sp = ((s[u] >> (8 * j)) & 0xFFu) * 0x00010001u | 0x3c003c00u;   // stand-in scale pair
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          __nv_bfloat162 a = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&v[p]), *reinterpret_cast<const __nv_bfloat162*>(&sp));
#pragma unroll
          for (int t = 0; t < T; ++t) {
            if (ACC == 0) {
              hacc[t] = __hfma2(a, *reinterpret_cast<const __nv_bfloat162*>(&xa[t]), hacc[t]);
            } else {
              const uint32_t ab = *reinterpret_cast<uint32_t*>(&a);
              facc[t][0] = fmaf(__uint_as_float(ab << 16), __uint_as_float(xa[t] << 16), facc[t][0]);
              facc[t][1] = fmaf(__uint_as_float(ab & 0xFFFF0000u), __uint_as_float(xa[t] & 0xFFFF0000u), facc[t][1]);
            }
          }
        }
      }
    }
  }
  float r = 0.f;
#pragma unroll
  for (int t = 0; t < T; ++t) r += ACC == 0 ? __bfloat162float(hacc[t].x) + __bfloat162float(hacc[t].y) : facc[t][0] + facc[t][1];
  if (r == 1234.5f) sink[0] = r;
}

static torch::Tensor g_w, g_s, g_sink;
void gemv_setup(int64_t bytes) {
  g_w = torch::ones({bytes}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_s = torch::ones({bytes / 8 + 64}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_sink = torch::zeros({4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
}
void gemv_launch(int64_t t, int64_t acc, int64_t ctas_per_sm, int64_t bytes) {
  auto st = at::cuda::getCurrentCUDAStream();
  const size_t n16 = (size_t)bytes / 16;
  const uint4* w = reinterpret_cast<const uint4*>(g_w.data_ptr());
  const uint32_t* s = reinterpret_cast<const uint32_t*>(g_s.data_ptr());
  float* sk = g_sink.data_ptr<float>();
  const int grid = 132 * (int)ctas_per_sm;
#define GL(T_, A_) if (t == T_ && acc == A_) { gemv_stream<T_, A_, 4><<<grid, 256, 0, st>>>(w, s, n16, sk); return; }
  GL(1, 0) GL(2, 0) GL(4, 0) GL(8, 0) GL(1, 1) GL(2, 1) GL(4, 1) GL(8, 1)
}
"""
_GEMV_CPP = "void gemv_setup(int64_t bytes);\nvoid gemv_launch(int64_t t, int64_t acc, int64_t ctas_per_sm, int64_t bytes);\n"


def run_gemv_probe(sizes_mb=(132.0, 264.0, 523.0)):
    import time
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="okc_gemv_probe", cpp_sources=_GEMV_CPP, cuda_sources=_GEMV_SRC,
                      functions=["gemv_setup", "gemv_launch"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"gemv probe: compiled in {time.time() - t0:.0f} s", flush=True)
    bench = SimBench()
    tiny = torch.zeros(1, device="cuda")
    mod.gemv_setup(int(max(sizes_mb) * 1e6) // 512 * 512)
    for mb in sizes_mb:
        nbytes = int(mb * 1e6) // 512 * 512
        for acc in (0, 1):
            line = []
            for t in (1, 2, 4, 8):
                best = None
                for cps in (4, 8):
                    us = bench.compare(lambda t=t, acc=acc, cps=cps: mod.gemv_launch(t, acc, cps, nbytes),
                                       lambda: tiny.add_(1), pairs=8)[0]
                    if best is None or us < best[0]:
                        best = (us, cps)
                line.append(f"T{t} {best[0]:.1f} us ({nbytes / best[0] / 1e6:.2f} TB/s, {best[1]} CTA/SM)")
            print(f"gemv {mb:.0f} MB {'bf16x2 HFMA2' if acc == 0 else 'fp32 FFMA'}: " + "; ".join(line), flush=True)
