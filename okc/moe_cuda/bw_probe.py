# HBM read-bandwidth probe on the judge's H200 (run in any host's prepare()): what the MoE
# kernels' memory path can reach, measured instead of assumed. The v6 kernel streams each
# work item's weights as 8 KB + 1 KB cp.async.bulk copies into 4 stages x 4 CTAs per SM and
# its weights-only ablation tops out at 3.3-3.4 TB/s; is that the access pattern's ceiling?
#   ldg:  plain LDG.128 grid-stride streaming (the reference ceiling)
#   tma:  one producer lane per CTA, STAGES stages of CHUNK (+ EXTRA from a second buffer),
#         consumers wait / (optionally) read the stage with LDS.128 / release

import torch

_BW_SRC = r"""
#include <torch/extension.h>
#include <stdint.h>

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!__all_sync(0xffffffffu, mbar_try(bar, parity))) {
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__global__ void ldg_stream(const uint4* __restrict__ src, size_t n, uint32_t* sink) {
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

// Warp 0 lane 0 produces; warps 1..NC consume. Each CTA streams its own contiguous range.
__global__ void tma_stream(const uint8_t* __restrict__ src, const uint8_t* __restrict__ src2, int nchunks,
                           int chunk, int extra, int stages, int lds, uint32_t* sink, int hold, int pair) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const int stage_bytes = chunk + extra;
  const uint32_t bar_full = base + stages * stage_bytes, bar_empty = bar_full + 8 * stages;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nc = blockDim.x / 32 - 1;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, hold < 0 ? nc : nc * 32);   // hold < 0: one arrive per warp
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const uint8_t* my = src + (size_t)blockIdx.x * nchunks * chunk;
  const uint8_t* my2 = src2 + (size_t)blockIdx.x * nchunks * extra;
  if (warp == 0) {
    if (lane == 0)
      for (int k = 0; k < nchunks; ++k) {
        const int st = k % stages;
        if (k >= stages) {
          while (!mbar_try(bar_empty + 8 * st, ((k / stages) - 1) & 1)) {
          }
          if (pair && (k % 2) == 0 && k + 1 < nchunks) {   // issue pairs back to back: wait for the next slot too
            const int st1 = (k + 1) % stages;
            while (!mbar_try(bar_empty + 8 * st1, (((k + 1) / stages) - 1) & 1)) {
            }
          }
        }
        mbar_expect_tx(bar_full + 8 * st, stage_bytes);
        bulk_g2s(base + st * stage_bytes, my + (size_t)k * chunk, chunk, bar_full + 8 * st);
        if (extra) bulk_g2s(base + st * stage_bytes + chunk, my2 + (size_t)k * extra, extra, bar_full + 8 * st);
      }
    return;
  }
  uint32_t x = 0;
  const int t = threadIdx.x - 32;
  for (int k = 0; k < nchunks; ++k) {
    const int st = k % stages;
    mbar_wait(bar_full + 8 * st, (k / stages) & 1);
    if (lds)
      for (int o = t * 16; o < stage_bytes; o += nc * 32 * 16) {
        uint32_t a, b, c, d;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                     : "r"(base + st * stage_bytes + o));
        x ^= a ^ d;
      }
    const int h = hold < 0 ? -hold : hold;
    if (h > 1) {   // emulate the consumers' compute on this stage before releasing it
      const long long t0 = clock64();
      while (clock64() - t0 < h) {
      }
    }
    if (hold < 0) {   // lane 0 only, as a predicated instruction (no divergent branch)
      asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %1, 0;\n @p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n"
                   :: "r"(bar_empty + 8 * st), "r"(lane) : "memory");
    } else {
      mbar_arrive(bar_empty + 8 * st);
    }
  }
  if (x == 0x12345678u) sink[0] = x;
}

static torch::Tensor g_buf, g_buf2, g_sink;

// Returns GB/s (bytes read / kernel time, median of 5 after a warmup).
double run_bw(int mode, int ctas_per_sm, int threads, int chunk, int extra, int stages, int lds, int grid_override, int nch_fixed,
              int hold, int pair) {
  const size_t total = (size_t)1 << 30;
  if (!g_buf.defined()) {
    g_buf = torch::ones({(long)total}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    g_buf2 = torch::ones({(long)(total / 4)}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    g_sink = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  }
  const int grid = grid_override > 0 ? grid_override : 132 * ctas_per_sm;
  size_t bytes = 0;
  int nchunks = 0, smem = 0;
  if (mode == 1) {
    nchunks = (int)(total / ((size_t)grid * chunk));
    if ((size_t)nchunks * grid * extra > total / 4) nchunks = (int)(total / 4 / ((size_t)grid * extra));
    if (nch_fixed > 0) nchunks = nch_fixed;
    smem = stages * (chunk + extra) + 16 * stages;
    cudaFuncSetAttribute(tma_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    bytes = (size_t)nchunks * grid * (chunk + extra);
  } else {
    bytes = total;
  }
  auto launch = [&]() {
    if (mode == 0)
      ldg_stream<<<grid, threads>>>(reinterpret_cast<const uint4*>(g_buf.data_ptr()), total / 16,
                                    reinterpret_cast<uint32_t*>(g_sink.data_ptr()));
    else
      tma_stream<<<grid, threads, smem>>>(g_buf.data_ptr<uint8_t>(), g_buf2.data_ptr<uint8_t>(), nchunks, chunk,
                                          extra, stages, lds, reinterpret_cast<uint32_t*>(g_sink.data_ptr()), hold, pair);
  };
  launch();
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) return -1.0;
  std::vector<float> ts;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  return bytes / (ts[2] * 1e-3) / 1e9;
}
"""

_BW_CPP = ("double run_bw(int mode, int ctas_per_sm, int threads, int chunk, int extra, int stages, int lds, int grid_override, "
           "int nch_fixed, int hold, int pair);")


def run_bw_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_bw_probe", cpp_sources=_BW_CPP, cuda_sources=_BW_SRC, functions=["run_bw"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"bw probe: compiled in {time.time() - t:.0f} s", flush=True)
    # Consumers hold each stage `hold` cycles (the compute on it) before releasing it. v6 at bs 8
    # holds a stage ~1.1 us (~2200 clk) with 4 CTAs/SM; the bytes alone need 1.2 us per stage per
    # CTA at 4.49 TB/s. Rate with no hold = HBM; the question is how fast it falls with the hold.
    # Bytes: 528 CTAs x 32 chunks (bs 8 GEMM1-sized), and 1 GB streams.
    # Is it the copy size or the burst? 8 KB copies issued in back-to-back pairs vs 16 KB copies.
    for cps, stages, chunk, extra, th, pr in ((4, 4, 8192, 1024, 288, 0), (4, 4, 8192, 1024, 288, 1), (4, 2, 16384, 2048, 288, 0),
                                              (4, 4, 8192, 0, 288, 0), (4, 4, 8192, 0, 288, 1), (3, 4, 16384, 0, 288, 0),
                                              (3, 2, 32768, 0, 288, 0), (4, 6, 8192, 1024, 288, 1)):
        line = []
        for per9k in (1, 2000, 2500):
            hold = max(1, per9k * (chunk + extra) // 9216)
            gbs = mod.run_bw(1, cps, th, chunk, extra, stages, 0, 0, 0, hold, pr)
            line.append(f"{per9k}:{gbs:.0f}")
        print(f"bw paced {cps} CTAs x {stages} x {chunk // 1024}K+{extra // 1024}K{' pairs' if pr else ''}"
              f" ({cps * stages * (chunk + extra) // 1024} KB/SM): GB/s by hold clk per 9 KB " + " ".join(line), flush=True)
    return
    # DRAM under pacing: same consumer time per byte (hold scaled with the stage size), bigger
    # copies / fewer streams. ~2000 clk per 9 KB is v6's pace at bs 8.
    for cps, stages, chunk, extra, th in ((4, 4, 8192, 1024, 288), (4, 4, 8192, 0, 288), (4, 2, 16384, 2048, 288),
                                          (4, 2, 16384, 0, 288), (2, 2, 32768, 0, 288), (2, 4, 16384, 0, 288),
                                          (1, 4, 32768, 0, 288), (8, 2, 8192, 0, 64), (4, 4, 4096, 0, 288)):
        line = []
        for per9k in (1, 1000, 2000, 3000):
            hold = max(1, per9k * (chunk + extra) // 9216)
            gbs = mod.run_bw(1, cps, th, chunk, extra, stages, 0, 0, 0, hold, pr)
            line.append(f"{per9k}:{gbs:.0f}")
        print(f"bw paced {cps} CTAs x {stages} x {chunk // 1024}K+{extra // 1024}K ({cps * stages * (chunk + extra) // 1024} KB/SM):"
              f" GB/s by hold clk per 9 KB " + " ".join(line), flush=True)
