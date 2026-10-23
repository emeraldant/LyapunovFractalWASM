# Instruction-rate microbenchmarks on the judge's H200 (run in any host's prepare()): the
# "Hopper facts" the MoE kernel design rests on, measured instead of assumed.
#   wgmma m64nNk16 (A from registers, bf16 -> f32): MAC / clock / SM vs N and warpgroups per SM
#   mma.sync m16n8k16 bf16: MAC / clock / SM vs warps per SM
#   HMUL2.BF16, LOP3: warp-instructions / clock / SM sub-partition

import torch

_RATE_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t desc0(uint32_t a) { return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(256 >> 4) << 16) | ((uint64_t)(128 >> 4) << 32); }

template <int N>
__global__ void wg_rate(float* sink, int iters) {
  __shared__ __align__(128) uint8_t sB[64 * 256];
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) reinterpret_cast<uint32_t*>(sB)[i] = 0x3f803f80u;
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const uint64_t desc = desc0((uint32_t)__cvta_generic_to_shared(sB));
  uint32_t a0 = 0x3f803f80u + threadIdx.x, a1 = a0 ^ 1, a2 = a0 ^ 2, a3 = a0 ^ 3;
  float d[N / 2];
  for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
  for (int it = 0; it < iters; ++it) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (N == 8)
        asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
      if (N == 16)
        asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
      if (N == 32)
        asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                       "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  float s = 0.f;
  for (int i = 0; i < N / 2; ++i) s += d[i];
  if (s == 12345.f) sink[0] = s;
}

__global__ void mma_rate(float* sink, int iters) {
  uint32_t a[4] = {0x3f803f80u + threadIdx.x, 1u, 2u, 3u}, b0 = 0x3f803f80u, b1 = b0 ^ 7;
  float c[4][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < 4; ++k)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  float s = 0.f;
  for (int k = 0; k < 4; ++k) s += c[k][0] + c[k][1] + c[k][2] + c[k][3];
  if (s == 12345.f) sink[0] = s;
}

__global__ void hmul2_rate(float* sink, int iters) {
  uint32_t x[8];
  for (int i = 0; i < 8; ++i) x[i] = 0x3f803f80u + threadIdx.x * 8 + i;
  const uint32_t m = 0x3f803f81u;
  for (int it = 0; it < iters; ++it)
#pragma unroll
    for (int i = 0; i < 8; ++i) asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(x[i]) : "r"(m));
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= x[i];
  if (s == 12345u) sink[0] = s;
}

__global__ void lop3_rate(float* sink, int iters) {
  uint32_t x[8];
  for (int i = 0; i < 8; ++i) x[i] = threadIdx.x * 8 + i;
  for (int it = 0; it < iters; ++it)
#pragma unroll
    for (int i = 0; i < 8; ++i) asm volatile("lop3.b32 %0, %0, 0x81C081C0, %0, 0x6a;" : "+r"(x[i]));
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= x[i];
  if (s == 12345u) sink[0] = s;
}

double run(int which, int n, int blocks, int threads, int iters, torch::Tensor sink) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto go = [&]() {
    float* p = sink.data_ptr<float>();
    if (which == 0 && n == 8) wg_rate<8><<<blocks, threads>>>(p, iters);
    if (which == 0 && n == 16) wg_rate<16><<<blocks, threads>>>(p, iters);
    if (which == 0 && n == 32) wg_rate<32><<<blocks, threads>>>(p, iters);
    if (which == 1) mma_rate<<<blocks, threads>>>(p, iters);
    if (which == 2) hmul2_rate<<<blocks, threads>>>(p, iters);
    if (which == 3) lop3_rate<<<blocks, threads>>>(p, iters);
  };
  go();
  cudaEventRecord(e0);
  go();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  return ms;
}
"""

_RATE_CPP = "double run(int which, int n, int blocks, int threads, int iters, torch::Tensor sink);"


def run_rate_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_rate_probe", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC, functions=["run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"rate probe: compiled in {time.time() - t:.0f} s")
    sink = torch.zeros(4, device="cuda")
    clk, sms = 1.98e9, 132
    for n in (8, 16, 32):
        for wgs in (1, 2, 4, 8):
            iters = 2000
            ms = mod.run(0, n, sms * wgs, 128, iters, sink)
            macs = sms * wgs * iters * 8 * 64 * n * 16
            print(f"rate wgmma m64n{n}k16 RS, {wgs} WG/SM: {macs / (ms * 1e-3) / sms / clk:.0f} MAC/clk/SM"
                  f" ({ms * 1e-3 * clk / (iters * 8 * wgs):.1f} clk per wgmma per SM)")
    for warps in (8, 16, 32):
        iters = 4000
        ms = mod.run(1, 0, sms, 32 * warps, iters, sink)
        macs = sms * warps * iters * 4 * 16 * 8 * 16
        print(f"rate mma.sync m16n8k16, {warps} warps/SM: {macs / (ms * 1e-3) / sms / clk:.0f} MAC/clk/SM")
    for which, name in ((2, "HMUL2.BF16"), (3, "LOP3")):
        for warps in (8, 32):
            iters = 20000
            ms = mod.run(which, 0, sms, 32 * warps, iters, sink)
            inst = sms * warps * iters * 8
            print(f"rate {name}, {warps} warps/SM: {inst / (ms * 1e-3) / sms / 4 / clk:.2f} warp-instr/clk/SMSP")
