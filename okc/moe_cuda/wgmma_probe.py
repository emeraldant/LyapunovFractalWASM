# wgmma probe: run in a host's prepare() on the judge's H200 (tools/make_host.py). Settles,
# by measurement rather than by memory of the PTX docs, the two conventions the v7 MoE kernel
# depends on:
#   1. wgmma's register A fragment (m64nNk16, bf16) = mma.sync m16n8k16's A fragment per warp,
#      warp w of the warpgroup owning rows 16w .. 16w + 15;
#   2. the shared-memory descriptor of a K-major, no-swizzle B tile stored as core matrices
#      [k segment (8 k)][8-token block][8 rows][16 B]: which of LBO / SBO is the K-direction
#      stride and which the N-direction stride.
# Also: the D fragment layout (assumed mma.sync's C layout per 8-column block).

import torch

_WG_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}

template <int N>
__device__ __forceinline__ void wgmma_bf16(float* d, const uint32_t a[4], uint64_t desc);

template <>
__device__ __forceinline__ void wgmma_bf16<8>(float* d, const uint32_t a[4], uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <>
__device__ __forceinline__ void wgmma_bf16<16>(float* d, const uint32_t a[4], uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// A: [64, KS*16] bf16 row-major; B: [N, KS*16] bf16 (tokens x k); D: [64, N] f32.
template <int N>
__global__ void probe_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D, int KS, int kdir_is_lbo) {
  __shared__ __align__(128) uint8_t sB[8 * N * 64 * 2];   // up to KS = 8 steps
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, r = lane / 4, c = (lane % 4) * 2;
  // B into core matrices: [k segment ks][token block nb][row 0-7][16 B].
  const int nseg = KS * 2;
  for (int i = tid; i < N * nseg; i += 128) {
    const int n = i / nseg, ks = i % nseg;
    const uint4 v = *reinterpret_cast<const uint4*>(B + (size_t)n * KS * 16 + ks * 8);
    *reinterpret_cast<uint4*>(sB + ks * (N / 8) * 128 + (n / 8) * 128 + (n % 8) * 16) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sB);
  const uint32_t kstride = (N / 8) * 128, nstride = 128;
  float d[N / 2];
  for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
  for (int s = 0; s < KS; ++s) {
    const int row = 16 * warp + r;
    const uint32_t* Ar = reinterpret_cast<const uint32_t*>(A);
    const int ld = KS * 16 / 2;   // u32 per row
    uint32_t a[4];
    a[0] = Ar[row * ld + (s * 16 + c) / 2];
    a[1] = Ar[(row + 8) * ld + (s * 16 + c) / 2];
    a[2] = Ar[row * ld + (s * 16 + c + 8) / 2];
    a[3] = Ar[(row + 8) * ld + (s * 16 + c + 8) / 2];
    const uint64_t desc = kdir_is_lbo ? make_desc(base + 2 * s * kstride, kstride, nstride)
                                      : make_desc(base + 2 * s * kstride, nstride, kstride);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_bf16<N>(d, a, desc);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  // D assumed: per 8-column block j: d[4j], d[4j+1] = (row r, cols 8j + c, c + 1); d[4j+2..3] row r + 8.
  for (int j = 0; j < N / 8; ++j) {
    const int row = 16 * warp + r;
    D[row * N + 8 * j + c] = d[4 * j];
    D[row * N + 8 * j + c + 1] = d[4 * j + 1];
    D[(row + 8) * N + 8 * j + c] = d[4 * j + 2];
    D[(row + 8) * N + 8 * j + c + 1] = d[4 * j + 3];
  }
}

void probe(torch::Tensor A, torch::Tensor B, torch::Tensor D, int64_t n, int64_t kdir_is_lbo) {
  const int KS = A.size(1) / 16;
  auto a = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
  auto b = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr());
  if (n == 8) probe_kernel<8><<<1, 128>>>(a, b, D.data_ptr<float>(), KS, (int)kdir_is_lbo);
  else probe_kernel<16><<<1, 128>>>(a, b, D.data_ptr<float>(), KS, (int)kdir_is_lbo);
}
"""

_WG_CPP = "void probe(torch::Tensor A, torch::Tensor B, torch::Tensor D, int64_t n, int64_t kdir_is_lbo);"


def run_wgmma_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_wgmma_probe", cpp_sources=_WG_CPP, cuda_sources=_WG_SRC, functions=["probe"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"wgmma probe: compiled in {time.time() - t:.0f} s")
    g = torch.Generator(device="cuda").manual_seed(0)
    for n in (8, 16):
        for ks in (1, 4):
            A = torch.randn(64, ks * 16, generator=g, device="cuda").to(torch.bfloat16)
            B = torch.randn(n, ks * 16, generator=g, device="cuda").to(torch.bfloat16)
            ref = A.float() @ B.float().T
            for kdir_is_lbo in (1, 0):
                D = torch.full((64, n), float("nan"), device="cuda")
                mod.probe(A, B, D, n, kdir_is_lbo)
                torch.cuda.synchronize()
                err = float((D - ref).norm() / ref.norm())
                print(f"wgmma probe n{n} ksteps{ks} K-stride in {'LBO' if kdir_is_lbo else 'SBO'}: rel err {err:.2e}"
                      f"{'  <- matches' if err < 1e-5 else ''}")
