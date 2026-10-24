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

// Overlap probe: per iteration each warp does a decode-like ALU block (64 LOP3 + 32 HMUL2
// over 8 independent chains) and/or its warpgroup issues 4 wgmma m64n16k16 from registers.
// MODE bits: 1 = ALU block, 2 = wgmma (fence, 4 x wgmma, commit, wait 0), 4 = wait 1 (async).
template <int MODE>
__global__ void overlap_probe(float* sink, int iters) {
  __shared__ __align__(128) uint8_t sB[16 * 256];
  for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) reinterpret_cast<uint32_t*>(sB)[i] = 0x3f803f80u;
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const uint64_t desc = desc0((uint32_t)__cvta_generic_to_shared(sB));
  uint32_t x[8];
  for (int i = 0; i < 8; ++i) x[i] = threadIdx.x * 8 + i;
  const uint32_t m = 0x3f803f81u;
  float d[8] = {};
  for (int it = 0; it < iters; ++it) {
    if (MODE & 17) {   // 1: LOP3 + HMUL2 block; 16: LOP3 part only
#pragma unroll
      for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) asm volatile("lop3.b32 %0, %0, 0x81C081C0, %0, 0x6a;" : "+r"(x[i]));
    }
    if (MODE & 33) {   // 1: LOP3 + HMUL2 block; 32: HMUL2 part only
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(x[i]) : "r"(m));
    }
    if (MODE & 8) {   // the same MACs per warp as 4 x m64n16k16 per warpgroup: 8 x m16n8k16
#pragma unroll
      for (int k = 0; k < 8; ++k)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[4 * (k & 1)]), "+f"(d[4 * (k & 1) + 1]), "+f"(d[4 * (k & 1) + 2]), "+f"(d[4 * (k & 1) + 3])
                     : "r"(x[k]), "r"(x[(k + 1) % 8]), "r"(x[(k + 2) % 8]), "r"(x[(k + 3) % 8]), "r"(x[(k + 4) % 8]), "r"(m));
    }
    if (MODE & 2) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < 4; ++k)
        asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                     : "r"(x[2 * k]), "r"(x[2 * k + 1]), "r"(x[(2 * k + 2) % 8]), "r"(x[(2 * k + 3) % 8]), "l"(desc));
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      if (MODE & 4) asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= x[i];
  float f = 0.f;
  for (int i = 0; i < 8; ++i) f += d[i];
  if (s == 12345u || f == 12345.f) sink[0] = f;
}

// Warp-specialized overlap probe: even warpgroups issue wgmma (A from registers: RS, or from
// shared memory: SS), odd warpgroups run the decode-like ALU block. KIND: 0 = both roles,
// 1 = only the wgmma role (odd WGs idle), 2 = only the ALU role (even WGs idle).
template <int N> struct WgAcc { float d[N / 2]; };

template <int N, bool SS>
__device__ __forceinline__ void wg_step(float* d, const uint32_t* a, uint64_t da, uint64_t db) {
  if (N == 16 && !SS)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
  if (N == 16 && SS)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da), "l"(db));
  if (N == 64 && SS)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                   "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db));
  if (N == 64 && !SS)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                   "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

template <int KIND, int N, bool SS, int OPS = 0>
__global__ void spec_probe(float* sink, int iters, int alu_reps) {
  __shared__ __align__(128) uint8_t sA[64 * 64 * 2], sB[64 * 64 * 2];
  for (int i = threadIdx.x; i < 64 * 64 / 2; i += blockDim.x) {
    reinterpret_cast<uint32_t*>(sA)[i] = 0x3f803f80u;
    reinterpret_cast<uint32_t*>(sB)[i] = 0x3f803f80u;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const uint64_t da = desc0((uint32_t)__cvta_generic_to_shared(sA)), db = desc0((uint32_t)__cvta_generic_to_shared(sB));
  const int wg = threadIdx.x / 128;
  const bool mma_role = (wg % 2) == 0;
  uint32_t x[8];
  for (int i = 0; i < 8; ++i) x[i] = threadIdx.x * 8 + i;
  const uint32_t m = 0x3f803f81u;
  float d[N / 2];
  for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
  if (mma_role && KIND != 2) {
    for (int it = 0; it < iters; ++it) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < 4; ++k) wg_step<N, SS>(d, x + (k & 1) * 4, da, db);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  if (!mma_role && KIND != 1) {
    // OPS 0: 64 LOP3 + 32 HMUL2 (the decode's mix); 1: 96 LOP3; 2: 96 HMUL2 (bf16x2);
    // 3: 96 IMAD; 4: 96 FFMA (fp32); per warp per rep.
    float fx[8];
    for (int i = 0; i < 8; ++i) fx[i] = (float)x[i];
    for (int it = 0; it < iters * alu_reps; ++it) {
      if (OPS == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r)
#pragma unroll
          for (int i = 0; i < 8; ++i) asm volatile("lop3.b32 %0, %0, 0x81C081C0, %0, 0x6a;" : "+r"(x[i]));
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int i = 0; i < 8; ++i) asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(x[i]) : "r"(m));
      } else {
#pragma unroll
        for (int r = 0; r < 12; ++r)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if (OPS == 1) asm volatile("lop3.b32 %0, %0, 0x81C081C0, %0, 0x6a;" : "+r"(x[i]));
            if (OPS == 2) asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(x[i]) : "r"(m));
            if (OPS == 3) asm volatile("mad.lo.u32 %0, %0, 0x4108, %1;" : "+r"(x[i]) : "r"(m));
            if (OPS == 4) asm volatile("fma.rn.f32 %0, %0, 0f3F800001, 0f00000001;" : "+f"(fx[i]));
          }
      }
    }
    for (int i = 0; i < 8; ++i) x[i] ^= __float_as_uint(fx[i]);
  }
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= x[i];
  float f = 0.f;
  for (int i = 0; i < N / 2; ++i) f += d[i];
  if (s == 12345u || f == 12345.f) sink[0] = f;
}

double run_spec(int kind, int n, int ss, int blocks, int threads, int iters, int alu_reps, torch::Tensor sink, int ops) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto go = [&]() {
    float* p = sink.data_ptr<float>();
#define SPECC(K_, N_, S_) if (kind == K_ && n == N_ && ss == S_ && ops == 0) spec_probe<K_, N_, S_><<<blocks, threads>>>(p, iters, alu_reps);
    SPECC(0, 16, 0) SPECC(1, 16, 0) SPECC(2, 16, 0) SPECC(0, 16, 1) SPECC(1, 16, 1)
    SPECC(0, 64, 0) SPECC(1, 64, 0) SPECC(0, 64, 1) SPECC(1, 64, 1)
#define SPECO(K_, N_, S_, O_) if (kind == K_ && n == N_ && ss == S_ && ops == O_) spec_probe<K_, N_, S_, O_><<<blocks, threads>>>(p, iters, alu_reps);
#define SPECOS(O_) SPECO(0, 16, 0, O_) SPECO(2, 16, 0, O_) SPECO(0, 64, 0, O_) SPECO(0, 64, 1, O_)
    SPECOS(1) SPECOS(2) SPECOS(3) SPECOS(4)
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

double run_overlap(int mode, int blocks, int threads, int iters, torch::Tensor sink) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto go = [&]() {
    float* p = sink.data_ptr<float>();
    if (mode == 1) overlap_probe<1><<<blocks, threads>>>(p, iters);
    if (mode == 2) overlap_probe<2><<<blocks, threads>>>(p, iters);
    if (mode == 3) overlap_probe<3><<<blocks, threads>>>(p, iters);
    if (mode == 7) overlap_probe<7><<<blocks, threads>>>(p, iters);
    if (mode == 8) overlap_probe<8><<<blocks, threads>>>(p, iters);
    if (mode == 9) overlap_probe<9><<<blocks, threads>>>(p, iters);
    if (mode == 16) overlap_probe<16><<<blocks, threads>>>(p, iters);
    if (mode == 18) overlap_probe<18><<<blocks, threads>>>(p, iters);
    if (mode == 32) overlap_probe<32><<<blocks, threads>>>(p, iters);
    if (mode == 34) overlap_probe<34><<<blocks, threads>>>(p, iters);
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

// Decode microbenchmark: the MoE kernel's real per-warp-chunk decode (2 groups x 4 words ->
// 16 A fragments: unpack, scale build, HMUL2) on register data, no memory. VAR 0: current
// unpack (rotations by SHF); 1: rotations by IMAD.WIDE + one LOP3 ((lo | hi) & M). MMA: also
// 8 x mma.sync m16n8k16 per chunk (8 tokens). Reports SM-cycles per weight.
template <int VAR>
__device__ __forceinline__ void unpack8v(uint32_t q, uint32_t out[4], uint32_t k10, uint32_t k13) {
  const uint32_t M = 0x81C081C0u;
  out[0] = q & M;
  if ((VAR & 3) == 0) {
    out[1] = __funnelshift_l(q, q, 10) & M;
    out[2] = __funnelshift_l(q, q, 13) & M;
  } else if ((VAR & 3) == 2) {   // rotations on the FMA pipe: lo / hi words of q * 2^r, merged by one LOP3
    out[1] = ((q * k10) | __umulhi(q, k10)) & M;
    out[2] = ((q * k13) | __umulhi(q, k13)) & M;
  } else if ((VAR & 3) == 3) {   // one rotation each way: balance the ALU and FMA pipes
    out[1] = ((q * k10) | __umulhi(q, k10)) & M;
    out[2] = __funnelshift_l(q, q, 13) & M;
  } else {
    uint64_t w1, w2;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(w1) : "r"(q), "r"(k10));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(w2) : "r"(q), "r"(k13));
    out[1] = ((uint32_t)w1 | (uint32_t)(w1 >> 32)) & M;
    out[2] = ((uint32_t)w2 | (uint32_t)(w2 >> 32)) & M;
  }
  out[3] = ((q & 0x001B001Bu) * 0x4108u) & M;
}
__device__ __forceinline__ uint32_t sc_even(uint32_t w) { uint32_t r; asm("mad.lo.u32 %0, %1, 16, %2;" : "=r"(r) : "r"(w & 0x00FF00FFu), "r"((238u << 7) | (238u << 23))); return r; }
__device__ __forceinline__ uint32_t sc_even_r(uint32_t w, uint32_t k16) { uint32_t r; asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(w & 0x00FF00FFu), "r"(k16), "r"((238u << 7) | (238u << 23))); return r; }
__device__ __forceinline__ uint32_t sc_odd(uint32_t w) { uint32_t r; asm("mad.hi.u32 %0, %1, 268435456, %2;" : "=r"(r) : "r"(w & 0xFF00FF00u), "r"((238u << 7) | (238u << 23))); return r; }
__device__ __forceinline__ uint32_t hmul2h(uint32_t a, uint32_t s, int h) {
  __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h == 0 ? __low2bfloat162(sv) : __high2bfloat162(sv));
  return *reinterpret_cast<uint32_t*>(&r);
}
template <int VAR, bool MMA>
__global__ void decode_bench(float* sink, int iters, uint32_t k10, uint32_t k13) {
  uint32_t wv[2][4], sv[2][2];
  for (int j = 0; j < 2; ++j) { for (int i = 0; i < 4; ++i) wv[j][i] = threadIdx.x * 977u + j * 31 + i; sv[j][0] = 0x40404040u; sv[j][1] = 0x41414141u; }
  uint32_t accu = 0;
  float c[2][4] = {};
  const uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f81u;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t v[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j) unpack8v<VAR>(wv[j][w] ^ it, v[j], k10, k13);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int step = 2 * w + half;
        uint32_t a[4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint32_t sw = w < 2 ? sv[j][0] : sv[j][1];
          const uint32_t sp = half == 0 ? ((VAR & 4) ? sc_even_r(sw, k10 >> 6) : sc_even(sw)) : sc_odd(sw);
          const int h = (step % 4) < 2 ? 0 : 1;
          a[j] = hmul2h(v[j][2 * half], sp, h);
          a[2 + j] = hmul2h(v[j][2 * half + 1], sp, h);
        }
        if (MMA) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(c[half][0]), "+f"(c[half][1]), "+f"(c[half][2]), "+f"(c[half][3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        } else {
          accu ^= a[0] ^ a[1] ^ a[2] ^ a[3];
        }
      }
    }
  }
  if (accu == 12345u || c[0][0] + c[1][1] == 12345.f) sink[0] = (float)accu;
}

double run_decode(int var, int mma, int blocks, int threads, int iters, torch::Tensor sink) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto go = [&]() {
    float* p = sink.data_ptr<float>();
#define DCASE(V_) if (var == V_ && !mma) decode_bench<V_, false><<<blocks, threads>>>(p, iters, 1u << 10, 1u << 13); \
                  if (var == V_ && mma) decode_bench<V_, true><<<blocks, threads>>>(p, iters, 1u << 10, 1u << 13);
    DCASE(0) DCASE(1) DCASE(2) DCASE(3) DCASE(4) DCASE(6) DCASE(7)
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

_RATE_CPP = ("double run(int which, int n, int blocks, int threads, int iters, torch::Tensor sink);\n"
             "double run_overlap(int mode, int blocks, int threads, int iters, torch::Tensor sink);\n"
             "double run_spec(int kind, int n, int ss, int blocks, int threads, int iters, int alu_reps, torch::Tensor sink, int ops);\n"
             "double run_decode(int var, int mma, int blocks, int threads, int iters, torch::Tensor sink);")


def run_rate_probe():
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_rate_probe", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC, functions=["run", "run_overlap", "run_decode"],
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


def run_overlap_probe():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_rate_probe", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC, functions=["run", "run_overlap", "run_decode"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    sink = torch.zeros(4, device="cuda")
    clk, sms, iters = 1.98e9, 132, 2000
    for wgs_per_cta, ctas in ((2, 4), (2, 2)):
        res = {}
        for mode, name in ((1, "alu"), (2, "wgmma"), (3, "alu+wgmma"), (8, "mma.sync"), (9, "alu+mma.sync"),
                           (16, "lop3"), (18, "lop3+wgmma"), (32, "hmul2"), (34, "hmul2+wgmma")):
            ms = mod.run_overlap(mode, sms * ctas, 128 * wgs_per_cta, iters, sink)
            res[name] = ms * 1e-3 * clk / iters
        print(f"overlap {wgs_per_cta * ctas} WG/SM: clocks per iteration (all WGs): "
              + ", ".join(f"{k} {v:.0f}" for k, v in res.items())
              + f"; sums: alu+wgmma {res['alu'] + res['wgmma']:.0f}, lop3+wgmma {res['lop3'] + res['wgmma']:.0f},"
              f" hmul2+wgmma {res['hmul2'] + res['wgmma']:.0f}")


def run_decode_probe():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="okc_rate_probe", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC,
                      functions=["run", "run_overlap", "run_decode"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    sink = torch.zeros(4, device="cuda")
    clk, sms, iters = 1.98e9, 132, 4000
    hbm = 1.0 / (4.49e12 / 0.5625 / sms / clk)   # SM-cycles per weight at the HBM roofline (4.49 TB/s measured)
    # var: rotations 0 SHF, 1 IMAD.WIDE, 2 IMAD lo + IMAD.HI (FMA pipe), 3 one of each; +4: scale via IMAD
    for warps in (16, 32):
        for var in (0, 2, 3, 4, 6, 7):
            for mma in (0, 1):
                ms = mod.run_decode(var, mma, sms, 32 * warps, iters, sink)
                weights = sms * warps * iters * 2048
                cyc = ms * 1e-3 * clk * sms / weights
                print(f"decode var{var} {'+mma' if mma else '    '} {warps} warps/SM: {cyc:.4f} SM-cycles/weight"
                      f" ({100 * cyc / hbm:.0f}% of the HBM budget {hbm:.4f})")


def run_spec_probe():
    """Does one warpgroup's decode-like ALU work overlap another warpgroup's wgmma? (RS: A from
    registers, SS: A from shared memory; N 16 and 64.) Per iteration: 4 wgmma m64nNk16 per
    mma warpgroup, alu_reps x (64 LOP3 + 32 HMUL2) per warp of each ALU warpgroup."""
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_rate_probe", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC,
                      functions=["run", "run_overlap", "run_decode", "run_spec"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"spec probe: compiled in {time.time() - t:.0f} s", flush=True)
    sink = torch.zeros(4, device="cuda")
    clk, sms, iters = 1.98e9, 132, 2000
    for n, ss, reps in ((16, 0, 1), (16, 1, 1), (64, 0, 2), (64, 1, 2), (64, 1, 4)):
        for wgs, ctas in ((2, 2), (4, 1)):
            r = {}
            for kind in ((0, 1, 2) if (n, ss) == (16, 0) or True else (0, 1)):
                if kind == 2 and n != 16:
                    continue
                ms = mod.run_spec(kind, n, ss, sms * ctas, 128 * wgs, iters, reps, sink, 0)
                r[kind] = ms * 1e-3 * clk / iters
            if 2 not in r:
                r[2] = mod.run_spec(2, 16, 0, sms * ctas, 128 * wgs, iters, reps, sink, 0) * 1e-3 * clk / iters
            print(f"spec {'SS' if ss else 'RS'} N{n} alu x{reps}, {wgs * ctas} WG/SM: clocks/iter both {r[0]:.0f},"
                  f" wgmma only {r[1]:.0f}, alu only {r[2]:.0f} (max {max(r[1], r[2]):.0f}, sum {r[1] + r[2]:.0f})", flush=True)


def run_spec_ops_probe():
    """Which instruction types overlap another warpgroup's wgmma? Per iteration: 4 wgmma m64nNk16
    per mma warpgroup; per ALU warp 96 ops of one kind (or the decode's 64 LOP3 + 32 HMUL2).
    Reported in SM clocks per iteration: both, wgmma only, ALU only."""
    import time
    from torch.utils.cpp_extension import load_inline
    t = time.time()
    mod = load_inline(name="okc_rate_probe_ops", cpp_sources=_RATE_CPP, cuda_sources=_RATE_SRC,
                      functions=["run", "run_overlap", "run_decode", "run_spec"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"spec ops probe: compiled in {time.time() - t:.0f} s", flush=True)
    sink = torch.zeros(4, device="cuda")
    clk, sms, iters = 1.98e9, 132, 2000
    names = {0: "decode mix (64 LOP3 + 32 HMUL2)", 1: "LOP3 x96", 2: "HMUL2 bf16x2 x96", 3: "IMAD x96", 4: "FFMA x96"}
    mma_only = {}
    for n, ss in ((16, 0), (64, 0), (64, 1)):
        mma_only[(n, ss)] = mod.run_spec(1, n, ss, sms * 2, 256, iters, 1, sink, 0) * 1e-3 * clk / iters
    for ops in (0, 1, 2, 3, 4):
        alu = mod.run_spec(2, 16, 0, sms * 2, 256, iters, 1, sink, ops) * 1e-3 * clk / iters
        for n, ss in ((16, 0), (64, 0), (64, 1)):
            both = mod.run_spec(0, n, ss, sms * 2, 256, iters, 1, sink, ops) * 1e-3 * clk / iters
            w = mma_only[(n, ss)]
            hidden = (w + alu - both) / min(w, alu) if min(w, alu) > 0 else 0
            print(f"spec ops {names[ops]:<32} vs {'SS' if ss else 'RS'} N{n}: both {both:.0f}, wgmma only {w:.0f},"
                  f" alu only {alu:.0f} (max {max(w, alu):.0f}, sum {w + alu:.0f}; hidden {100 * hidden:.0f}% of the smaller)",
                  flush=True)
