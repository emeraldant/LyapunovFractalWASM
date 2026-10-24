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
    its m16n8k16 B-fragment codes for the chunk's 8 k16 steps (bit layout: unpack8);
    GEMM1 groups alternate gate / up so a warp's group pair gives silu(gate) * up directly.
  scales: [expert][n8 group][k chunk][row 8][step 8] E4M3 bytes.
"""
import torch
import triton
import triton.language as tl

MOE_DEVICE_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

#define KTOT 4096

// One lane word -> the four bf16x2 B-fragment operands of two k16 steps, each e2m1 * 2^-126
// (bits m0 m1 m2 at 6-8 / 22-24, sign at 15 / 31). Layout chosen by exhaustive search for the
// fewest instructions (8 per word): out[0] sits in place, out[1] / out[2] are rotations by
// 10 / 13, and out[3]'s scattered bits (0,1,3,4 | 16,17,19,20) are moved by one multiply
// with three shifted copies (3, 8, 14; no carries reach the kept bits: checked for all inputs).
__device__ __forceinline__ void unpack8(uint32_t q, uint32_t out[4]) {
  const uint32_t M = 0x81C081C0u;
  out[0] = q & M;
  out[1] = __funnelshift_l(q, q, 10) & M;
  out[2] = __funnelshift_l(q, q, 13) & M;
  out[3] = ((q & 0x001B001Bu) * 0x4108u) & M;
}

// Byte i of a scale word (E4M3, sign 0, normal: prepare checks) -> bf16x2 broadcast of
// scale * 2^118: PRMT puts the byte in both halves, then one IMAD shifts it into the BF16
// exponent/mantissa fields and adds the exponent bias (238) to both halves.
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
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks) {
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
  // active[] is the compact list of non-empty token chunks, e | (chunk index << 8), so every
  // item has rows: chunks of the same expert and n block are n_blocks items apart and share
  // their weight reads through L2.
  const int n_active = *n_active_ptr;
  const int total = n_active * n_blocks;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int nb = item % n_blocks;
    const int ch = active[item / n_blocks];
    const int e = ch & 255, mc = ch >> 8;
    const int row0 = exp_start[e] + mc * MT * 16;
    const int rows_e = exp_start[e + 1] - row0;
    if (rows_e <= 0) continue;
    const int tiles = min(MT, (rows_e + 15) / 16);
    const float gsc = gs[e] * 256.0f;

    // Streaming pointers, set once per item and advanced by constant strides per chunk.
    const uint32_t smem_u32 = (uint32_t)__cvta_generic_to_shared(smem);
    const uint8_t* a_ptr[MT];
    uint32_t a_dst[MT], a_bytes[MT];
    bool a_live[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      int seg = tid + i * 256;
      int lr = seg / 16, cs = seg % 16;
      a_live[i] = lr < tiles * 16;
      bool ok = lr < rows_e && a_live[i];
      int src = 0;
      if (ok) src = MODE == 1 ? row_route[row0 + lr] / 8 : row0 + lr;
      a_ptr[i] = reinterpret_cast<const uint8_t*>(a_src + (size_t)src * KTOT) + cs * 16;
      a_dst[i] = smem_u32 + lr * 256 + ((cs ^ (lr & 7)) * 16);
      a_bytes[i] = ok ? 16u : 0u;
    }
    const uint8_t* w_ptr[NGW];
    const uint8_t* s_ptr[NGW];
    uint32_t w_dst[NGW], s_dst[NGW];
#pragma unroll
    for (int j = 0; j < NGW; ++j) {
      int grp = MODE == 1 ? 2 * (nb * 8 + warp) + j : nb * 8 + warp;
      size_t gch = ((size_t)e * GROUPS + grp) * NCHUNK;
      w_ptr[j] = reinterpret_cast<const uint8_t*>(wq) + (gch * 32 + lane) * 16;
      s_ptr[j] = ws + gch * 64 + (lane & 3) * 16;
      w_dst[j] = smem_u32 + STAGES * A_STAGE + ((warp * NGW + j) * 32 + lane) * 16;
      s_dst[j] = smem_u32 + STAGES * (A_STAGE + W_STAGE) + (warp * NGW + j) * 64 + (lane & 3) * 16;
    }
    // Issues the next chunk (in order) into stage st, then advances the pointers.
    auto load_next = [&](int st) {
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (a_live[i]) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                       :: "r"(a_dst[i] + st * A_STAGE), "l"(a_ptr[i]), "r"(a_bytes[i]));
        }
        a_ptr[i] += 256;
      }
#pragma unroll
      for (int j = 0; j < NGW; ++j) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(w_dst[j] + st * W_STAGE), "l"(w_ptr[j]));
        if (lane < 4) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_dst[j] + st * S_STAGE), "l"(s_ptr[j]));
        }
        w_ptr[j] += 32 * 16;
        s_ptr[j] += 64;
      }
    };
    // ldmatrix source offsets for each k16 step within a stage (tile mt adds mt * 4096).
    uint32_t lds_off[8];
    {
      const int row = lane % 16, hi = lane / 16;
#pragma unroll
      for (int step = 0; step < 8; ++step) lds_off[step] = row * 256 + (((step * 2 + hi) ^ (row & 7)) * 16);
    }
    const uint32_t w_src = smem_u32 + STAGES * A_STAGE + ((warp * NGW) * 32 + lane) * 16;
    const uint32_t s_src = smem_u32 + STAGES * (A_STAGE + W_STAGE) + (warp * NGW) * 64 + r * 8;

    // MT 1: even and odd k16 steps accumulate separately, halving the HMMA dependency chain.
    constexpr int NACC = MT == 1 ? 2 : 1;
    float acc[NACC][MT][NGW][4];
#pragma unroll
    for (int x = 0; x < NACC; ++x)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NGW; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[x][mt][j][i] = 0.f;

#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
      load_next(st);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    // Unrolled by STAGES so every stage offset is a compile-time constant.
    static_assert(NCHUNK % STAGES == 0, "chunks must divide into stages");
#pragma unroll 1
    for (int kc0 = 0; kc0 < NCHUNK; kc0 += STAGES) {
#pragma unroll
      for (int st = 0; st < STAGES; ++st) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        if (kc0 + st + STAGES - 1 < NCHUNK) load_next((st + STAGES - 1) % STAGES);
        asm volatile("cp.async.commit_group;\n" ::);
        uint4 wv[NGW];
        uint2 sv[NGW];
#pragma unroll
        for (int j = 0; j < NGW; ++j) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(wv[j].x), "=r"(wv[j].y), "=r"(wv[j].z), "=r"(wv[j].w)
                       : "r"(w_src + st * W_STAGE + j * 512));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(sv[j].x), "=r"(sv[j].y)
                       : "r"(s_src + st * S_STAGE + j * 64));
        }
        const uint32_t a_stage = smem_u32 + st * A_STAGE;
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
              uint32_t sw = step < 4 ? sv[j].x : sv[j].y;
              uint32_t s2 = (step % 4 == 0) ? scale2<0>(sw) : (step % 4 == 1) ? scale2<1>(sw)
                          : (step % 4 == 2) ? scale2<2>(sw) : scale2<3>(sw);
              b0[j] = mul2(v[j][2 * half], s2);
              b1[j] = mul2(v[j][2 * half + 1], s2);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              if (mt < tiles) {
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"(a_stage + lds_off[step] + mt * 4096));
#pragma unroll
                for (int j = 0; j < NGW; ++j) mma16816(acc[half % NACC][mt][j], a, b0[j], b1[j]);
              }
            }
          }
        }
      }
    }
    if (NACC == 2) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NGW; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[0][mt][j][i] += acc[NACC - 1][mt][j][i];
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
          float g0 = acc[0][mt][0][2 * h] * gsc, g1 = acc[0][mt][0][2 * h + 1] * gsc;
          float u0 = acc[0][mt][NGW - 1][2 * h] * gsc, u1 = acc[0][mt][NGW - 1][2 * h + 1] * gsc;
          v0 = g0 / (1.f + __expf(-g0)) * u0;
          v1 = g1 / (1.f + __expf(-g1)) * u1;
          n = 8 * (nb * 8 + warp) + c;
        } else {
          v0 = acc[0][mt][0][2 * h] * gsc;
          v1 = acc[0][mt][0][2 * h + 1] * gsc;
          n = 8 * (nb * 8 + warp) + c;
        }
        __nv_bfloat162 pr = __floats2bfloat162_rn(v0, v1);
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(row0 + orow) * KTOT + n) = pr;
      }
    }
  }
}


// ---------------------------------------------------------------------------------------
// v5: warp-specialized. Warp 8 is the producer: per 128-k chunk it issues bulk copies
// (cp.async.bulk, completion counted in bytes on the stage's "full" mbarrier) of the CTA's
// contiguous weight block, its scale block and the item's activation rows (272-byte padded
// rows: conflict-free ldmatrix without a swizzle). Warps 0-7 consume: wait on "full", compute,
// arrive on "empty". No CTA-wide barrier, so warps drift up to STAGES - 1 chunks apart, and
// the pipeline runs on across items (the producer prefetches the next item's chunks).
// Weight layout: [expert][n block][chunk][slot = warp * NGW + j][lane][16 B]; scales
// [expert][n block][chunk][slot][row 8][step 8].

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
// A consumer releasing a pipeline stage. OKC_ARRIVE1: one arrive per warp (lane 0, as a
// predicated instruction: no divergent branch) on a barrier expecting 8; otherwise every lane
// arrives (256 shared-memory atomics on one word per stage: ~1200 clk per stage in the
// bandwidth probe's hold test).
#ifdef OKC_ARRIVE1
#define STAGE_ARRIVALS 8
#else
#define STAGE_ARRIVALS 256
#endif
__device__ __forceinline__ void stage_release(uint32_t bar, int lane) {
#ifdef OKC_ARRIVE1
  __syncwarp();
  asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %1, 0;\n @p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n"
               :: "r"(bar), "r"(lane) : "memory");
#else
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
#endif
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
// All lanes of a warp wait on the same barrier phase, so the loop exits warp-uniformly; the
// vote tells the compiler so (no per-ldmatrix WARPSYNC).
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!__all_sync(0xffffffffu, mbar_try(bar, parity))) {
  }
}
// L2 prefetch of a weight block a few chunks ahead: HBM requests in flight no longer limited
// by the shared-memory stages (the stage's own bulk copy then hits L2).
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Two steps' scales at once: bytes 0, 2 (or 1, 3) of a scale word -> bf16x2 (s_a, s_b) * 2^118.
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

template <int N>
struct IntC {
  static constexpr int value = N;
};

template <int MODE, int MT, int NGW>
struct WsCfg {
  static constexpr int A_ROW = 272;
  static constexpr int A_STAGE = MT * 16 * A_ROW;
  static constexpr int W_STAGE = 8 * NGW * 512;
  static constexpr int S_STAGE = 8 * NGW * 64;
  static constexpr int STAGE = A_STAGE + W_STAGE + S_STAGE;
};

// KS > 1 (small batches, too few items to fill the SMs): KS groups of 8 consumer warps per CTA
// split K, group g taking stages g, g + KS, ...; at the end of an item the partial
// accumulators meet in shared memory and group 0 runs the epilogue.
// PROF (profiling build only; production code is the PROF = false instantiation): per item,
// prof[item * PSTR + ...] = {smid, start ns, end ns (global timer), consumer warp 0's cycles
// waiting on "full" (not measured: it perturbs the hot loop; read the ablations instead),
// producer's cycles waiting on "empty", item cycles (warp 0), producer start ns, 0}.
#define PSTR 12   // profiling record: int64 fields per item
#define PCTA 2048   // v6 profiling: int64 per-CTA entry/exit slots before prof (grid <= 1024)
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int smid() {
  int r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

template <int MODE, int MT, int NGW, int STAGES, int KS, bool PROF, int ABL>
__global__ void __launch_bounds__(32 * (8 * KS + 1), KS > 1 ? 1 : MT == 1 ? 4 : 3) moe_gemm_ws_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks,
    long long* __restrict__ prof, int* __restrict__ sched) {
  // abl (profiling build only): bit 0 consumers skip the math; bit 1 producer skips the
  // activation rows; bit 2 producer copies nothing (consumers compute on stale smem);
  // bit 3 consumers skip the HMMAs (decode only); bit 4 consumers skip the decode (HMMAs on
  // the raw words). Compile-time: each ablation is its own instantiation (the profiling
  // module only), so production code and the measured code paths stay branch-free.
  constexpr int abl = PROF ? ABL : 0;
  using C = WsCfg<MODE, MT, NGW>;
  constexpr int NCHUNK = KTOT / 128;
  static_assert(NCHUNK % STAGES == 0 && STAGES % KS == 0, "chunks / stages / K groups must divide");
  static_assert(8 * KS + 1 <= 32, "at most 1024 threads per CTA");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + STAGES * C::STAGE, bar_empty = bar_full + 8 * STAGES;
  // Per-stage header: the item id of the chunk in that stage (-1: no more items).
  volatile int* hdr = reinterpret_cast<volatile int*>(smem + STAGES * (C::STAGE + 16));
  float* red = reinterpret_cast<float*>(smem + STAGES * (C::STAGE + 32));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, 256);   // every consumer lane arrives (no divergent branch)
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int n_active = *n_active_ptr;
  const int total = n_active * n_blocks;

  if (warp == 8 * KS) {
    // Producer. Fill f uses stage f % STAGES; before reusing a stage it waits for the
    // consumers' release of the previous fill there (empty-barrier phase (f / STAGES - 1) & 1).
    // Items beyond the first come from a global counter (no CTA idles while work is left);
    // the item id travels to the consumers in the stage header. After its last fetch, each
    // CTA counts itself done; the last one resets the counter for the next launch.
    // The first item is static (blockIdx.x: consecutive CTAs land on different SMs, so a
    // small batch spreads over the GPU); later ones come from the counter.
    uint32_t f = 0;
    for (bool first = true;; first = false) {
      int item = blockIdx.x;
      if (!first) {
        if (lane == 0) item = (int)gridDim.x + atomicAdd(sched, 1);
        item = __shfl_sync(0xffffffffu, item, 0);
      }
      if (item >= total) {
        for (int g = 0; g < KS; ++g, ++f) {      // one end marker per K group
          const uint32_t st = f % STAGES;
          if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
          if (lane == 0) {
            hdr[st] = -1;
            mbar_arrive(bar_full + 8 * st);
          }
        }
        if (lane == 0 && atomicAdd(sched + 1, 1) == (int)gridDim.x - 1) {
          atomicExch(sched, 0);
          atomicExch(sched + 1, 0);
        }
        return;
      }
      const int nb = item % n_blocks;
      const int ch = active[item / n_blocks];
      const int e = ch & 255, mc = ch >> 8;
      const int row0 = exp_start[e] + mc * MT * 16;
      const int live = min(MT * 16, exp_start[e + 1] - row0);
      const uint8_t* a_row[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int lr = lane + 32 * h;
        int src = 0;
        if (lr < live) src = MODE == 1 ? row_route[row0 + lr] / 8 : row0 + lr;
        a_row[h] = reinterpret_cast<const uint8_t*>(a_src + (size_t)src * KTOT);
      }
      const uint8_t* w_blk = wq + ((size_t)e * n_blocks + nb) * NCHUNK * C::W_STAGE;
      const uint8_t* s_blk = ws + ((size_t)e * n_blocks + nb) * NCHUNK * C::S_STAGE;
      const uint32_t tx = (abl & 4) ? 0u : ((abl & 2) ? 0u : live * 256) + C::W_STAGE + C::S_STAGE;
      long long pw = 0;
      if (PROF && lane == 0) prof[item * PSTR + 6] = gtimer();
      for (int kc = 0; kc < NCHUNK; ++kc, ++f) {
        const uint32_t st = f % STAGES;
        if (PROF) {
          long long c0 = clock64();
          if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
          pw += clock64() - c0;
        } else if (f >= STAGES) {
          mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        }
        const uint32_t sb = base + st * C::STAGE, full = bar_full + 8 * st;
        if (lane == 0) {
          hdr[st] = item;
          mbar_expect_tx(full, tx);
          if (!(abl & 4)) {
            bulk_g2s(sb + C::A_STAGE, w_blk + (size_t)kc * C::W_STAGE, C::W_STAGE, full);
            bulk_g2s(sb + C::A_STAGE + C::W_STAGE, s_blk + (size_t)kc * C::S_STAGE, C::S_STAGE, full);
          }
        }
        __syncwarp();
        if (!(abl & 6)) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int lr = lane + 32 * h;
            if (lr < live) bulk_g2s(sb + lr * C::A_ROW, a_row[h] + kc * 256, 256, full);
          }
        }
      }
      if (PROF && lane == 0) prof[item * PSTR + 4] = pw;
    }
  }

  // Consumers: K group kg, slot warp ws.
  const int kg = warp / 8, wsl = warp % 8;
  const int r = lane / 4, c = (lane % 4) * 2;
  uint32_t lds_off[8];
  {
    const int row = lane % 16, hi = lane / 16;
#pragma unroll
    for (int step = 0; step < 8; ++step) lds_off[step] = base + kg * C::STAGE + row * C::A_ROW + (step * 2 + hi) * 16;
  }
  const uint32_t w_src = base + kg * C::STAGE + C::A_STAGE + ((wsl * NGW) * 32 + lane) * 16;
  const uint32_t s_src = base + kg * C::STAGE + C::A_STAGE + C::W_STAGE + (wsl * NGW) * 64 + r * 8;
  const uint32_t my_full = bar_full + 8 * kg, my_empty = bar_empty + 8 * kg;
  uint32_t f = 0;
  for (;;) {
    // The group's first stage of the round carries the item id (the chunk loop's own wait on
    // it then returns at once).
    mbar_wait(my_full, (f / STAGES) & 1);
    const int item = hdr[kg];
    if (item < 0) break;
    const int nb = item % n_blocks;
    const int ch = active[item / n_blocks];
    const int e = ch & 255, mc = ch >> 8;
    const int row0 = exp_start[e] + mc * MT * 16;
    const int rows_e = exp_start[e + 1] - row0;
    const int tiles = min(MT, (rows_e + 15) / 16);
    const float gsc = gs[e] * 256.0f;
    long long cw = 0, ct0 = 0;
    if (PROF && warp == 0) {
      ct0 = clock64();
      if (lane == 0) prof[item * PSTR + 1] = gtimer();
    }
    constexpr int NACC = MT == 1 ? 2 : 1;
    float acc[NACC][MT][NGW][4];
#pragma unroll
    for (int x = 0; x < NACC; ++x)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NGW; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[x][mt][j][i] = 0.f;

    // The chunk loop, specialized on the tiles this item has (T), so no per-ldmatrix guard:
    // whether any lane's item has 2+ tiles is decided once, by a warp vote (uniform).
    auto chunks = [&](auto tt) {
      constexpr int T = decltype(tt)::value;
  #pragma unroll 1
      for (int kc0 = 0; kc0 < NCHUNK; kc0 += STAGES, f += STAGES) {
        const uint32_t parity = (f / STAGES) & 1;
  #pragma unroll
        for (int jst = 0; jst < STAGES / KS; ++jst) {
          const int st = jst * KS;   // + kg, folded into the base addresses
          mbar_wait(my_full + 8 * st, parity);
          if (PROF && (abl & 1)) {
            mbar_arrive(my_empty + 8 * st);
            continue;
          }
          uint4 wv[NGW];
          uint2 sv[NGW];
  #pragma unroll
          for (int j = 0; j < NGW; ++j) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(wv[j].x), "=r"(wv[j].y), "=r"(wv[j].z), "=r"(wv[j].w)
                         : "r"(w_src + st * C::STAGE + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(sv[j].x), "=r"(sv[j].y)
                         : "r"(s_src + st * C::STAGE + j * 64));
          }
          const uint32_t a_stage = st * C::STAGE;   // + lds_off (which holds the smem base)
  #pragma unroll
          for (int w = 0; w < 4; ++w) {
            uint32_t v[NGW][4], s_pair[NGW];
  #pragma unroll
            for (int j = 0; j < NGW; ++j) {
              uint32_t q = w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w;
              unpack8(q, v[j]);
              // Steps 2w, 2w + 1: word (w < 2 ? x : y); even steps use bytes 0/2, odd 1/3.
              uint32_t sw = w < 2 ? sv[j].x : sv[j].y;
              s_pair[j] = sw;
            }
  #pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int step = 2 * w + half;
              uint32_t b0[NGW], b1[NGW];
  #pragma unroll
              for (int j = 0; j < NGW; ++j) {
                // step % 4: 0 -> even.lo, 1 -> odd.lo, 2 -> even.hi, 3 -> odd.hi
                const uint32_t sp = half == 0 ? scale_even(s_pair[j]) : scale_odd(s_pair[j]);
                if (PROF && (abl & 16)) {
                  b0[j] = v[j][0] ^ wv[j].x;   // keep a data dependency, skip unpack/scale/HMUL2
                  b1[j] = wv[j].y;
                } else if ((step % 4) < 2) {
                  b0[j] = mul2h<0>(v[j][2 * half], sp);
                  b1[j] = mul2h<0>(v[j][2 * half + 1], sp);
                } else {
                  b0[j] = mul2h<1>(v[j][2 * half], sp);
                  b1[j] = mul2h<1>(v[j][2 * half + 1], sp);
                }
              }
  #pragma unroll
              for (int mt = 0; mt < MT; ++mt) {
                if (mt < T) {
                  uint32_t a[4];
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                               : "r"(lds_off[step] + a_stage + mt * 16 * C::A_ROW));
  #pragma unroll
                  for (int j = 0; j < NGW; ++j) {
                    // One tile: even / odd k16 steps into separate accumulators (MT 1: acc[1];
                    // MT > 1: the idle tile 1 slot), halving the HMMA dependency chain.
                    float* c = MT == 1 ? acc[half % NACC][0][j] : T == 1 ? acc[0][half][j] : acc[0][mt][j];
                    if (PROF && (abl & 8)) {
                      c[0] += __int_as_float((b0[j] ^ b1[j] ^ a[0]) & 0x3f800000u);   // consume, no HMMA
                    } else {
                      mma16816(c, a, b0[j], b1[j]);
                    }
                  }
                }
              }
            }
          }
          mbar_arrive(my_empty + 8 * st);
        }
      }
    };
    if (MT > 1 && __any_sync(0xffffffffu, tiles > 1)) {
      chunks(IntC<MT>{});
    } else {
      chunks(IntC<1>{});
      if (MT > 1) {
#pragma unroll
        for (int j = 0; j < NGW; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[0][0][j][i] += acc[0][MT > 1 ? 1 : 0][j][i];
      }
    }
    if (NACC == 2) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NGW; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[0][mt][j][i] += acc[NACC - 1][mt][j][i];
    }
    if (KS > 1) {
      // red: [K group - 1][slot warp][accumulator index][lane] (lane-contiguous: conflict-free).
      constexpr int NV = MT * NGW * 4;
      if (kg > 0) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int j = 0; j < NGW; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i)
              red[(((kg - 1) * 8 + wsl) * NV + (mt * NGW + j) * 4 + i) * 32 + lane] = acc[0][mt][j][i];
      }
      asm volatile("bar.sync 1, %0;\n" :: "n"(256 * KS) : "memory");
      if (kg == 0) {
#pragma unroll
        for (int g = 1; g < KS; ++g)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < NGW; ++j)
#pragma unroll
              for (int i = 0; i < 4; ++i)
                acc[0][mt][j][i] += red[(((g - 1) * 8 + wsl) * NV + (mt * NGW + j) * 4 + i) * 32 + lane];
      }
      asm volatile("bar.sync 1, %0;\n" :: "n"(256 * KS) : "memory");
      if (kg > 0) continue;
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= tiles) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int orow = mt * 16 + r + 8 * h;
        if (orow >= rows_e) continue;
        float v0, v1;
        if (MODE == 1) {
          float g0 = acc[0][mt][0][2 * h] * gsc, g1 = acc[0][mt][0][2 * h + 1] * gsc;
          float u0 = acc[0][mt][NGW - 1][2 * h] * gsc, u1 = acc[0][mt][NGW - 1][2 * h + 1] * gsc;
          v0 = g0 / (1.f + __expf(-g0)) * u0;
          v1 = g1 / (1.f + __expf(-g1)) * u1;
        } else {
          v0 = acc[0][mt][0][2 * h] * gsc;
          v1 = acc[0][mt][0][2 * h + 1] * gsc;
        }
        const int n = 8 * (nb * 8 + wsl) + c;
        __nv_bfloat162 pr = __floats2bfloat162_rn(v0, v1);
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(row0 + orow) * KTOT + n) = pr;
      }
    }
    if (PROF && warp == 0 && lane == 0) {
      prof[item * PSTR + 0] = smid();
      prof[item * PSTR + 2] = gtimer();
      prof[item * PSTR + 3] = cw;
      prof[item * PSTR + 5] = clock64() - ct0;
    }
  }
}

// ---------------------------------------------------------------------------------------
// v6: swap-AB. The roofline has a shared-memory term (128 B/clk/SM): in v5 every warp
// ldmatrix-read the same 16-row activation tile, 4 B of smem per weight, which capped GEMM2
// at ~3.2 TB/s. Here the decoded weights are the MMA's A operand (16 n-rows x 16 k, straight
// from registers: a warp's two n8 groups are rows r and r + 8), and the tokens are B (8 per
// group, one ldmatrix.x4 per two k16 steps): 1 B of activation smem per weight, half the HMMAs.
// GEMM1's group pair is (gate, up), so C rows r / r + 8 are gate / up of the same column n.
// Producer, pipeline and scheduling as v5; 16 weight groups (slots) per CTA in both GEMMs:
// weights [expert][n block][chunk][slot 16][lane][16 B], scales [...][slot 16][row 8][step 8].

// SUB = 128-k chunks per pipeline stage: 2 makes each weight copy 16 KB (one cp.async.bulk),
// which the memory system serves better when the consumers pace it (bandwidth probe
// 692595f8: 3.58 vs 3.11 TB/s at v6's bs 8 pace; two back-to-back 8 KB copies: no gain).
// Atomic post (pa.mode): GEMM1 (mode & 1) zeroes y32 [P, 4096], out32 [M, 4096] and the bf16 output at
// its start; GEMM2 (mode & 2) adds its (split-K) partials into y32 with red.add (no partial slots, no
// reduction pass: not bitwise repeatable, well inside the 1.5% budget), and the last-arriving item
// of an expert chunk RMS-norms its rows and adds topk_w * norm into out32; each token's last route
// converts out32 to the bf16 output. Replaces _post_norm_*_kernel (one launch and its latency).
struct PostArgs {
  float* y32;
  float* out32;
  __nv_bfloat16* outb;
  const __nv_bfloat16* norm_w;
  const float* topk_w;
  const int32_t* pos;
  int* tok_cnt;
  int* chunk_cnt;
  float eps;
  int M;
  int P;
  int mode;
};

// RB = 128-row blocks per item: 2 gives each consumer warp 32 output rows (two m16 tiles) that
// share every activation fragment, halving the ldmatrix traffic per weight byte (the loop's
// shared-memory bandwidth bound, graph ablations 2db7fe47). A stage holds the RB blocks' weights
// [block 0][block 1], then their scales; the global layout is unchanged (RB adjacent blocks).
template <int MT, int SUB = 1, int RB = 1>
struct SabCfg {
  static constexpr int A_ROW = 256 * SUB + 16;   // + 16 B: conflict-free ldmatrix rows
  static constexpr int A_STAGE = MT * 8 * A_ROW;
  static constexpr int W_STAGE = SUB * 16 * 512;
  static constexpr int S_STAGE = SUB * 16 * 64;
  static constexpr int STAGE = A_STAGE + RB * (W_STAGE + S_STAGE);
};

template <int MODE, int MT, int STAGES, bool PROF, int ABL = 0, int SUB = 1, int RB = 1>
__global__ void __launch_bounds__(288, RB > 1 ? 2 : (MT == 1 ? 4 : 3)) moe_gemm_sab_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks,
    long long* __restrict__ prof, int* __restrict__ sched, int kparts, float* __restrict__ part, int part_rows,
    int flags, const uint8_t* __restrict__ pf_w, const uint8_t* __restrict__ pf_s, int* __restrict__ counters,
    PostArgs pa) {
  // flags: 1 = stream K (below); 2 = launched as a programmatic dependent (PDL) of the previous
  // kernel, whose output (our activations) may not be ready: the producer warms L2 with its
  // first item's first stages of weights, then waits for that grid (griddepcontrol.wait).
  // Every producer signals its dependents (griddepcontrol.launch_dependents) once the work
  // queue is empty, so the next GEMM's CTAs can take SM slots as this one's CTAs retire.
  // flags & 4 (GEMM1 at tiny batches): also warm L2 with the NEXT GEMM's weights (pf_w / pf_s:
  // GEMM2's repacked blocks, contiguous per expert) for the experts with routes, if they total
  // <= 32 MB: GEMM1 here is latency-bound (~2.5 of 4.5 TB/s), so the spare HBM bandwidth
  // streams GEMM2's weights early and GEMM2 then reads L2.
  // flags & 8 (GEMM1, split K): the CTA that completes an item's last K part (counters[item],
  // self-resetting) sums the parts in slot order and writes silu(gate) * up itself: no
  // separate reduction kernel, and GEMM2 can depend on this kernel directly.
  // flags & 16: a PDL co-runner of the previous kernel (another GEMM's launch over other
  // experts): no input dependency, so no wait at the start; every thread waits for that grid
  // before exiting, so this kernel completes after it (the next launch then sees both).
  // flags & 32: signal dependents at the start (the dualsab big launch: its co-runner reads none
  // of its output, and with fewer items than CTAs the queue-empty signal would come only after
  // the hot experts' weights had streamed, serializing the two launches).
  // flags & 128 (chain): every launch of the pipeline is a programmatic dependent of the one
  // before (route -> GEMM1 -> [split-K reduction] -> GEMM2 -> post-norm) and signals its own
  // dependents at its start, so launch, CTA setup and (GEMM2) the first stages' weight copies
  // overlap the previous kernel. Without flags & 2 (GEMM1 after the route) every thread waits
  // for the previous grid before the first read of anything (the route's output), then signals:
  // the next kernels' pre-wait code reads only the route's output and weights, complete by then.
  // With flags & 2 (GEMM2) the producer's wait guards the activations; the signal comes first
  // (the post-norm reads GEMM2's output only after its own wait).
  const int sk = flags & 1;
  const int l2d = (flags >> 8) & 15;   // L2 prefetch distance in stages (0: off)
  // Split K (kparts > 1, small batches: too few items for the SMs): item = (chunk entry,
  // n block, k part); each part streams NCHUNK / kparts chunks and writes its scaled fp32
  // partial to part[k part][row][column] (GEMM1: gate at n, up at 4096 + n); the parts are
  // summed in a fixed order afterwards (_splitk_silu_kernel, _post_norm_parts_kernel), so the
  // result stays bitwise repeatable.
  constexpr int abl = PROF ? ABL : 0;   // ablation bits as moe_gemm_ws_kernel's
  using C = SabCfg<MT, SUB, RB>;
  // RB > 1: n_blocks counts items' row-block groups; the weights' block index is nb * RB + h.
  // (Host: no split K, stream K, L2 warming or metadata prefetch with RB > 1.)
  constexpr int NCHUNK = KTOT / (128 * SUB);   // pipeline stages per item
  static_assert(NCHUNK % STAGES == 0 && MT * 8 <= 32, "stages divide chunks; <= 32 token rows");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + STAGES * C::STAGE, bar_empty = bar_full + 8 * STAGES;
  volatile int* hdr = reinterpret_cast<volatile int*>(smem + STAGES * (C::STAGE + 16));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // Profiling build: per-CTA entry / consumer-exit times, PCTA records before the item records.
  if (PROF && tid == 0) prof[-PCTA + 2 * (int)blockIdx.x] = gtimer();
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, STAGE_ARRIVALS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (flags & 128) {
    if (!(flags & 2)) asm volatile("griddepcontrol.wait;\n" ::: "memory");
    if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  }
  if (pa.mode & 1) {   // atomic post, GEMM1: zero the accumulators and the output for this call
    const size_t gt = (size_t)blockIdx.x * blockDim.x + tid, gn = (size_t)gridDim.x * blockDim.x;
    for (size_t i = gt; i < (size_t)pa.P * KTOT / 4; i += gn) reinterpret_cast<float4*>(pa.y32)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = gt; i < (size_t)pa.M * KTOT / 4; i += gn) reinterpret_cast<float4*>(pa.out32)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = gt; i < (size_t)pa.M * KTOT / 8; i += gn) reinterpret_cast<uint4*>(pa.outb)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  const int total = *n_active_ptr * n_blocks * kparts;
  const int nch = NCHUNK / kparts;   // chunks per item
  // Stream K (sk, kparts = partial slots >= 2): the items' K ranges, in units of STAGES chunks,
  // laid end to end and cut into geff equal CTA ranges [s0, s1), so every CTA (and, at a full
  // grid of 4 CTAs on every SM, every SM) streams the same bytes. A CTA range is >= NU /
  // (kparts - 1) units, so an item spans <= kparts CTAs; its segment from CTA v goes to
  // partial slot v - (first CTA of the item); that CTA zero-fills the unused slots.
  constexpr int NU = NCHUNK / STAGES;
  // (int: U <= (32 + 256) chunk entries x 64 n blocks x 8 units, U x geff < 2^31)
  int U = 1, s0 = 0, s1 = 0, geff = 1;
  if (sk) {
    U = *n_active_ptr * n_blocks * NU;
    const int minlen = (NU + kparts - 2) / (kparts - 1);
    geff = min((int)gridDim.x, max(1, U / minlen));
    if ((int)blockIdx.x < geff) {
      s0 = (int)blockIdx.x * U / geff;
      s1 = ((int)blockIdx.x + 1) * U / geff;
    }
  }

  if (warp == 8) {
    if ((flags & 32) && lane == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    int pre = 0;   // chain: stages of the first item whose weights + scales were issued before the wait
    if (flags & 2) {
      if ((flags & 128) && abl == 0 && !sk && (int)blockIdx.x < total) {
        // The first item's metadata is the route's output (complete before this grid launched);
        // only its activation rows (the previous GEMM's output) wait for griddepcontrol.wait.
        const int it0 = (int)blockIdx.x / kparts, kb0 = ((int)blockIdx.x % kparts) * nch;
        const int ch0 = active[it0 / n_blocks];
        const int e0 = ch0 & 255, nb0 = it0 % n_blocks;
        const int r00 = exp_start[e0] + (ch0 >> 8) * MT * 8;
        const int live0 = min(MT * 8, exp_start[e0 + 1] - r00);
        const uint32_t tx0 = live0 * 256 * SUB + RB * (C::W_STAGE + C::S_STAGE);
        pre = min(STAGES, nch);
        if (lane == 0) {
          for (int st = 0; st < pre; ++st) {
            const uint32_t sb = base + st * C::STAGE, full = bar_full + 8 * st;
            hdr[st] = (int)blockIdx.x;
            mbar_expect_tx(full, tx0);
#pragma unroll
            for (int h = 0; h < RB; ++h) {
              const size_t blk = ((size_t)e0 * n_blocks + nb0) * RB + h;
              bulk_g2s(sb + C::A_STAGE + h * C::W_STAGE, wq + (blk * NCHUNK + kb0 + st) * C::W_STAGE, C::W_STAGE, full);
              bulk_g2s(sb + C::A_STAGE + RB * C::W_STAGE + h * C::S_STAGE, ws + (blk * NCHUNK + kb0 + st) * C::S_STAGE,
                       C::S_STAGE, full);
            }
          }
        }
        __syncwarp();
      } else if (lane == 0 && !sk && (int)blockIdx.x < total) {
        const int it0 = (int)blockIdx.x / kparts, kb0 = ((int)blockIdx.x % kparts) * nch;
        const int e0 = active[it0 / n_blocks] & 255, nb0 = it0 % n_blocks;
        const uint8_t* w0 = wq + ((((size_t)e0 * n_blocks + nb0) * RB) * NCHUNK + kb0) * C::W_STAGE;
        const uint8_t* s0p = ws + ((((size_t)e0 * n_blocks + nb0) * RB) * NCHUNK + kb0) * C::S_STAGE;
        prefetch_l2(w0, (uint32_t)min(STAGES, nch) * C::W_STAGE);
        prefetch_l2(s0p, (uint32_t)min(STAGES, nch) * C::S_STAGE);
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }
    if ((flags & 4) && pf_w != nullptr) {
      constexpr uint32_t PW = 32u * 32u * 8192u, PS = 32u * 32u * 1024u, PIECE = 64u * 1024u;   // per expert
      const unsigned act = __ballot_sync(0xffffffffu, exp_start[lane + 1] > exp_start[lane]);
      const int nexp = __popc(act);
      if ((size_t)nexp * (PW + PS) <= (32u << 20)) {
        const int per = (PW + PS) / PIECE, pieces = nexp * per;
        for (int pc = (int)blockIdx.x * 32 + lane; pc < pieces; pc += (int)gridDim.x * 32) {
          const int j = pc / per, o = pc % per;
          const int e = __fns(act, 0, j + 1);   // the j-th expert with routes
          if (o < (int)(PW / PIECE)) prefetch_l2(pf_w + (size_t)e * PW + (size_t)o * PIECE, PIECE);
          else prefetch_l2(pf_s + (size_t)e * PS + (size_t)(o - PW / PIECE) * PIECE, PIECE);
        }
      }
    }
    // Producer (as v5): first item static, then the global counter; one end marker.
    // flags & 64 (not stream K): the next item's metadata is fetched one dependent step per
    // stage over the current item's last stages (claim -> active -> exp_start -> row_route),
    // so each load has a stage time to land before use instead of four round trips at the
    // item boundary with the ring draining. The claim stays late (PF_LEAD stages before the
    // end), so dynamic balancing is unchanged.
    constexpr int PF_LEAD = 5;
    const bool pf = (flags & 64) && !sk;
    int nx_raw = 0, nx_item = 0, nx_ch = 0, nx_es0 = 0, nx_es1 = 0, nx_src = 0, nx_step = 0;
    auto pf_step = [&](int step) {
      if (step == 0) {
        if (lane == 0) nx_raw = atomicAdd(sched, 1);
      } else if (step == 1) {
        nx_item = (int)gridDim.x + __shfl_sync(0xffffffffu, nx_raw, 0);
        if (nx_item < total) nx_ch = active[(nx_item / kparts) / n_blocks];
      } else if (step == 2) {
        if (nx_item < total) {
          nx_es0 = exp_start[nx_ch & 255];
          nx_es1 = exp_start[(nx_ch & 255) + 1];
        }
      } else if (nx_item < total) {
        const int r0 = nx_es0 + (nx_ch >> 8) * MT * 8;
        nx_src = 0;
        if (lane < min(MT * 8, nx_es1 - r0)) nx_src = MODE == 1 ? row_route[r0 + lane] / 8 : r0 + lane;
      }
    };
    uint32_t f = 0;
    int u = s0;
    for (bool first = true;; first = false) {
      int item = blockIdx.x, kb = 0, ke = nch;
      bool done;
      bool have_nx = false;
      if (pf && !first) {
        for (; nx_step < 4; ++nx_step) pf_step(nx_step);   // finish what the item did not cover
        nx_step = 0;
        item = nx_item;
        have_nx = true;
        done = item >= total;
        kb = (item % kparts) * nch;
        ke = kb + nch;
      } else if (sk) {
        done = u >= s1;
        if (!done) {
          item = u / NU;
          kb = (u - item * NU) * STAGES;
          ke = min(s1 - item * NU, NU) * STAGES;
          u = item * NU + ke / STAGES;
        }
      } else {
        if (!first) {
          if (lane == 0) item = (int)gridDim.x + atomicAdd(sched, 1);
          item = __shfl_sync(0xffffffffu, item, 0);
        }
        done = item >= total;
        kb = (item % kparts) * nch;
        ke = kb + nch;
      }
      if (done) {
        if (lane == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
        const uint32_t st = f % STAGES;
        if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        if (lane == 0) {
          hdr[st] = -1;
          mbar_arrive(bar_full + 8 * st);
          if (atomicAdd(sched + 1, 1) == (int)gridDim.x - 1) {
            atomicExch(sched, 0);
            atomicExch(sched + 1, 0);
          }
        }
        if (flags & 16) asm volatile("griddepcontrol.wait;\n" ::: "memory");
        return;
      }
      const int it = sk ? item : item / kparts;
      const int nb = it % n_blocks;
      const int ch = have_nx ? nx_ch : active[it / n_blocks];
      const int e = ch & 255, mc = ch >> 8;
      const int es0 = have_nx ? nx_es0 : exp_start[e], es1 = have_nx ? nx_es1 : exp_start[e + 1];
      const int row0 = es0 + mc * MT * 8;
      const int live = min(MT * 8, es1 - row0);
      int src = 0;
      if (have_nx) src = nx_src;
      else if (lane < live) src = MODE == 1 ? row_route[row0 + lane] / 8 : row0 + lane;
      const uint8_t* a_row = reinterpret_cast<const uint8_t*>(a_src + (size_t)src * KTOT);
      const uint8_t* w_blk = wq + ((size_t)e * n_blocks + nb) * RB * NCHUNK * C::W_STAGE;   // block nb * RB
      const uint8_t* s_blk = ws + ((size_t)e * n_blocks + nb) * RB * NCHUNK * C::S_STAGE;
      const uint32_t tx = (abl & 4) ? 0u : ((abl & 2) ? 0u : live * 256 * SUB) + RB * (C::W_STAGE + C::S_STAGE);
      // profiling record: one per item (stream K: per item segment)
      const int prec = sk ? item * kparts + (int)blockIdx.x - ((it * NU + 1) * geff - 1) / U : item;
      long long pw = 0;
      if (PROF && lane == 0) prof[prec * PSTR + 6] = gtimer();
      for (int kc = kb; kc < ke; ++kc, ++f) {
        const uint32_t st = f % STAGES;
        long long c0 = PROF ? clock64() : 0;
        if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        if (PROF) pw += clock64() - c0;
        const uint32_t sb = base + st * C::STAGE, full = bar_full + 8 * st;
        if (lane == 0 && f >= (uint32_t)pre) {   // (chain: the first item's first stages went out pre-wait)
          hdr[st] = item;
          mbar_expect_tx(full, tx);
          if (!(abl & 4)) {
#pragma unroll
            for (int h = 0; h < RB; ++h) {   // block nb * RB + h: NCHUNK chunks further on
              bulk_g2s(sb + C::A_STAGE + h * C::W_STAGE, w_blk + ((size_t)h * NCHUNK + kc) * C::W_STAGE, C::W_STAGE, full);
              bulk_g2s(sb + C::A_STAGE + RB * C::W_STAGE + h * C::S_STAGE,
                       s_blk + ((size_t)h * NCHUNK + kc) * C::S_STAGE, C::S_STAGE, full);
              // flags bits 8-11 (L2 distance D, in stages): chunk kc + D goes to L2 now, so the
              // HBM stream runs ahead of the smem ring when the consumers hold its stages.
              if (l2d && kc + l2d < ke) {
                prefetch_l2(w_blk + ((size_t)h * NCHUNK + kc + l2d) * C::W_STAGE, C::W_STAGE);
                prefetch_l2(s_blk + ((size_t)h * NCHUNK + kc + l2d) * C::S_STAGE, C::S_STAGE);
              }
            }
          }
        }
        __syncwarp();
        if (!(abl & 6) && lane < live) bulk_g2s(sb + lane * C::A_ROW, a_row + kc * 256 * SUB, 256 * SUB, full);
        if (pf && nx_step < 4 && kc >= ke - PF_LEAD) pf_step(nx_step++);
      }
      if (PROF && lane == 0) prof[prec * PSTR + 4] = pw;
    }
  }

  // Consumers: warp w owns slots 2w (A rows 0-7) and 2w + 1 (rows 8-15).
  const int r = lane / 4, c = (lane % 4) * 2;
  // ldmatrix.x4 for k16 steps 2w', 2w'+1: lanes 0-7 / 8-15 / 16-23 / 24-31 address matrices
  // (step 2w', k 0-7), (2w', k 8-15), (2w'+1, k 0-7), (2w'+1, k 8-15) of token rows 0-7.
  uint32_t ldb[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) ldb[w] = base + (lane % 8) * C::A_ROW + (4 * w + lane / 8) * 16;
  const uint32_t w_src = base + C::A_STAGE + ((warp * 2) * 32 + lane) * 16;
  const uint32_t s_src = base + C::A_STAGE + RB * C::W_STAGE + (warp * 2) * 64 + r * 8;
  uint32_t f = 0;
  for (;;) {
    mbar_wait(bar_full, (f / STAGES) & 1);
    const int item = hdr[0];
    if (item < 0) break;
    int kp = item % kparts, it = item / kparts, nchi = nch, nseg = 0;
    if (sk) {
      it = item;
      const int i0 = it * NU;
      nchi = (min(s1, i0 + NU) - max(s0, i0)) * STAGES;
      const int vfirst = ((i0 + 1) * geff - 1) / U;
      kp = (int)blockIdx.x - vfirst;
      nseg = ((i0 + NU) * geff - 1) / U - vfirst + 1;
    }
    const int prec = sk ? it * kparts + kp : item;   // profiling record
    const int nb = it % n_blocks;
    const int ch = active[it / n_blocks];
    const int e = ch & 255, mc = ch >> 8;
    const int row0 = exp_start[e] + mc * MT * 8;
    const int rows_e = exp_start[e + 1] - row0;
    const int groups = min(MT, (rows_e + 7) / 8);
    const float gsc = gs[e] * 256.0f;
    long long ct0 = 0;
    if (PROF && warp == 0) {
      ct0 = clock64();
      if (lane == 0) prof[prec * PSTR + 1] = gtimer();
    }
    // accs[h][x][t]: row block h, token group t, k16 steps of parity x (two chains per group).
    float accs[RB][2][MT][4];
#pragma unroll
    for (int h = 0; h < RB; ++h)
#pragma unroll
      for (int x = 0; x < 2; ++x)
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i) accs[h][x][t][i] = 0.f;
    auto& acc = accs[0];   // (split-K epilogue: RB == 1 only)

    auto chunks = [&](auto tt) {
      constexpr int T = decltype(tt)::value;
#pragma unroll 1
      for (int kc0 = 0; kc0 < nchi; kc0 += STAGES, f += STAGES) {
        const uint32_t parity = (f / STAGES) & 1;
#pragma unroll
        for (int st = 0; st < STAGES; ++st) {
          mbar_wait(bar_full + 8 * st, parity);
          if (abl & 1) {
            stage_release(bar_empty + 8 * st, lane);
            continue;
          }
#pragma unroll
          for (int sub = 0; sub < SUB; ++sub) {   // the stage's 128-k chunks
          if constexpr (RB > 1) {
            // RB row blocks: the chunk's activation fragments (all 8 k16 steps) are loaded once and
            // reused by every row block; each block's weights are loaded inside its own pass, so only
            // one block's words are live. One accumulator chain per block (the blocks give the ILP).
            uint32_t bb[4][T][4];
#pragma unroll
            for (int w = 0; w < 4; ++w)
#pragma unroll
              for (int t = 0; t < T; ++t)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(bb[w][t][0]), "=r"(bb[w][t][1]), "=r"(bb[w][t][2]), "=r"(bb[w][t][3])
                             : "r"(ldb[w] + st * C::STAGE + sub * 256 + t * 8 * C::A_ROW));
#pragma unroll
            for (int h = 0; h < RB; ++h) {
              uint4 wv[2];
              uint2 sv[2];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wv[j].x), "=r"(wv[j].y), "=r"(wv[j].z), "=r"(wv[j].w)
                             : "r"(w_src + st * C::STAGE + h * C::W_STAGE + sub * 8192 + j * 512));
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(sv[j].x), "=r"(sv[j].y)
                             : "r"(s_src + st * C::STAGE + h * C::S_STAGE + sub * 1024 + j * 64));
              }
#pragma unroll
              for (int w = 0; w < 4; ++w) {
                uint32_t v[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) unpack8(w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w, v[j]);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                  const int step = 2 * w + half;
                  uint32_t a[4];
#pragma unroll
                  for (int j = 0; j < 2; ++j) {
                    const uint32_t sw = w < 2 ? sv[j].x : sv[j].y;
                    const uint32_t sp = half == 0 ? scale_even(sw) : scale_odd(sw);
                    if ((step % 4) < 2) {
                      a[j] = mul2h<0>(v[j][2 * half], sp);
                      a[2 + j] = mul2h<0>(v[j][2 * half + 1], sp);
                    } else {
                      a[j] = mul2h<1>(v[j][2 * half], sp);
                      a[2 + j] = mul2h<1>(v[j][2 * half + 1], sp);
                    }
                  }
#pragma unroll
                  for (int t = 0; t < T; ++t) mma16816(accs[h][0][t], a, bb[w][t][2 * half], bb[w][t][2 * half + 1]);
                }
              }
            }
          } else {
          uint4 wvs[RB][2];
          uint2 svs[RB][2];
#pragma unroll
          for (int h = 0; h < RB; ++h)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(wvs[h][j].x), "=r"(wvs[h][j].y), "=r"(wvs[h][j].z), "=r"(wvs[h][j].w)
                         : "r"(w_src + st * C::STAGE + h * C::W_STAGE + sub * 8192 + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(svs[h][j].x), "=r"(svs[h][j].y)
                         : "r"(s_src + st * C::STAGE + h * C::S_STAGE + sub * 1024 + j * 64));
          }
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            uint32_t b[T][4];
#pragma unroll
            for (int t = 0; t < T; ++t)
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                           : "=r"(b[t][0]), "=r"(b[t][1]), "=r"(b[t][2]), "=r"(b[t][3])
                           : "r"(ldb[w] + st * C::STAGE + sub * 256 + t * 8 * C::A_ROW));
#pragma unroll
            for (int h = 0; h < RB; ++h) {   // row blocks: each reuses the activation fragments b
            const uint4* wv = wvs[h];
            const uint2* sv = svs[h];
            uint32_t v[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) unpack8(w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w, v[j]);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int step = 2 * w + half;
              uint32_t a[4];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const uint32_t sw = w < 2 ? sv[j].x : sv[j].y;
                const uint32_t sp = half == 0 ? scale_even(sw) : scale_odd(sw);
                if (abl & 16) {
                  a[j] = v[j][0] ^ wv[j].x;
                  a[2 + j] = wv[j].y;
                } else if ((step % 4) < 2) {
                  a[j] = mul2h<0>(v[j][2 * half], sp);
                  a[2 + j] = mul2h<0>(v[j][2 * half + 1], sp);
                } else {
                  a[j] = mul2h<1>(v[j][2 * half], sp);
                  a[2 + j] = mul2h<1>(v[j][2 * half + 1], sp);
                }
              }
#pragma unroll
              for (int t = 0; t < T; ++t) {
                if (abl & 8) {
                  accs[h][half][t][0] += __int_as_float((a[0] ^ a[3] ^ b[t][2 * half]) & 0x3f800000u);
                } else {
                  mma16816(accs[h][half][t], a, b[t][2 * half], b[t][2 * half + 1]);
                }
              }
            }
            }
          }
          }   // RB == 1
          }
          stage_release(bar_empty + 8 * st, lane);
        }
      }
    };
    if (MT > 1 && __any_sync(0xffffffffu, groups > 1)) {
      chunks(IntC<MT>{});
    } else {
      chunks(IntC<1>{});
    }

    // Epilogue. Lane holds C[n = r][tokens 2c, 2c+1] (c0, c1) and C[n = r + 8][...] (c2, c3)
    // per token group; one shuffle with lane ^ 4 (column r ^ 1) pairs adjacent columns so
    // each lane stores one bf16x2: even r -> token 2c at (r, r+1), odd r -> token 2c+1 at (r-1, r).
    if (MODE == 2 && (pa.mode & 2)) {
      // Atomic post: partials straight into y32, then the chunk's last item normalizes its rows.
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= groups) continue;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int tok = row0 + t * 8 + c + i;
          if (tok >= row0 + min(MT * 8, rows_e)) continue;
          float* yr = pa.y32 + (size_t)tok * KTOT;
#pragma unroll
          for (int h = 0; h < RB; ++h) {
            const int nbh = nb * RB + h;
            atomicAdd(yr + 8 * (nbh * 16 + 2 * warp) + r, (accs[h][0][t][i] + accs[h][1][t][i]) * gsc);
            atomicAdd(yr + 8 * (nbh * 16 + 2 * warp + 1) + r, (accs[h][0][t][2 + i] + accs[h][1][t][2 + i]) * gsc);
          }
        }
      }
      __threadfence();
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      if (warp == 0 && lane == 0) {
        const int ci = it / n_blocks;
        const int old = atomicAdd(pa.chunk_cnt + ci, 1);
        hdr[STAGES] = old == n_blocks * kparts - 1;
        if (old == n_blocks * kparts - 1) pa.chunk_cnt[ci] = 0;
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      if (hdr[STAGES]) {   // this chunk's rows are complete: RMSNorm, weight, add into the tokens
        __threadfence();
        const int nrows = min(MT * 8, rows_e);
        for (int rr = warp; rr < nrows; rr += 8) {
          const int row = row0 + rr;
          const int route = row_route[row], tk = route / 8;
          const float* yr = pa.y32 + (size_t)row * KTOT;
          float ss = 0.f;
          for (int i = lane; i < KTOT; i += 32) {
            const float v = __bfloat162float(__float2bfloat16_rn(__ldcg(yr + i))) * __bfloat162float(pa.norm_w[i]);
            ss += v * v;
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
          const float sc = rsqrtf(ss / KTOT + pa.eps) * pa.topk_w[route];
          float* orow = pa.out32 + (size_t)tk * KTOT;
          for (int i = lane; i < KTOT; i += 32) {
            const float v = __bfloat162float(__float2bfloat16_rn(__ldcg(yr + i))) * __bfloat162float(pa.norm_w[i]);
            atomicAdd(orow + i, v * sc);
          }
          __threadfence();
          int last = 0;
          if (lane == 0) {
            int need = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) need += pa.pos[tk * 8 + j] >= 0;
            const int old = atomicAdd(pa.tok_cnt + tk, 1);
            last = old == need - 1;
            if (last) pa.tok_cnt[tk] = 0;
          }
          last = __shfl_sync(0xffffffffu, last, 0);
          if (last) {   // every local route of this token is in: the bf16 output row
            __threadfence();
            for (int i = lane * 2; i < KTOT; i += 64)
              *reinterpret_cast<__nv_bfloat162*>(pa.outb + (size_t)tk * KTOT + i) =
                  __floats2bfloat162_rn(__ldcg(orow + i), __ldcg(orow + i + 1));
          }
        }
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");   // hdr[STAGES] is rewritten next item
    } else if (kparts > 1) {
      // Partial sums (scaled, fp32): C[n = r / r + 8][tokens 2c, 2c + 1] per token group.
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if (t >= groups) continue;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int tok = row0 + t * 8 + c + i;
          if (tok >= row0 + rows_e) continue;
          const float lo = (acc[0][t][i] + acc[1][t][i]) * gsc, hi = (acc[0][t][2 + i] + acc[1][t][2 + i]) * gsc;
          // Slot kp; stream K: the item's first segment also zeroes slots nseg .. kparts - 1.
          for (int sl = kp; sl < (sk && kp == 0 ? kparts : kp + 1); sl = sl == kp ? max(kp + 1, nseg) : sl + 1) {
            float* prow = part + ((size_t)sl * part_rows + tok) * (MODE == 1 ? 2 * KTOT : KTOT);
            const float l = sl == kp ? lo : 0.f, h = sl == kp ? hi : 0.f;
            if (MODE == 1) {
              const int n = 8 * (nb * 8 + warp) + r;
              prow[n] = l;
              prow[KTOT + n] = h;
            } else {
              prow[8 * (nb * 16 + 2 * warp) + r] = l;
              prow[8 * (nb * 16 + 2 * warp + 1) + r] = h;
            }
          }
        }
      }
      if (MODE == 1 && (flags & 8)) {
        // Last-arriving part reduces: parts are written above; publish, count, and the CTA
        // that completes the item sums the slots in order (deterministic) and writes inter.
        __threadfence();
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (warp == 0 && lane == 0) {
          const int old = atomicAdd(counters + it, 1);
          hdr[STAGES] = old == kparts - 1;
          if (old == kparts - 1) counters[it] = 0;
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (hdr[STAGES]) {
          __threadfence();
#pragma unroll
          for (int t = 0; t < MT; ++t) {
            if (t >= groups) continue;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              const int tok = row0 + t * 8 + c + i;
              if (tok >= row0 + rows_e) continue;
              const int n = 8 * (nb * 8 + warp) + r;
              float g = 0.f, u = 0.f;
              for (int k = 0; k < kparts; ++k) {
                const float* prow = part + ((size_t)k * part_rows + tok) * (2 * KTOT);
                g += __ldcg(prow + n);
                u += __ldcg(prow + KTOT + n);
              }
              out[(size_t)tok * KTOT + n] = __float2bfloat16_rn(g / (1.f + __expf(-g)) * u);
            }
          }
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");   // hdr[STAGES] is rewritten next item
      }
    } else
#pragma unroll
    for (int h = 0; h < RB; ++h)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= groups) continue;
      const int nbh = nb * RB + h;   // the weights' 128-row block
      float y[2][2];   // [row block 0 / 1 (MODE 2), or gate*up (MODE 1, block 0)][token 2c / 2c+1]
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float lo = (accs[h][0][t][i] + accs[h][1][t][i]) * gsc, hi = (accs[h][0][t][2 + i] + accs[h][1][t][2 + i]) * gsc;
        if (MODE == 1) {
          y[0][i] = lo / (1.f + __expf(-lo)) * hi;
        } else {
          y[0][i] = lo;
          y[1][i] = hi;
        }
      }
      const bool odd = r & 1;
      const int tok = row0 + t * 8 + c + (odd ? 1 : 0);
#pragma unroll
      for (int blk = 0; blk < (MODE == 1 ? 1 : 2); ++blk) {
        const float mine_keep = odd ? y[blk][1] : y[blk][0];
        const float give = odd ? y[blk][0] : y[blk][1];
        const float got = __shfl_xor_sync(0xffffffffu, give, 4);
        const int n = MODE == 1 ? 8 * (nbh * 8 + warp) + (r & ~1) : 8 * (nbh * 16 + 2 * warp + blk) + (r & ~1);
        if (tok < row0 + rows_e) {
          __nv_bfloat162 pr = odd ? __floats2bfloat162_rn(got, mine_keep) : __floats2bfloat162_rn(mine_keep, got);
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)tok * KTOT + n) = pr;
        }
      }
    }
    if (PROF && warp == 0 && lane == 0) {
      prof[prec * PSTR + 0] = smid();
      prof[prec * PSTR + 2] = gtimer();
      prof[prec * PSTR + 5] = clock64() - ct0;
    }
  }
  if (PROF && tid == 0) prof[-PCTA + 2 * (int)blockIdx.x + 1] = gtimer();
  if (flags & 16) asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

// ---------------------------------------------------------------------------------------
// v7: swap-AB on wgmma. v6's remaining shared-memory cost was every warp re-reading the token
// tile (B) with ldmatrix; with wgmma each warpgroup's tensor core reads it from shared memory
// once per k16 step (4 warps share it). A stays the decoded weights in registers (wgmma's
// register A fragment is mma.sync's, per warp: probe daa8882c), so the decode is unchanged.
// Two consumer warpgroups (64 n-rows each = slots 0-7 / 8-15) + one producer warp per CTA.
// The token tile is stored in wgmma's canonical K-major, no-swizzle layout, core matrices
// [k segment (8 k)][8-token block][8 rows][16 B]; descriptor LBO = K-direction stride, SBO =
// 8-token-block stride (probe). The producer fills it with 16-byte cp.async (zero-filled past
// the item's rows) completing on the stage's full barrier (cp.async.mbarrier.arrive.noinc).

// A CUtensorMap's 128 bytes, passed by value (__grid_constant__) to TMA kernels; device code
// only takes its address (so it compiles without cuda.h).
struct alignas(64) TmapBlob {
  unsigned long long v[16];
};

__device__ __forceinline__ void tma_2d(uint32_t dst, const TmapBlob* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

template <int N>
struct WgCfg {
  static constexpr int A_STAGE = N * 256;           // N tokens x 128 k x 2 B
  static constexpr int KSTRIDE = (N / 8) * 128;     // bytes between k segments
  static constexpr int W_STAGE = 16 * 512;
  static constexpr int S_STAGE = 16 * 64;
  static constexpr int STAGE = A_STAGE + W_STAGE + S_STAGE;
};

__device__ __forceinline__ uint64_t wg_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}

template <int N>
__device__ __forceinline__ void wgmma_rs(float* d, const uint32_t* a, uint64_t desc);

template <>
__device__ __forceinline__ void wgmma_rs<8>(float* d, const uint32_t* a, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <>
__device__ __forceinline__ void wgmma_rs<16>(float* d, const uint32_t* a, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <>
__device__ __forceinline__ void wgmma_rs<32>(float* d, const uint32_t* a, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <>
__device__ __forceinline__ void wgmma_rs<64>(float* d, const uint32_t* a, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
               "%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// W0 (occupancy variant): one A buffer, each half chunk's wgmmas waited at once (tensor waits
// are 1-2% of warp time, timed profile 4c3d61e9), so the kernel fits 4 CTAs (8 consumer warps
// per SM sub-partition) instead of 3: decode latency is hidden by more warps, not by overlap.
// BIG: one wgmma batch per chunk (8 k16 steps, one warpgroup meeting per chunk instead of
// two: the fine-timed profile 222d366f put fence + issue at 34-38% of warp time).
// TMA: the token tile arrives by TMA (GEMM2: one 64 x N box per 64-k slab; GEMM1: one row box
// per token and slab, rows being gathered) in the 128-byte-swizzled K-major layout wgmma
// reads natively, instead of 16 B cp.async per token and segment: the producer issued 512
// cp.async per chunk at N = 32 and was the bottleneck (ablation 06bbc21a: -98 us at bs 256).
template <int MODE, int N, int STAGES, bool PROF, int ABL = 0, bool W0 = false, bool BIG = false, bool TMA = false,
          bool VARN = false>
__global__ void __launch_bounds__(288, N >= 64 ? 2 : N >= 32 ? (W0 ? 3 : 2) : W0 && !BIG ? 4 : 3) moe_gemm_wg_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks,
    long long* __restrict__ prof, int* __restrict__ sched, const __grid_constant__ TmapBlob tmap, int pfd) {
  constexpr int abl = PROF ? ABL : 0;   // ablation bits as moe_gemm_ws_kernel's
  using C = WgCfg<N>;
  constexpr bool SINGLE = W0 || BIG;   // one A buffer; each batch waited at once
  constexpr int NCHUNK = KTOT / 128;
  static_assert(NCHUNK % STAGES == 0 && (N == 8 || N == 16 || N == 32 || N == 64), "stages / tokens");
  extern __shared__ __align__(1024) uint8_t smem[];   // 1 KB: the 128 B swizzle's atom
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + STAGES * C::STAGE, bar_empty = bar_full + 8 * STAGES;
  volatile int* hdr = reinterpret_cast<volatile int*>(smem + STAGES * (C::STAGE + 16));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bar_full + 8 * s, TMA ? 1 : 33);    // lane 0's expect_tx arrive (+ 32 lanes' cp.async arrives)
      mbar_init(bar_empty + 8 * s, STAGE_ARRIVALS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int total = *n_active_ptr * n_blocks;

  if (warp == 8) {
    // Producer: lane copies k segment (lane % 16) of tokens lane / 16 + 2i, i < N / 2.
    uint32_t f = 0;
    const int seg = lane % 16;
    for (bool first = true;; first = false) {
      int item = blockIdx.x;
      if (!first) {
        if (lane == 0) item = (int)gridDim.x + atomicAdd(sched, 1);
        item = __shfl_sync(0xffffffffu, item, 0);
      }
      if (item >= total) {
        if (lane == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
        const uint32_t st = f % STAGES;
        if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        if (lane == 0) hdr[st] = -1;
        __syncwarp();
        if (!TMA) mbar_arrive(bar_full + 8 * st);
        if (lane == 0) {
          mbar_arrive(bar_full + 8 * st);
          if (atomicAdd(sched + 1, 1) == (int)gridDim.x - 1) {
            atomicExch(sched, 0);
            atomicExch(sched + 1, 0);
          }
        }
        return;
      }
      const int nb = item % n_blocks;
      const int ch = active[item / n_blocks];
      const int e = ch & 255, mc = ch >> 8;
      const int row0 = exp_start[e] + mc * N;
      const int live = min(N, exp_start[e + 1] - row0);
      int src_row[TMA ? 1 : N / 2];   // (TMA: the token rows arrive as boxes)
#pragma unroll
      for (int i = 0; i < (TMA ? 0 : N / 2); ++i) {
        const int n = lane / 16 + 2 * i;
        src_row[i] = n < live ? (MODE == 1 ? row_route[row0 + n] / 8 : row0 + n) : -1;
      }
      const uint8_t* w_blk = wq + ((size_t)e * n_blocks + nb) * NCHUNK * C::W_STAGE;
      const uint8_t* s_blk = ws + ((size_t)e * n_blocks + nb) * NCHUNK * C::S_STAGE;
      long long pw = 0;
      if (PROF && lane == 0) prof[item * PSTR + 6] = gtimer();
      if (lane == 0)
        for (int kc = 0; kc < pfd; ++kc) {
          prefetch_l2(w_blk + (size_t)kc * C::W_STAGE, C::W_STAGE);
          prefetch_l2(s_blk + (size_t)kc * C::S_STAGE, C::S_STAGE);
        }
      for (int kc = 0; kc < NCHUNK; ++kc, ++f) {
        const uint32_t st = f % STAGES;
        if (lane == 0 && kc + pfd < NCHUNK) {
          prefetch_l2(w_blk + (size_t)(kc + pfd) * C::W_STAGE, C::W_STAGE);
          prefetch_l2(s_blk + (size_t)(kc + pfd) * C::S_STAGE, C::S_STAGE);
        }
        long long c0 = PROF ? clock64() : 0;
        if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        if (PROF) pw += clock64() - c0;
        const uint32_t sb = base + st * C::STAGE, full = bar_full + 8 * st;
        if (lane == 0) {
          hdr[st] = item;
          const uint32_t act_tx = !TMA || (abl & 6) ? 0 : N * 256;
          mbar_expect_tx(full, (abl & 4) ? 0 : C::W_STAGE + C::S_STAGE + act_tx);
          if (!(abl & 4)) {
            bulk_g2s(sb + C::A_STAGE, w_blk + (size_t)kc * C::W_STAGE, C::W_STAGE, full);
            bulk_g2s(sb + C::A_STAGE + C::W_STAGE, s_blk + (size_t)kc * C::S_STAGE, C::S_STAGE, full);
          }
        }
        if (TMA) {
          __syncwarp();   // lane 0's expect_tx precedes the boxes' complete_tx
          if (!(abl & 6)) {
#pragma unroll
            for (int slab = 0; slab < 2; ++slab) {
              const uint32_t dst = sb + slab * (N * 128);
              // GEMM1's token rows were gathered in expert order beforehand (gather_rows), so
              // both GEMMs load rows row0 .. row0 + N - 1 as one box per slab.
              if (lane == 0) tma_2d(dst, &tmap, kc * 128 + slab * 64, row0, full);
            }
          }
          continue;
        }
#pragma unroll
        for (int i = 0; i < (TMA ? 0 : N / 2); ++i) {
          const int n = lane / 16 + 2 * i;
          const uint32_t dst = sb + seg * C::KSTRIDE + (n / 8) * 128 + (n % 8) * 16;
          const bool ok = src_row[i] >= 0;
          const uint8_t* src = reinterpret_cast<const uint8_t*>(a_src + (size_t)(ok ? src_row[i] : 0) * KTOT) + kc * 256 + seg * 16;
          if (!(abl & 6))
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16u : 0u));
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(full) : "memory");
      }
      if (PROF && lane == 0) prof[item * PSTR + 4] = pw;
    }
  }

  // Consumers: warp w owns slots 2w (rows 16 (w % 4) + r of its warpgroup's 64) and 2w + 1 (+8).
  const int r = lane / 4, c = (lane % 4) * 2;
  const uint32_t w_src = base + C::A_STAGE + ((warp * 2) * 32 + lane) * 16;
  const uint32_t s_src = base + C::A_STAGE + C::W_STAGE + (warp * 2) * 64 + r * 8;
  uint32_t f = 0;
  for (;;) {
    mbar_wait(bar_full, (f / STAGES) & 1);
    const int item = hdr[0];
    if (item < 0) break;
    const int nb = item % n_blocks;
    const int ch = active[item / n_blocks];
    const int e = ch & 255, mc = ch >> 8;
    const int row0 = exp_start[e] + mc * N;
    const int rows_e = exp_start[e + 1] - row0;
    const float gsc = gs[e] * 256.0f;
    // abl & 32: cycles waiting for data / tensor core; abl & 64 also: decode (stage data ready
    // -> A fragments built) and issue (wgmma.fence .. commit: includes the warpgroup's meeting)
    long long ct0 = 0, t_full = 0, t_mma = 0, t_dec = 0, t_iss = 0, tm = 0;
    if (PROF && warp == 0) {
      ct0 = clock64();
      if (lane == 0) prof[item * PSTR + 1] = gtimer();
    }
    float d[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
    // A fragments, double-buffered by half chunk (4 k16 steps): the tensor core runs one
    // half's wgmmas while the warp decodes the next.
    uint32_t afr[SINGLE ? 1 : 2][BIG ? 8 : 4][4];
    // VARN: the tile holds up to N = 64 token rows (layout and TMA box fixed), but the wgmma
    // width follows the chunk's live rows (16 / 32 / 64): N 16 runs at ~55% of the tensor
    // peak and N 32 / 64 at ~90% (rate probe), so small experts stop paying for a 64-wide tile.
    auto kloop = [&](auto nn) {
      constexpr int NI = decltype(nn)::value;
#pragma unroll 1
    for (int kc0 = 0; kc0 < NCHUNK; kc0 += STAGES, f += STAGES) {
      const uint32_t parity = (f / STAGES) & 1;
#pragma unroll
      for (int st = 0; st < STAGES; ++st) {
        if (abl & 32) {
          const long long t0 = clock64();
          mbar_wait(bar_full + 8 * st, parity);
          t_full += clock64() - t0;
        } else {
          mbar_wait(bar_full + 8 * st, parity);
        }
        if (abl & 64) tm = clock64();
        if (abl & 1) {
          // no math: release this stage at once (and nothing is ever released late)
          stage_release(bar_empty + 8 * st, lane);
          continue;
        }
        if (!TMA) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // cp.async (generic) -> wgmma (async)
        uint4 wv[2];
        uint2 sv[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(wv[j].x), "=r"(wv[j].y), "=r"(wv[j].z), "=r"(wv[j].w)
                       : "r"(w_src + st * C::STAGE + j * 512));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(sv[j].x), "=r"(sv[j].y)
                       : "r"(s_src + st * C::STAGE + j * 64));
        }
        // no swizzle: core matrices, K stride in LBO; TMA: 128 B swizzle K-major, 8-row groups
        // 1024 B apart (SBO), 64-k slabs N * 128 B apart, k16 steps 32 B apart within a slab.
        const uint64_t desc0 = TMA ? (wg_desc(base + st * C::STAGE, 16, 1024) | (1ull << 62))
                                   : wg_desc(base + st * C::STAGE, C::KSTRIDE, 128);
#pragma unroll
        for (int hc = 0; hc < 2; ++hc) {          // half chunk: words 2hc, 2hc+1 = steps 4hc .. 4hc+3
#pragma unroll
          for (int ww = 0; ww < 2; ++ww) {
            const int w = 2 * hc + ww;
            uint32_t v[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) unpack8(w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w, v[j]);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int step = 2 * w + half;
              uint32_t* a = afr[SINGLE ? 0 : hc][(BIG ? 4 * hc : 0) + 2 * ww + half];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const uint32_t sw = w < 2 ? sv[j].x : sv[j].y;
                const uint32_t sp = half == 0 ? scale_even(sw) : scale_odd(sw);
                if (abl & 16) {
                  a[j] = v[j][0] ^ wv[j].x;
                  a[2 + j] = wv[j].y;
                } else if ((step % 4) < 2) {
                  a[j] = mul2h<0>(v[j][2 * half], sp);
                  a[2 + j] = mul2h<0>(v[j][2 * half + 1], sp);
                } else {
                  a[j] = mul2h<1>(v[j][2 * half], sp);
                  a[2 + j] = mul2h<1>(v[j][2 * half + 1], sp);
                }
              }
            }
          }
          if (BIG && hc == 0) continue;   // decode the second half before the one batch
          long long ti = 0;
          if (abl & 64) {
            ti = clock64();
            t_dec += ti - tm;
          }
          asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
          for (int i = 0; i < (BIG ? 8 : 4); ++i) {
            const int step = BIG ? i : 4 * hc + i;
            uint32_t* a = afr[SINGLE ? 0 : hc][i];
            if (abl & 8) {
              d[0] += __int_as_float((a[0] ^ a[3]) & 0x3f800000u);
            } else {
              const uint32_t off = TMA ? (step / 4) * (N * 128) + (step % 4) * 32 : 2 * step * C::KSTRIDE;
              wgmma_rs<NI>(d, a, desc0 + ((uint64_t)off >> 4));
            }
          }
          asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
          if (abl & 64) t_iss += clock64() - ti;
          // <= 1 group in flight: the other half's A buffer (and, after half 1, this stage's
          // token tile once the next wait completes) are free again.
          if (abl & 32) {
            const long long t0 = clock64();
            if (SINGLE) asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            else asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
            t_mma += clock64() - t0;
            if (abl & 64) tm = clock64();
          } else if (SINGLE) {
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
          } else {
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
          }
        }
        // This stage's first half is done (waited above); its second half may still read the
        // token tile: release the stage after the next half's wait, i.e. one stage late.
        if (abl & 1) continue;
        if (SINGLE) {
          stage_release(bar_empty + 8 * st, lane);   // all of this stage's wgmmas are complete
          continue;
        }
        if (st > 0) stage_release(bar_empty + 8 * (st - 1), lane);
        else if (kc0 > 0) stage_release(bar_empty + 8 * (STAGES - 1), lane);
      }
    }
    };
    if constexpr (VARN) {
      if (rows_e <= 16) kloop(IntC<16>{});
      else if (rows_e <= 32) kloop(IntC<32>{});
      else kloop(IntC<N>{});
    } else {
      kloop(IntC<N>{});
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (!(abl & 1) && !SINGLE) stage_release(bar_empty + 8 * (STAGES - 1), lane);

    // Epilogue (v6's): d[4t + i] = C[n = r / r + 8][token 8t + c + i]; pair columns by shuffle.
#pragma unroll
    for (int t = 0; t < N / 8; ++t) {
      if (t * 8 >= rows_e) continue;
      float y[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float lo = d[4 * t + i] * gsc, hi = d[4 * t + 2 + i] * gsc;
        if (MODE == 1) {
          y[0][i] = lo / (1.f + __expf(-lo)) * hi;
        } else {
          y[0][i] = lo;
          y[1][i] = hi;
        }
      }
      const bool odd = r & 1;
      const int tok = row0 + t * 8 + c + (odd ? 1 : 0);
#pragma unroll
      for (int blk = 0; blk < (MODE == 1 ? 1 : 2); ++blk) {
        const float keep = odd ? y[blk][1] : y[blk][0];
        const float give = odd ? y[blk][0] : y[blk][1];
        const float got = __shfl_xor_sync(0xffffffffu, give, 4);
        const int n = MODE == 1 ? 8 * (nb * 8 + warp) + (r & ~1) : 8 * (nb * 16 + 2 * warp + blk) + (r & ~1);
        if (tok < row0 + rows_e) {
          __nv_bfloat162 pr = odd ? __floats2bfloat162_rn(got, keep) : __floats2bfloat162_rn(keep, got);
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)tok * KTOT + n) = pr;
        }
      }
    }
    if (PROF && warp == 0 && lane == 0) {
      prof[item * PSTR + 0] = smid();
      prof[item * PSTR + 2] = gtimer();
      prof[item * PSTR + 5] = clock64() - ct0;
      if (abl & 32) {
        prof[item * PSTR + 3] = t_full;
        prof[item * PSTR + 7] = t_mma;
        prof[item * PSTR + 8] = t_dec;
        prof[item * PSTR + 9] = t_iss;
      }
    }
  }
}
// ---------------------------------------------------------------------------------------
// v9 "tiny": the whole expert MLP for tiny batches (<= 8 rows per expert) in ONE persistent grid
// (one CTA per SM). Every byte of both GEMMs' weights leaves HBM in one burst from the start:
// each CTA TMA-loads its first GEMM2 unit (a 128-row W2 block x one K half: 147 KB) into shared
// memory and L2-prefetches any further ones, while its producer streams its GEMM1 units (a
// 128-row W13 block x one K half) through a small ring. The last-arriving GEMM1 half of a block
// writes silu(gate) * up (inter); a GEMM2 unit waits only for its expert's 64 GEMM1 blocks
// (counter), loads that expert's inter rows and multiplies the weights already in shared
// memory. GEMM2 halves go out as fp32 parts for the post-norm (_post_norm_parts_kernel, KP 2).
// Counters reset themselves (graph replay). Same partial-sum order and roundings as the
// split-K v6 path (s1dw2fp), so the result matches it.
#define TINY_WCH 8192
#define TINY_SCH 1024
#define TINY_AROW 272     // one 128-k chunk of a token row (256 B) + 16 B
#define TINY_A2ROW 4112   // one GEMM2 K half of a token row (2048 k = 4096 B) + 16 B
// RING: ring stages; RES: the first GEMM2 unit resident in shared memory (else every GEMM2 unit
// is L2-prefetched at the start and streamed through the ring, and the ring gets that space).
template <int TINY_RING, bool RES>
struct TinyCfg {
  static constexpr int RSTAGE = TINY_WCH + TINY_SCH + 8 * TINY_AROW;   // ring stage: W, S, act rows
  static constexpr int W2W = 16 * TINY_WCH, W2S = 16 * TINY_SCH;        // resident GEMM2 unit
  static constexpr int OFF_W2 = TINY_RING * RSTAGE;
  static constexpr int OFF_A2 = OFF_W2 + (RES ? W2W + W2S : 0);
  static constexpr int OFF_BAR = OFF_A2 + 8 * TINY_A2ROW;
  static constexpr int OFF_MISC = OFF_BAR + 8 * (2 * TINY_RING + 1);
  static constexpr int SMEM = OFF_MISC + 4 * 40 + 64;
};
#define TINY_CNT1 0          // [32 experts x 64 blocks] GEMM1 half arrivals
#define TINY_DONE1 2048      // [32] GEMM1 blocks completed per expert
#define TINY_DONE2 2080      // [32] GEMM2 units completed per expert
#define TINY_CNT_N 2112

// One warp, one 128-k chunk: acc[parity][4] += W(16 rows of this warp) x act(8 token rows).
__device__ __forceinline__ void tiny_chunk(float acc[2][4], uint32_t wsrc, uint32_t ssrc, uint32_t abase, uint32_t arow,
                                           int lane) {
  uint4 wv[2];
  uint2 sv[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(wv[j].x), "=r"(wv[j].y), "=r"(wv[j].z), "=r"(wv[j].w)
                 : "r"(wsrc + j * 512));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(sv[j].x), "=r"(sv[j].y) : "r"(ssrc + j * 64));
  }
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t v[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) unpack8(w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w, v[j]);
    uint32_t b[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                 : "r"(abase + (lane % 8) * arow + (4 * w + lane / 8) * 16));
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int step = 2 * w + half;
      uint32_t a[4];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t sw = w < 2 ? sv[j].x : sv[j].y;
        const uint32_t sp = half == 0 ? scale_even(sw) : scale_odd(sw);
        if ((step % 4) < 2) {
          a[j] = mul2h<0>(v[j][2 * half], sp);
          a[2 + j] = mul2h<0>(v[j][2 * half + 1], sp);
        } else {
          a[j] = mul2h<1>(v[j][2 * half], sp);
          a[2 + j] = mul2h<1>(v[j][2 * half + 1], sp);
        }
      }
      mma16816(acc[half], a, b[2 * half], b[2 * half + 1]);
    }
  }
}

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// flags & 128: a programmatic dependent of the route (waits for it before reading its output,
// signals its own dependent, the post-norm, at the start).
template <int TINY_RING, bool RES>
__global__ void __launch_bounds__(288, RES ? 1 : 2) moe_tiny_kernel(
    const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ row_route, const int32_t* __restrict__ exp_start,
    const uint8_t* __restrict__ w13q, const uint8_t* __restrict__ w13s, const float* __restrict__ gs13,
    const uint8_t* __restrict__ w2q, const uint8_t* __restrict__ w2s, const float* __restrict__ gs2,
    __nv_bfloat16* __restrict__ inter, float* __restrict__ part1, float* __restrict__ part2, int P, int* __restrict__ cnt,
    int flags) {
  using C = TinyCfg<TINY_RING, RES>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + C::OFF_BAR, bar_empty = bar_full + 8 * TINY_RING, bar_w2 = bar_empty + 8 * TINY_RING;
  int* misc = reinterpret_cast<int*>(smem + C::OFF_MISC);   // [0] touched experts, [1..32] their ids, [33] flag
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // flags & 0x8000 (probe): per-CTA phase times (globaltimer) at cnt + 4096: [cta][start, GEMM1 done,
  // first GEMM2 wait passed, GEMM2 done] as int64.
  long long* stamp = (flags & 0x8000) ? reinterpret_cast<long long*>(cnt + 4096) + 4 * blockIdx.x : nullptr;
  if (stamp && tid == 0) stamp[0] = gtimer();
  if (tid == 0) {
    for (int s = 0; s < TINY_RING; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, STAGE_ARRIVALS);
    }
    mbar_init(bar_w2, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (flags & 128) {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  }
  if (tid == 0) {
    int n = 0;
    for (int e = 0; e < 32; ++e)
      if (exp_start[e + 1] > exp_start[e]) misc[1 + n++] = e;
    misc[0] = n;
  }
  __syncthreads();
  const int ne = misc[0];
  const int G = gridDim.x;
  const int U1 = ne * 128, U2 = ne * 64;   // GEMM1 half units, GEMM2 units

  if (warp == 8) {
    // Producer.
    if (lane == 0 && (int)blockIdx.x < U2) {   // the first GEMM2 unit: into shared memory now
      const int v = blockIdx.x, h = v / (ne * 32), e = misc[1 + (v % (ne * 32)) / 32], b = v % 32;
      const size_t c0 = ((size_t)e * 32 + b) * 32 + h * 16;
      if (RES) {
        mbar_expect_tx(bar_w2, C::W2W + C::W2S);
        bulk_g2s(base + C::OFF_W2, w2q + c0 * TINY_WCH, C::W2W, bar_w2);
        bulk_g2s(base + C::OFF_W2 + C::W2W, w2s + c0 * TINY_SCH, C::W2S, bar_w2);
      } else {
        prefetch_l2(w2q + c0 * TINY_WCH, C::W2W);
        prefetch_l2(w2s + c0 * TINY_SCH, C::W2S);
      }
      for (int v2 = v + G; v2 < U2; v2 += G) {   // further units: to L2, streamed through the ring later
        const int h2 = v2 / (ne * 32), e2 = misc[1 + (v2 % (ne * 32)) / 32], b2 = v2 % 32;
        const size_t c2 = ((size_t)e2 * 32 + b2) * 32 + h2 * 16;
        prefetch_l2(w2q + c2 * TINY_WCH, C::W2W);
        prefetch_l2(w2s + c2 * TINY_SCH, C::W2S);
      }
    }
    uint32_t f = 0;
    for (int u = blockIdx.x; u < U1; u += G) {   // GEMM1 half units
      const int h = u / (ne * 64), e = misc[1 + (u % (ne * 64)) / 64], b = u % 64;
      const int row0 = exp_start[e], live = min(8, exp_start[e + 1] - row0);
      int src = 0;
      if (lane < live) src = row_route[row0 + lane] / 8;
      const uint8_t* a_row = reinterpret_cast<const uint8_t*>(x + (size_t)src * KTOT);
      const size_t cb = ((size_t)e * 64 + b) * 32 + h * 16;
      for (int c = 0; c < 16; ++c, ++f) {
        const uint32_t st = f % TINY_RING;
        if (f >= TINY_RING) mbar_wait(bar_empty + 8 * st, ((f / TINY_RING) - 1) & 1);
        const uint32_t sb = base + st * C::RSTAGE, full = bar_full + 8 * st;
        if (lane == 0) {
          mbar_expect_tx(full, TINY_WCH + TINY_SCH + live * 256);
          bulk_g2s(sb, w13q + (cb + c) * TINY_WCH, TINY_WCH, full);
          bulk_g2s(sb + TINY_WCH, w13s + (cb + c) * TINY_SCH, TINY_SCH, full);
        }
        __syncwarp();
        if (lane < live) bulk_g2s(sb + TINY_WCH + TINY_SCH + lane * TINY_AROW, a_row + (h * 16 + c) * 256, 256, full);
      }
    }
    for (int v = blockIdx.x + (RES ? G : 0); v < U2; v += G) {   // GEMM2 units not resident: weights via the ring
      const int h = v / (ne * 32), e = misc[1 + (v % (ne * 32)) / 32], b = v % 32;
      const size_t cb = ((size_t)e * 32 + b) * 32 + h * 16;
      for (int c = 0; c < 16; ++c, ++f) {
        const uint32_t st = f % TINY_RING;
        if (f >= TINY_RING) mbar_wait(bar_empty + 8 * st, ((f / TINY_RING) - 1) & 1);
        const uint32_t sb = base + st * C::RSTAGE, full = bar_full + 8 * st;
        if (lane == 0) {
          mbar_expect_tx(full, TINY_WCH + TINY_SCH);
          bulk_g2s(sb, w2q + (cb + c) * TINY_WCH, TINY_WCH, full);
          bulk_g2s(sb + TINY_WCH, w2s + (cb + c) * TINY_SCH, TINY_SCH, full);
        }
        __syncwarp();
      }
    }
    return;
  }

  // Consumers (warps 0-7: 16 output rows each of a 128-row block).
  const int r = lane / 4, c = (lane % 4) * 2;
  uint32_t f = 0;
  for (int u = blockIdx.x; u < U1; u += G) {
    const int h = u / (ne * 64), e = misc[1 + (u % (ne * 64)) / 64], b = u % 64;
    const int row0 = exp_start[e], rows_e = min(8, exp_start[e + 1] - row0);
    float acc[2][4] = {};
    for (int k = 0; k < 16; ++k, ++f) {
      const uint32_t st = f % TINY_RING;
      mbar_wait(bar_full + 8 * st, (f / TINY_RING) & 1);
      const uint32_t sb = base + st * C::RSTAGE;
      tiny_chunk(acc, sb + ((warp * 2) * 32 + lane) * 16, sb + TINY_WCH + (warp * 2) * 64 + r * 8,
                 sb + TINY_WCH + TINY_SCH, TINY_AROW, lane);
      stage_release(bar_empty + 8 * st, lane);
    }
    const float gsc = gs13[e] * 256.0f;
    const int n = 8 * (b * 8 + warp) + r;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int tok = row0 + c + i;
      if (tok >= row0 + rows_e) continue;
      float* prow = part1 + ((size_t)h * P + tok) * (2 * KTOT);
      prow[n] = (acc[0][i] + acc[1][i]) * gsc;
      prow[KTOT + n] = (acc[0][2 + i] + acc[1][2 + i]) * gsc;
    }
    __threadfence();
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    if (tid == 0) {
      const int old = atomicAdd(cnt + TINY_CNT1 + e * 64 + b, 1);
      misc[33] = old == 1;
      if (old == 1) cnt[TINY_CNT1 + e * 64 + b] = 0;
    }
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    if (misc[33]) {   // both K halves are in: silu(gate) * up for this block's 64 outputs
      __threadfence();
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int tok = row0 + c + i;
        if (tok >= row0 + rows_e) continue;
        const float* p0 = part1 + (size_t)tok * (2 * KTOT);
        const float* p1 = part1 + ((size_t)P + tok) * (2 * KTOT);
        const float g = __ldcg(p0 + n) + __ldcg(p1 + n), uu = __ldcg(p0 + KTOT + n) + __ldcg(p1 + KTOT + n);
        inter[(size_t)tok * KTOT + n] = __float2bfloat16_rn(g / (1.f + __expf(-g)) * uu);
      }
      __threadfence();
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      if (tid == 0) atomicAdd(cnt + TINY_DONE1 + e, 1);
    }
    asm volatile("bar.sync 1, 256;\n" ::: "memory");   // misc[33] is rewritten next unit
  }

  if (stamp && tid == 0) stamp[1] = gtimer();
  for (int v = blockIdx.x, k = 0; v < U2; v += G, ++k) {
    const int h = v / (ne * 32), e = misc[1 + (v % (ne * 32)) / 32], b = v % 32;
    const int row0 = exp_start[e], rows_e = min(8, exp_start[e + 1] - row0);
    if (tid == 0)
      while (ld_acquire(cnt + TINY_DONE1 + e) < 64) {
      }
    if (stamp && tid == 0 && k == 0) stamp[2] = gtimer();
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    // This expert's inter rows, K half h, into the activation region (row stride TINY_A2ROW).
    for (int o = tid; o < rows_e * 256; o += 256) {   // 16 B per thread step: 256 per row
      const int row = o / 256, q = o % 256;
      const uint4 val = __ldcg(reinterpret_cast<const uint4*>(inter + (size_t)(row0 + row) * KTOT + h * 2048) + q);
      *reinterpret_cast<uint4*>(smem + C::OFF_A2 + row * TINY_A2ROW + q * 16) = val;
    }
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    float acc[2][4] = {};
    if (RES && k == 0) {
      mbar_wait(bar_w2, 0);
      const uint32_t wb = base + C::OFF_W2, sbb = base + C::OFF_W2 + C::W2W;
      for (int q = 0; q < 16; ++q)
        tiny_chunk(acc, wb + q * TINY_WCH + ((warp * 2) * 32 + lane) * 16, sbb + q * TINY_SCH + (warp * 2) * 64 + r * 8,
                   base + C::OFF_A2 + q * 256, TINY_A2ROW, lane);
    } else {
      for (int q = 0; q < 16; ++q, ++f) {
        const uint32_t st = f % TINY_RING;
        mbar_wait(bar_full + 8 * st, (f / TINY_RING) & 1);
        const uint32_t sb = base + st * C::RSTAGE;
        tiny_chunk(acc, sb + ((warp * 2) * 32 + lane) * 16, sb + TINY_WCH + (warp * 2) * 64 + r * 8,
                   base + C::OFF_A2 + q * 256, TINY_A2ROW, lane);
        stage_release(bar_empty + 8 * st, lane);
      }
    }
    const float gsc = gs2[e] * 256.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int tok = row0 + c + i;
      if (tok >= row0 + rows_e) continue;
      float* prow = part2 + ((size_t)h * P + tok) * KTOT;
      prow[8 * (b * 16 + 2 * warp) + r] = (acc[0][i] + acc[1][i]) * gsc;
      prow[8 * (b * 16 + 2 * warp + 1) + r] = (acc[0][2 + i] + acc[1][2 + i]) * gsc;
    }
    asm volatile("bar.sync 1, 256;\n" ::: "memory");   // the activation region is rewritten next unit
    if (tid == 0) {
      const int old = atomicAdd(cnt + TINY_DONE2 + e, 1);
      if (old == 63) {   // every GEMM2 unit of e has passed its wait: reset for the next replay
        cnt[TINY_DONE1 + e] = 0;
        cnt[TINY_DONE2 + e] = 0;
      }
    }
  }
  if (stamp && tid == 0) stamp[3] = gtimer();
}

"""

MOE_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda.h>
#include <cudaTypedefs.h>

// cuTensorMapEncodeTiled through the runtime's driver entry point (no -lcuda needed).
static PFN_cuTensorMapEncodeTiled_v12000 tmap_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault);
#endif
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// [rows, 4096] bf16 activations, boxes of 64 k x box_rows rows, 128-byte swizzle.
static TmapBlob act_tmap(const torch::Tensor& a, int box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {4096, (cuuint64_t)a.size(0)};
  cuuint64_t strides[1] = {4096 * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t elem[2] = {1, 1};
  CUresult r = tmap_encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.data_ptr(), dims, strides, box, elem,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  TmapBlob b;
  static_assert(sizeof(CUtensorMap) == sizeof(TmapBlob), "tensor map size");
  memcpy(&b, &m, sizeof(b));
  return b;
}

template <int MODE, int MT, int STAGES>
void launch(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
            torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
            int n_blocks, int grid) {
  constexpr int NGW = MODE == 1 ? 2 : 1;
  size_t smem = (size_t)STAGES * (MT * 16 * 128 * 2 + 8 * NGW * 512 + 8 * NGW * 64);
  auto k = moe_gemm_kernel<MODE, MT, NGW, STAGES>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  // A persistent grid larger than what can be resident would serialize items onto late CTAs:
  // cap the requested CTAs per SM by the occupancy shared memory and registers allow.
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 256, smem);
  grid = std::min(grid, std::max(1, per_sm) * sms);
  k<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), row_route.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), ws.data_ptr<uint8_t>(), gs.data_ptr<float>(),
      active.data_ptr<int32_t>(), n_active.data_ptr<int32_t>(), exp_start.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks);
}

template <int MODE, int MT, int STAGES, int KS, bool PROF, int ABL>
void launch_ws(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
               torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
               int n_blocks, int grid, long long* prof, int* sched) {
  constexpr int NGW = MODE == 1 ? 2 : 1;
  constexpr int THREADS = 32 * (8 * KS + 1);
  size_t smem = (size_t)STAGES * (WsCfg<MODE, MT, NGW>::STAGE + 32) + (size_t)(KS - 1) * 8 * MT * NGW * 4 * 32 * 4;
  auto k = moe_gemm_ws_kernel<MODE, MT, NGW, STAGES, KS, PROF, ABL>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, THREADS, smem);
  grid = std::min(grid, std::max(1, per_sm) * sms);
  k<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), row_route.data_ptr<int32_t>(),
      wq.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), gs.data_ptr<float>(),
      active.data_ptr<int32_t>(), n_active.data_ptr<int32_t>(), exp_start.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, prof, sched);
}

void moe_gemm_ws(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, int64_t ks,
                 torch::Tensor prof, int64_t abl, torch::Tensor sched) {
  // sched: int32 [2], zero before the first launch; the kernel leaves it zero.
  int* sp = sched.data_ptr<int32_t>();
  // prof: empty -> production kernel; int64 [items * 8] -> the profiling build with ablation
  // abl (only in the module compiled with -DOKC_PROF).
  long long* pp = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  if (pp) {
#ifdef OKC_PROF
#define PCASE(M_, T_, S_, K_, A_) if (mode == M_ && mt == T_ && stages == S_ && ks == K_ && abl == A_) { launch_ws<M_, T_, S_, K_, true, A_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp); return; }
#define PCASES(A_) PCASE(1, 1, 4, 1, A_) PCASE(2, 1, 4, 1, A_)
    PCASES(0) PCASES(1) PCASES(2) PCASES(3) PCASES(4) PCASES(12) PCASES(20) PCASES(28)
    PCASE(1, 2, 4, 1, 0) PCASE(2, 2, 4, 1, 0) PCASE(1, 1, 8, 2, 0) PCASE(2, 1, 8, 2, 0)
#endif
    TORCH_CHECK(false, "moe_gemm_ws: no profiling build for this configuration (needs -DOKC_PROF)");
  }
#define WCASE(M_, T_, S_, K_) if (mode == M_ && mt == T_ && stages == S_ && ks == K_) { launch_ws<M_, T_, S_, K_, false, 0>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp); return; }
  WCASE(1, 1, 4, 1) WCASE(2, 1, 4, 1) WCASE(1, 2, 4, 1) WCASE(2, 2, 4, 1) WCASE(1, 1, 8, 1) WCASE(2, 1, 8, 1)
  WCASE(1, 1, 8, 2) WCASE(2, 1, 8, 2) WCASE(1, 1, 4, 2) WCASE(2, 1, 4, 2)   // KS * 8 + 1 warps <= 32
  TORCH_CHECK(false, "moe_gemm_ws: unsupported mode / m tiles / stages");
}

template <int MODE, int MT, int STAGES, bool PROF, int ABL = 0, int SUB = 1, int RB = 1>
void launch_sab(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
                torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
                int n_blocks, int grid, long long* prof, int* sched, int kparts, float* part, int part_rows, int sk,
                const uint8_t* pf_w, const uint8_t* pf_s, int* counters, PostArgs pa) {
  size_t smem = (size_t)STAGES * (SabCfg<MT, SUB, RB>::STAGE + 32);
  auto k = moe_gemm_sab_kernel<MODE, MT, STAGES, PROF, ABL, SUB, RB>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 288, smem);
  grid = std::min(grid, std::max(1, per_sm) * sms);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (sk & (2 | 16 | 128)) ? 1 : 0;   // flags 2 / 16 / 128: a programmatic dependent of the previous kernel
  cudaLaunchKernelEx(&cfg, k,
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), (const int32_t*)row_route.data_ptr<int32_t>(),
      (const uint8_t*)wq.data_ptr<uint8_t>(), (const uint8_t*)ws.data_ptr<uint8_t>(), (const float*)gs.data_ptr<float>(),
      (const int32_t*)active.data_ptr<int32_t>(), (const int32_t*)n_active.data_ptr<int32_t>(),
      (const int32_t*)exp_start.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, prof, sched, kparts, part, part_rows, sk, pf_w, pf_s, counters,
      pa);
}

void moe_gemm_sab(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                  torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                  torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                  torch::Tensor sched, int64_t abl, int64_t kparts, torch::Tensor part, int64_t sk, torch::Tensor pf_w,
                  torch::Tensor pf_s, torch::Tensor counters, torch::Tensor post) {
  int* sp = sched.data_ptr<int32_t>();
  PostArgs pa = {};   // post: CPU int64 [12] (atomic post) or empty
  if (post.numel() >= 12) {
    const int64_t* q = post.data_ptr<int64_t>();
    pa.y32 = reinterpret_cast<float*>(q[0]);
    pa.out32 = reinterpret_cast<float*>(q[1]);
    pa.outb = reinterpret_cast<__nv_bfloat16*>(q[2]);
    pa.norm_w = reinterpret_cast<const __nv_bfloat16*>(q[3]);
    pa.topk_w = reinterpret_cast<const float*>(q[4]);
    pa.pos = reinterpret_cast<const int32_t*>(q[5]);
    pa.tok_cnt = reinterpret_cast<int*>(q[6]);
    pa.chunk_cnt = reinterpret_cast<int*>(q[7]);
    const int32_t eb = (int32_t)q[8];
    pa.eps = *reinterpret_cast<const float*>(&eb);
    pa.M = (int)q[9];
    pa.P = (int)q[10];
    pa.mode = (int)q[11];
  }
  int* cnt = counters.numel() ? counters.data_ptr<int32_t>() : nullptr;
  TORCH_CHECK(!(sk & 8) || cnt, "moe_gemm_sab: the fused split epilogue needs counters");
  const uint8_t* pfw = pf_w.numel() ? pf_w.data_ptr<uint8_t>() : nullptr;
  const uint8_t* pfs = pf_s.numel() ? reinterpret_cast<const uint8_t*>(pf_s.data_ptr()) : nullptr;
  float* pt = part.numel() ? part.data_ptr<float>() : nullptr;
  const int prow = part.numel() ? (int)part.size(1) : 0;
  const int kp = (int)kparts;
  TORCH_CHECK(kp == 1 || pt, "moe_gemm_sab: split K needs a partial buffer");
  long long* pp = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  if (pp) {
#ifdef OKC_PROF
#define SPCASE(M_, T_, S_, A_) if (mode == M_ && mt == T_ && stages == S_ && abl == A_) { launch_sab<M_, T_, S_, true, A_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
#define SPCASES(A_) SPCASE(1, 2, 4, A_) SPCASE(2, 2, 4, A_) SPCASE(1, 1, 4, A_) SPCASE(2, 1, 4, A_)
    SPCASES(0) SPCASES(1) SPCASES(2) SPCASES(3) SPCASES(4) SPCASES(12) SPCASES(20) SPCASES(28)
    SPCASE(1, 1, 8, 0) SPCASE(2, 1, 8, 0)
#define SPCASE2(M_, T_, S_, A_) if (mode == M_ && mt == T_ && stages == 200 + S_ && abl == A_) { launch_sab<M_, T_, S_, true, A_, 2>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
#define SPCASES2(A_) SPCASE2(1, 1, 4, A_) SPCASE2(2, 1, 4, A_)
    SPCASES2(0) SPCASES2(1) SPCASES2(2) SPCASES2(3) SPCASES2(4) SPCASES2(12) SPCASES2(20) SPCASES2(28)
#define SPCASE4(M_, T_, S_, A_) if (mode == M_ && mt == T_ && stages == 400 + S_ && abl == A_) { launch_sab<M_, T_, S_, true, A_, 4>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
#define SPCASES4(A_) SPCASE4(1, 2, 2, A_) SPCASE4(2, 2, 2, A_)
    SPCASES4(0) SPCASES4(1) SPCASES4(3) SPCASES4(4) SPCASES4(12) SPCASES4(28)
#endif
    TORCH_CHECK(false, "moe_gemm_sab: no profiling build for this configuration (needs -DOKC_PROF)");
  }
#define SCASE(M_, T_, S_) if (mode == M_ && mt == T_ && stages == S_) { launch_sab<M_, T_, S_, false>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
  SCASE(1, 1, 4) SCASE(2, 1, 4) SCASE(1, 2, 4) SCASE(2, 2, 4) SCASE(1, 1, 8) SCASE(2, 1, 8) SCASE(1, 2, 8) SCASE(2, 2, 8)
  // 200 + S: two 128-k chunks per stage (16 KB weight copies)
#define SCASE2(M_, T_, S_) if (mode == M_ && mt == T_ && stages == 200 + S_) { launch_sab<M_, T_, S_, false, 0, 2>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
  SCASE2(1, 1, 2) SCASE2(2, 1, 2) SCASE2(1, 2, 2) SCASE2(2, 2, 2) SCASE2(1, 1, 4) SCASE2(2, 1, 4)
  // 400 + S: four 128-k chunks per stage (32 KB weight copies)
#define SCASE4(M_, T_, S_) if (mode == M_ && mt == T_ && stages == 400 + S_) { launch_sab<M_, T_, S_, false, 0, 4>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
  SCASE4(1, 1, 2) SCASE4(2, 1, 2) SCASE4(1, 2, 2) SCASE4(2, 2, 2)
  // 1000 + 100 * SUB + S: two 128-row blocks per item (RB 2: 32 rows per consumer warp)
#define SCASER(M_, T_, U_, S_) if (mode == M_ && mt == T_ && stages == 1000 + 100 * U_ + S_) { launch_sab<M_, T_, S_, false, 0, U_, 2>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
  SCASER(1, 1, 1, 4) SCASER(2, 1, 1, 4) SCASER(1, 2, 1, 4) SCASER(2, 2, 1, 4) SCASER(1, 1, 2, 2) SCASER(2, 1, 2, 2)
  SCASER(1, 2, 2, 2) SCASER(2, 2, 2, 2)
  // 2000 + 100 * SUB + S: four 128-row blocks per item (RB 4: 64 rows per consumer warp)
#define SCASER4(M_, T_, U_, S_) if (mode == M_ && mt == T_ && stages == 2000 + 100 * U_ + S_) { launch_sab<M_, T_, S_, false, 0, U_, 4>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk, pfw, pfs, cnt, pa); return; }
  SCASER4(1, 2, 1, 2) SCASER4(2, 2, 1, 2) SCASER4(1, 1, 1, 2) SCASER4(2, 1, 1, 2)
  TORCH_CHECK(false, "moe_gemm_sab: unsupported mode / token groups / stages");
}

template <int MODE, int N, int STAGES, bool PROF, int ABL = 0, bool W0 = false, bool BIG = false, bool TMA = false,
          bool VARN = false>
void launch_wg(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
               torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
               int n_blocks, int grid, long long* prof, int* sched, int pfd) {
  size_t smem = (size_t)STAGES * (WgCfg<N>::STAGE + 32) + 1024;   // + alignment slack
  auto k = moe_gemm_wg_kernel<MODE, N, STAGES, PROF, ABL, W0, BIG, TMA, VARN>;
  TmapBlob tm{};
  if (TMA) tm = act_tmap(a_src, N);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 288, smem);
  grid = std::min(grid, std::max(1, per_sm) * sms);
  k<<<grid, 288, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a_src.data_ptr()), row_route.data_ptr<int32_t>(),
      wq.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), gs.data_ptr<float>(),
      active.data_ptr<int32_t>(), n_active.data_ptr<int32_t>(), exp_start.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, prof, sched, tm, pfd);
}

void moe_gemm_wg(int64_t mode, int64_t n_tok, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                 torch::Tensor sched, int64_t abl, int64_t pfd) {
  int* sp = sched.data_ptr<int32_t>();
  long long* pp = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  if (pp) {
#ifdef OKC_PROF
#define GPCASE(M_, N_, S_, A_) if (mode == M_ && n_tok == N_ && stages == S_ && abl == A_) { launch_wg<M_, N_, S_, true, A_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, (int)pfd); return; }
#define GPCASES(A_) GPCASE(1, 16, 4, A_) GPCASE(2, 16, 4, A_)
    GPCASES(0) GPCASES(3) GPCASES(4) GPCASES(12) GPCASES(20) GPCASES(28) GPCASES(32) GPCASES(96)
#define GPCASE0(M_, N_, S_, A_) if (mode == M_ && n_tok == N_ && stages == 100 + S_ && abl == A_) { launch_wg<M_, N_, S_, true, A_, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, (int)pfd); return; }
    GPCASE0(1, 16, 4, 0) GPCASE0(2, 16, 4, 0) GPCASE0(1, 16, 4, 32) GPCASE0(2, 16, 4, 32) GPCASE0(1, 16, 4, 96) GPCASE0(2, 16, 4, 96)
#define GPCASEB(M_, N_, S_, A_) if (mode == M_ && n_tok == N_ && stages == 200 + S_ && abl == A_) { launch_wg<M_, N_, S_, true, A_, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, (int)pfd); return; }
    GPCASEB(1, 16, 4, 0) GPCASEB(2, 16, 4, 0) GPCASEB(1, 16, 4, 96) GPCASEB(2, 16, 4, 96)
    GPCASEB(1, 32, 4, 0) GPCASEB(2, 32, 4, 0) GPCASEB(1, 32, 4, 96) GPCASEB(2, 32, 4, 96)
    GPCASE(1, 32, 4, 0) GPCASE(2, 32, 4, 0) GPCASE(1, 32, 4, 96) GPCASE(2, 32, 4, 96)
    GPCASE(1, 32, 4, 2) GPCASE(2, 32, 4, 2) GPCASE(1, 32, 4, 3) GPCASE(2, 32, 4, 3)
    GPCASE0(1, 16, 4, 2) GPCASE0(2, 16, 4, 2) GPCASE0(1, 16, 4, 3) GPCASE0(2, 16, 4, 3)
#define GPCASET(M_, N_, S_, A_) if (mode == M_ && n_tok == N_ && stages == 300 + S_ && abl == A_) { launch_wg<M_, N_, S_, true, A_, N_ <= 16, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, (int)pfd); return; }
    GPCASET(1, 16, 4, 0) GPCASET(2, 16, 4, 0) GPCASET(1, 32, 4, 0) GPCASET(2, 32, 4, 0)
    GPCASET(1, 16, 4, 96) GPCASET(2, 16, 4, 96) GPCASET(1, 32, 4, 96) GPCASET(2, 32, 4, 96)
#define GPCASETW(M_, N_, S_, A_) if (mode == M_ && n_tok == N_ && stages == 400 + S_ && abl == A_) { launch_wg<M_, N_, S_, true, A_, true, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, (int)pfd); return; }
    GPCASETW(2, 32, 4, 0) GPCASETW(2, 32, 4, 96)
#endif
    TORCH_CHECK(false, "moe_gemm_wg: no profiling build for this configuration (needs -DOKC_PROF)");
  }
#define GCASE(M_, N_, S_) if (mode == M_ && n_tok == N_ && stages == S_) { launch_wg<M_, N_, S_, false>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASE(1, 8, 4) GCASE(2, 8, 4) GCASE(1, 16, 4) GCASE(2, 16, 4) GCASE(1, 8, 8) GCASE(2, 8, 8) GCASE(1, 16, 8) GCASE(2, 16, 8)
  // stages + 100: the W0 (single A buffer, 4 CTAs/SM) variant with that many stages.
#define GCASE0(M_, N_, S_) if (mode == M_ && n_tok == N_ && stages == 100 + S_) { launch_wg<M_, N_, S_, false, 0, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASE0(1, 8, 4) GCASE0(2, 8, 4) GCASE0(1, 16, 4) GCASE0(2, 16, 4)
  // stages + 200: BIG (one wgmma batch per chunk).
#define GCASEB(M_, N_, S_) if (mode == M_ && n_tok == N_ && stages == 200 + S_) { launch_wg<M_, N_, S_, false, 0, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASEB(1, 8, 4) GCASEB(2, 8, 4) GCASEB(1, 16, 4) GCASEB(2, 16, 4) GCASEB(1, 32, 4) GCASEB(2, 32, 4)
  GCASE(1, 32, 4) GCASE(2, 32, 4)
  // stages + 300: TMA token tiles (N = 16: the W0 4-CTA form; N = 32: double-buffered, 2 CTAs)
#define GCASET(M_, N_, S_) if (mode == M_ && n_tok == N_ && stages == 300 + S_) { launch_wg<M_, N_, S_, false, 0, N_ <= 16 || N_ == 64, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASET(1, 8, 4) GCASET(2, 8, 4) GCASET(1, 16, 4) GCASET(2, 16, 4) GCASET(1, 32, 4) GCASET(2, 32, 4)
  GCASET(1, 64, 4) GCASET(2, 64, 4)
  // 500 + S: TMA, 64-row chunks, wgmma width per chunk (16 / 32 / 64 live rows)
#define GCASEV(M_, S_) if (mode == M_ && n_tok == 64 && stages == 500 + S_) { launch_wg<M_, 64, S_, false, 0, true, false, true, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASEV(1, 4) GCASEV(2, 4)
  // stages + 400: TMA with the single-buffer W0 form at 32 tokens (GEMM2: 3 CTAs/SM, 71 registers;
  // GEMM1 would serialize its wgmmas at 72: ptxas C7512, so it is not instantiated).
#define GCASETW(M_, N_, S_) if (mode == M_ && n_tok == N_ && stages == 400 + S_) { launch_wg<M_, N_, S_, false, 0, true, false, true>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, (int)pfd); return; }
  GCASETW(2, 32, 4)
  TORCH_CHECK(false, "moe_gemm_wg: unsupported mode / tokens / stages");
}

void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid) {
#define CASE(M_, T_, S_) if (mode == M_ && mt == T_ && stages == S_) { launch<M_, T_, S_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid); return; }
  CASE(1, 1, 4) CASE(1, 2, 4) CASE(1, 4, 4) CASE(2, 1, 4) CASE(2, 2, 4) CASE(2, 4, 4)
  CASE(1, 1, 8) CASE(2, 1, 8) CASE(1, 2, 8) CASE(2, 2, 8) CASE(1, 4, 8) CASE(2, 4, 8)
  TORCH_CHECK(false, "moe_gemm: unsupported mode / m tiles / stages");
}
void moe_tiny(torch::Tensor x, torch::Tensor row_route, torch::Tensor exp_start, torch::Tensor w13q, torch::Tensor w13s,
              torch::Tensor gs13, torch::Tensor w2q, torch::Tensor w2s, torch::Tensor gs2, torch::Tensor inter,
              torch::Tensor part1, torch::Tensor part2, torch::Tensor cnt, int64_t flags) {
  // 0x4000: 2 CTAs per SM (6-stage ring, GEMM2 weights from L2): twice the consumer warps per SM.
  auto k = (flags & 0x4000) ? moe_tiny_kernel<6, false> : (flags & 0x1000) ? moe_tiny_kernel<16, false>
         : (flags & 0x2000) ? moe_tiny_kernel<8, false> : moe_tiny_kernel<4, true>;
  const int smem = (flags & 0x4000) ? TinyCfg<6, false>::SMEM : (flags & 0x1000) ? TinyCfg<16, false>::SMEM
                 : (flags & 0x2000) ? TinyCfg<8, false>::SMEM : TinyCfg<4, true>::SMEM;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((flags & 0x4000) ? 2 * sms : sms);   // all CTAs resident (the dependency waits need it)
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (flags & 128) ? 1 : 0;
  cudaLaunchKernelEx(&cfg, k,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (const int32_t*)row_route.data_ptr<int32_t>(),
      (const int32_t*)exp_start.data_ptr<int32_t>(), (const uint8_t*)w13q.data_ptr<uint8_t>(),
      (const uint8_t*)w13s.data_ptr<uint8_t>(), (const float*)gs13.data_ptr<float>(),
      (const uint8_t*)w2q.data_ptr<uint8_t>(), (const uint8_t*)w2s.data_ptr<uint8_t>(), (const float*)gs2.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()), part1.data_ptr<float>(), part2.data_ptr<float>(),
      (int)part1.size(1), cnt.data_ptr<int32_t>(), (int)flags);
}

"""

MOE_CUDA_SRC = MOE_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MOE_DEVICE_SRC, 1)

MOE_CUDA_CPP = """
void moe_tiny(torch::Tensor x, torch::Tensor row_route, torch::Tensor exp_start, torch::Tensor w13q, torch::Tensor w13s, torch::Tensor gs13, torch::Tensor w2q, torch::Tensor w2s, torch::Tensor gs2, torch::Tensor inter, torch::Tensor part1, torch::Tensor part2, torch::Tensor cnt, int64_t flags);
void moe_gemm_wg(int64_t mode, int64_t n_tok, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                 torch::Tensor sched, int64_t abl, int64_t pfd);
void moe_gemm_sab(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                  torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                  torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                  torch::Tensor sched, int64_t abl, int64_t kparts, torch::Tensor part, int64_t sk, torch::Tensor pf_w,
                  torch::Tensor pf_s, torch::Tensor counters, torch::Tensor post);
void moe_gemm_ws(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, int64_t ks,
                 torch::Tensor prof, int64_t abl, torch::Tensor sched);
void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid);
"""

_MOE_MODULES = {}   # (prof, variant) -> extension


def moe_cuda_module(prof=False, variant=""):
    """The extension; prof=True: a separate build that also holds the profiling / ablation
    instantiations (okc/moe_cuda/sim_src.py only; production never compiles them). variant:
    extra compile-time switches for A/B tests ("a1": OKC_ARRIVE1)."""
    key = (bool(prof), variant)
    if key not in _MOE_MODULES:
        from torch.utils.cpp_extension import load_inline
        flags = (["-DOKC_PROF"] if prof else []) + (["-DOKC_ARRIVE1"] if "a1" in variant else [])
        _MOE_MODULES[key] = load_inline(
            name="okc_moe_gemm" + ("_prof" if prof else "") + (f"_{variant}" if variant else ""),
            cpp_sources=MOE_CUDA_CPP, cuda_sources=MOE_CUDA_SRC,
            functions=["moe_gemm", "moe_gemm_ws", "moe_gemm_sab", "moe_gemm_wg", "moe_tiny"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + flags, verbose=False)
    return _MOE_MODULES[key]


# ---------------------------------------------------------------------------------------
# Repacking (prepare, once)

# Source bit, in a lane's word, of code bit j (0-2 magnitude m0-m2, 3 sign) of element e of
# operand o (o = 2 * (step % 2) + b register); matches unpack8 (target bit t <- source).
_TARGETS = (6, 7, 8, 15)
_LEFTOVER = {6: 3, 7: 4, 8: 0, 15: 1, 22: 19, 23: 20, 24: 16, 31: 17}


def _bit_source(o, e, j):
    t = _TARGETS[j] + 16 * e
    return (t, (t - 10) % 32, (t - 13) % 32, _LEFTOVER[t])[o]


def _lane_gather_index(device):
    """[32 lanes, 4 words, 32 bits] -> (row in group, k in chunk, code bit, bit position)."""
    rows, ks, js, pos = [], [], [], []
    for lane in range(32):
        r, c = lane // 4, (lane % 4) * 2
        for w in range(4):
            entries = []
            for o in range(4):
                half, bi = o // 2, o % 2
                for e in range(2):
                    for j in range(4):
                        entries.append((_bit_source(o, e, j), r, (2 * w + half) * 16 + c + 8 * bi + e, j))
            assert sorted(x[0] for x in entries) == list(range(32))
            for b, row, k, j in entries:
                rows.append(row); ks.append(k); js.append(j); pos.append(b)
    t = lambda v: torch.tensor(v, dtype=torch.int64, device=device).view(32, 4, 32)
    return t(rows), t(ks), t(js), t(pos)


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
    rows, ks, js, pos = _lane_gather_index(device)
    c = codes.view(groups, 8, 32, 128).permute(0, 2, 1, 3)                         # [g, chunk, row, k]
    words = torch.empty((groups, 32, 32, 4), dtype=torch.int64, device=device)
    for g0 in range(0, groups, 128):
        picked = c[g0:g0 + 128, :, rows, ks].to(torch.int64)                       # [g, chunk, 32, 4, 32]
        words[g0:g0 + 128] = (((picked >> js) & 1) << pos).sum(-1)                 # [g, chunk, 32, 4]
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    ws = sf.view(torch.uint8).view(groups, 8, 32, 8).permute(0, 2, 1, 3).contiguous()  # [g, chunk, row, step]
    return words.contiguous(), ws


def scales_supported(sf):
    """The scale decode (PRMT + IMAD exponent add) is exact for non-negative normal E4M3
    scales: sign 0, exponent field 1..15. Anything else keeps Marlin."""
    b = sf.view(torch.uint8)
    return bool(((b & 0x80) == 0).all() and ((b >> 3) > 0).all())


# ---------------------------------------------------------------------------------------
# Routing and the post-expert norm (Triton)


@triton.jit
def _route_compact_kernel(ids_ptr, local_start_ptr, row_route_ptr, pos_of_route_ptr, exp_start_ptr,
                          active_ptr, n_active_ptr, P, ROWS, NE: tl.constexpr, E: tl.constexpr, CHUNK: tl.constexpr,
                          SPLIT: tl.constexpr = 0, ROWS2: tl.constexpr = 64, PDL: tl.constexpr = False):
    """One program per local expert e: places e's routes (in route order) at rows
    exp_start[e] .. exp_start[e+1]; row_route[row] = route p (= token * 8 + slot) and
    pos_of_route[p] = row (-1 for non-local routes, written by program 0). Program 0 also
    writes exp_start [E + 1] and the list of non-empty chunks of ROWS rows, e | (chunk << 8).
    SPLIT > 0: experts with more than SPLIT routes are listed separately, in chunks of ROWS2
    rows, at active_ptr + E + P (count at n_active_ptr + 1).
    CHUNK >= P (a power of two): all route ids in one load, counts by histogram, ranks by one
    cumsum (a loop over 256-id chunks cost ~0.6 us of dependent load latency per pass: 10 us
    at bs 256).
    PDL (chain): GEMM1, a programmatic dependent that reads nothing before its own wait, may
    launch at once."""
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    e = tl.program_id(0)
    experts = tl.arange(0, 2 * E)   # histogram bins: E local experts + "not local" (bin E)
    p = tl.arange(0, CHUNK)
    in_range = p < P
    local_start = tl.load(local_start_ptr)
    loc = (tl.load(ids_ptr + p, mask=in_range, other=0) - local_start + NE) % NE
    loc = tl.where(in_range & (loc < E), loc, E)
    counts = tl.where(experts < E, tl.histogram(loc, 2 * E), 0)
    ends = tl.cumsum(counts, axis=0)
    starts = ends - counts
    my_start = tl.sum(tl.where(experts == e, starts, 0), axis=0)
    mine = loc == e
    rank = tl.cumsum(mine.to(tl.int32), axis=0) - 1
    tl.store(row_route_ptr + my_start + rank, p, mask=mine)
    tl.store(pos_of_route_ptr + p, my_start + rank, mask=mine)
    if e == 0:
        tl.store(pos_of_route_ptr + p, -1, mask=in_range & (loc >= E))
        tl.store(exp_start_ptr + experts, starts, mask=experts < E)
        tl.store(exp_start_ptr + E, tl.sum(counts, axis=0))
        big = counts > SPLIT if SPLIT > 0 else counts < 0
        nch = tl.where(big, 0, (counts + ROWS - 1) // ROWS)
        first = tl.cumsum(nch, axis=0) - nch
        for j in range(0, tl.max(nch, axis=0)):
            tl.store(active_ptr + first + j, experts | (j << 8), mask=j < nch)
        tl.store(n_active_ptr, tl.sum(nch, axis=0))
        if SPLIT > 0:
            nch2 = tl.where(big, (counts + ROWS2 - 1) // ROWS2, 0)
            first2 = tl.cumsum(nch2, axis=0) - nch2
            for j in range(0, tl.max(nch2, axis=0)):
                tl.store(active_ptr + E + P + first2 + j, experts | (j << 8), mask=j < nch2)
            tl.store(n_active_ptr + 1, tl.sum(nch2, axis=0))


@triton.jit
def _gather_rows_kernel(x_ptr, row_route_ptr, exp_start_ptr, out_ptr, E: tl.constexpr, ROWS: tl.constexpr,
                        H: tl.constexpr):
    """out[r] = x[row_route[r] // 8] for the local rows r < exp_start[E] (expert order), so
    GEMM1's token tile is contiguous rows, loadable by one TMA box."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = rows < tl.load(exp_start_ptr + E)
    src = tl.load(row_route_ptr + rows, mask=live, other=0) // 8
    cols = tl.arange(0, H)
    v = tl.load(x_ptr + src.to(tl.int64)[:, None] * H + cols[None, :], mask=live[:, None], other=0.0)
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * H + cols[None, :], v, mask=live[:, None])


@triton.jit
def _splitk_silu_kernel(part_ptr, exp_start_ptr, inter_ptr, P, E: tl.constexpr, KP: tl.constexpr, H: tl.constexpr,
                        PDL: tl.constexpr = False):
    """GEMM1 split K: inter[row] = silu(sum_k gate parts) * sum_k up parts, parts summed in order.
    PDL (chain): signal GEMM2 (whose pre-wait code reads only the route's output and weights),
    then wait for GEMM1's parts."""
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    if row < tl.load(exp_start_ptr + E):
        cols = tl.arange(0, H)
        g = tl.zeros((H,), tl.float32)
        u = tl.zeros((H,), tl.float32)
        for k in tl.static_range(KP):
            base = part_ptr + (k * P + row).to(tl.int64) * (2 * H)
            g += tl.load(base + cols)
            u += tl.load(base + H + cols)
        tl.store(inter_ptr + row.to(tl.int64) * H + cols, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16))


@triton.jit
def _post_norm_parts_kernel(part_ptr, w_ptr, s_ptr, pos_ptr, out_ptr, eps, P, KP: tl.constexpr, TOPK: tl.constexpr,
                            H: tl.constexpr, PDL: tl.constexpr = False):
    """_post_norm_rows_kernel reading GEMM2's split-K fp32 parts (summed in order) for y."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    ks = tl.arange(0, TOPK)
    rows = tl.load(pos_ptr + t * TOPK + ks)
    scales = tl.load(s_ptr + t * TOPK + ks)
    w = tl.load(w_ptr + offs).to(tl.float32)
    if PDL:   # chain: the route's output and the weights above are complete; the parts wait for GEMM2
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):   # branch-free: every slot's loads can be issued up front
        row = tl.sum(tl.where(ks == k, rows, 0), axis=0)
        live = row >= 0
        y = tl.zeros((H,), tl.float32)
        for kp in tl.static_range(KP):
            y += tl.load(part_ptr + (kp * P + tl.maximum(row, 0)).to(tl.int64) * H + offs, mask=(offs < H) & live,
                         other=0.0)
        y = y.to(tl.bfloat16).to(tl.float32) * w
        inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
        acc += y * (inv * tl.where(live, tl.sum(tl.where(ks == k, scales, 0.0), axis=0), 0.0))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _post_norm_rows_kernel(y_ptr, w_ptr, s_ptr, pos_ptr, out_ptr, eps, TOPK: tl.constexpr, H: tl.constexpr,
                           PDL: tl.constexpr = False):
    """out[t] = sum over slots k with a local row of s[t, k] * RMSNorm(y[row] * w). Branch-free
    (slot positions loaded as one vector, row loads masked) so the row loads overlap: a branch
    per slot serialized ~2 dependent loads per local route (~4 us at bs 1)."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    ks = tl.arange(0, TOPK)
    rows = tl.load(pos_ptr + t * TOPK + ks)
    scales = tl.load(s_ptr + t * TOPK + ks)
    w = tl.load(w_ptr + offs).to(tl.float32)
    if PDL:   # chain: the route's output and the weights above are complete; y waits for GEMM2
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.sum(tl.where(ks == k, rows, 0), axis=0)
        live = row >= 0
        y = tl.load(y_ptr + tl.maximum(row, 0).to(tl.int64) * H + offs, mask=(offs < H) & live, other=0.0).to(tl.float32) * w
        inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
        acc += y * (inv * tl.where(live, tl.sum(tl.where(ks == k, scales, 0.0), axis=0), 0.0))
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


def ws_layout(bank):
    """Adds the warp-specialized kernel's layout: each CTA's weight and scale bytes for one
    128-k chunk contiguous ([expert][n block][chunk][slot][...]), 16 slots (GEMM1) / 8 (GEMM2)."""
    for name, gpc in (("w13", 16), ("w2", 8)):
        q, s = bank[name + "q"], bank[name + "s"]
        e, g = q.shape[:2]
        bank[name + "q_ws"] = q.view(e, g // gpc, gpc, 32, 32, 4).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8)
        bank[name + "s_ws"] = s.view(e, g // gpc, gpc, 32, 8, 8).permute(0, 1, 3, 2, 4, 5).contiguous()
    # v6 (swap-AB): 16 slots per CTA in GEMM2 as well.
    q, s = bank["w2q"], bank["w2s"]
    e, g = q.shape[:2]
    bank["w2q_sab"] = q.view(e, g // 16, 16, 32, 32, 4).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8)
    bank["w2s_sab"] = s.view(e, g // 16, 16, 32, 8, 8).permute(0, 1, 3, 2, 4, 5).contiguous()
    # Work counters of the two GEMMs' dynamic schedulers (each launch leaves them at zero).
    bank["sched"] = torch.zeros(8, dtype=torch.int32, device=q.device)   # GEMM1 / GEMM2 x (main, dual big)
    # Per-item arrival counters of the fused split-K epilogue (each item's last CTA resets its own).
    bank["cnt"] = torch.zeros(1 << 16, dtype=torch.int32, device=q.device)
    return bank


def moe_cuda_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, mt=1, ctas_per_sm=2, topk=8, stages=4,
                    ws=False, ks=1, prof=None, abl=0, sab=False, wg=False, stages2=None, kparts=1, pfd=0, sk=False, dual=0, variant="", pdl=False, kparts2=None, l2pf=False,
                    fuse=False, dualv8=0, dualsab=0, mt2=2, stages_big=402, ctas2=2, mpf=False, chain=False, rb=1, l2d=0, tiny=False, _tiny_stamps=False, apost=False):
    """The pipeline as four separately callable stages (route, gemm1, gemm2, post) sharing
    buffers, for per-stage timing; moe_cuda_forward runs them in order.
    dual (v7 only): experts with more than `dual` routes go to a second launch per GEMM, the
    64-row-chunk kernel whose wgmma width follows each chunk's rows (stages 504, 2 CTAs/SM);
    the rest stay on the configured kernel.
    dualsab (v6 only): experts with more than `dualsab` routes run first on a v6 launch with
    `mt2` 8-token groups per item (stages `stages_big`), so a hot expert's weights are decoded
    once for up to 8 mt2 tokens instead of once per 8; the narrow launch over the rest is its PDL
    co-runner (judge profile: a 16-token expert on 8-token tiles cost ~10 us of GEMM1 at bs 16).
    dualv8 (v6 only): experts with more than `dualv8` routes run on v8 (64-row chunks, wgmma
    width per chunk) first; the v6 launch over the rest is its PDL co-runner, taking SM slots
    as v8's CTAs retire (small experts keep mma.sync's N 8 / 16 instead of wgmma's N 16).
    chain (v6 only): every launch after the route is a programmatic dependent (PDL) of the one
    before and each signals its dependents at its start (flag 128): launches, CTA setup and
    GEMM2's first weight stages overlap the previous kernel; each waits only before reading the
    previous kernel's output."""
    if tiny:   # bs <= 8: one persistent grid for both GEMMs (moe_tiny_kernel)
        return moe_tiny_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, chain=chain, topk=topk,
                               tiny_ring=tiny if tiny is not True else 4, stamps=_tiny_stamps)
    assert not chain or (sab and not (wg or dualv8 or dualsab or sk)), "chain: v6 single-launch configs"
    mod = moe_cuda_module(prof is not None, variant)
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
    active = torch.empty((2 * count + 2 * p + 1,), dtype=torch.int32, device=device)
    n_active = torch.empty((2,), dtype=torch.int32, device=device)
    active2, n_active2 = active[count + p:], n_active[1:2]   # dual: the big experts' 64-row chunks
    inter = torch.empty((p, 4096), dtype=torch.bfloat16, device=device)
    y = torch.empty((p, 4096), dtype=torch.bfloat16, device=device)
    out = torch.empty((m, hidden), dtype=x.dtype, device=device)
    grid = 132 * ctas_per_sm

    tma = (wg and stages >= 300) or (sab and dualv8 > 0)
    xs = torch.empty((p, 4096), dtype=torch.bfloat16, device=device) if tma else x

    def route():
        _route_compact_kernel[(count,)](ids, local_start, row_route, pos_of_route, exp_start, active, n_active, p, rows_per_chunk,
                                        NE=256, E=count, CHUNK=max(256, triton.next_power_of_2(p)),
                                        SPLIT=dual if wg else (dualv8 or dualsab), ROWS2=8 * mt2 if dualsab else 64,
                                        PDL=chain, num_warps=4)
        if tma:
            _gather_rows_kernel[(triton.cdiv(p, 4),)](x, row_route, exp_start, xs, E=count, ROWS=4, H=4096, num_warps=4)

    # prof: None, or (prof1, prof2) int64 buffers [max items * 8] for the profiling build.
    no_prof = torch.empty((0,), dtype=torch.int64, device=device)
    no_part = torch.empty((0, 0, 0), dtype=torch.float32, device=device)
    no_u8 = torch.empty((0,), dtype=torch.uint8, device=device)
    no_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    prof = prof or (no_prof, no_prof)
    sched = bank.get("sched")
    gemm_ws = lambda which: (lambda *a: mod.moe_gemm_ws(*a, ks, prof[which], abl, sched[2 * which:2 * which + 2]))
    g1 = gemm_ws(0) if ws else mod.moe_gemm
    g2 = gemm_ws(1) if ws else mod.moe_gemm
    rows_per_chunk = 8 * mt if (sab or wg) else 16 * mt
    sfx = "_ws" if ws else ""

    # kparts2: GEMM2's own K split (default: kparts). GEMM2 has half GEMM1's items, so at small
    # batches it runs too few concurrent weight streams to fill HBM (one CTA streams ~10-15 GB/s).
    kparts2 = kparts if kparts2 is None else kparts2
    split = sab and kparts > 1
    split2 = sab and kparts2 > 1
    pf64 = (64 if mpf else 0) | (l2d << 8)   # v6 producer: mpf next-item metadata; l2d L2 prefetch distance (stages)
    # rb 2 (stages 1000 + 100 * SUB + S): two 128-row blocks per v6 item, 32 rows per consumer warp
    assert rb == 1 or (sab and stages >= 1000 * (rb // 2) and kparts == 1 and (kparts2 or 1) == 1 and not sk and not l2pf
                       and not mpf and not dualv8 and not dualsab), "rb 2: plain v6 path only"
    ch1, ch2 = (128, 2 | 128) if chain else (0, 0)   # chain: GEMM1 waits at its start; GEMM2 as flag 2
    pdl_kw = dict(launch_pdl=True) if chain else {}
    fuse1 = fuse and split and not sk   # GEMM1's split parts reduced by their last CTA (no reduction kernel)
    part1 = torch.empty((kparts, p, 8192), dtype=torch.float32, device=device) if split else no_part
    part2 = torch.empty((kparts2, p, 4096), dtype=torch.float32, device=device) if split2 and not apost else no_part
    # apost: GEMM2's partials go to y32 by red.add and its last items do the post-norm (PostArgs)
    no_post = torch.empty((0,), dtype=torch.int64)
    post1 = post2 = no_post
    if apost:
        assert sab and not (dualv8 or dualsab or sk), "apost: v6 single-launch path"
        import struct
        y32 = torch.empty((p, 4096), dtype=torch.float32, device=device)
        out32 = torch.empty((m, hidden), dtype=torch.float32, device=device)
        tok_cnt = bank.setdefault("apost_tok", torch.zeros((1 << 12,), dtype=torch.int32, device=device))
        chunk_cnt = bank.setdefault("apost_chunk", torch.zeros((1 << 16,), dtype=torch.int32, device=device))
        eps_bits = struct.unpack("<i", struct.pack("<f", float(eps)))[0]
        base = [y32.data_ptr(), out32.data_ptr(), out.data_ptr(), norm_w.data_ptr(), topk_weights.data_ptr(),
                pos_of_route.data_ptr(), tok_cnt.data_ptr(), chunk_cnt.data_ptr(), eps_bits, m, p]
        post1 = torch.tensor(base + [1], dtype=torch.int64)
        post2 = torch.tensor(base + [2], dtype=torch.int64)

    def gemm1():
        if wg:    # v7: mt = 8-token groups per item (N = 8 mt)
            if dual:
                mod.moe_gemm_wg(1, 64, 504, xs, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active2,
                                n_active2, exp_start, inter, 64, 132 * 2, prof[0], sched[4:6], abl, pfd)
            mod.moe_gemm_wg(1, 8 * mt, stages, xs, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active,
                            n_active[0:1], exp_start, inter, 64, grid, prof[0], sched[0:2], abl, pfd)
            return
        if sab:   # 64 n blocks (GEMM1: 8 gate/up pairs per CTA), 32 for GEMM2 (16 n8 groups per CTA)
            if dualv8:
                mod.moe_gemm_wg(1, 64, 504, xs, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active2,
                                n_active2, exp_start, inter, 64, 132 * 2, prof[0], sched[4:6], abl, pfd)
            if dualsab:
                mod.moe_gemm_sab(1, mt2, stages_big, x, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"],
                                 active2, n_active2, exp_start, inter, 64, 132 * ctas2, prof[0], sched[4:6], abl, 1,
                                 no_part, 32 | pf64, no_u8, no_u8, no_i32, no_post)
            mod.moe_gemm_sab(1, mt, stages, x, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active,
                             n_active[0:1], exp_start, inter, 64 // rb, grid, prof[0], sched[0:2], abl, kparts, part1,
                             int(sk) | (4 if l2pf else 0) | (8 if fuse1 else 0) | (16 if (dualv8 or dualsab) else 0) | pf64
                             | ch1,
                             bank["w2q_sab"] if l2pf else no_u8,
                             bank["w2s_sab"] if l2pf else no_u8, bank["cnt"] if fuse1 else no_i32, post1)
            if split and not fuse1:
                _splitk_silu_kernel[(p,)](part1, exp_start, inter, p, E=count, KP=kparts, H=4096, PDL=chain,
                                          num_warps=8, **pdl_kw)
            return
        # 64 work items per active expert per GEMM: 8 warps x (MODE 1: a gate/up pair, MODE 2: an n8 group).
        g1(1, mt, stages, x, row_route, bank["w13q" + sfx], bank["w13s" + sfx], bank["w13_gs"], active, n_active,
             exp_start, inter, 512 // 8, grid)

    def gemm2():
        if wg:
            if dual:
                mod.moe_gemm_wg(2, 64, 504, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active2,
                                n_active2, exp_start, y, 32, 132 * 2, prof[1], sched[6:8], abl, pfd)
            mod.moe_gemm_wg(2, 8 * mt, stages2 or stages, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active,
                            n_active[0:1], exp_start, y, 32, grid, prof[1], sched[2:4], abl, pfd)
            return
        if sab:
            if dualv8:
                mod.moe_gemm_wg(2, 64, 504, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active2,
                                n_active2, exp_start, y, 32, 132 * 2, prof[1], sched[6:8], abl, pfd)
            if dualsab:
                mod.moe_gemm_sab(2, mt2, stages_big, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"],
                                 active2, n_active2, exp_start, y, 32, 132 * ctas2, prof[1], sched[6:8], abl, 1, no_part,
                                 32 | pf64, no_u8, no_u8, no_i32, no_post)
            mod.moe_gemm_sab(2, mt, stages, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active,
                             n_active[0:1], exp_start, y, 32 // rb, grid, prof[1], sched[2:4], abl, kparts2, part2,
                             int(sk) | (2 if pdl else 0) | (16 if (dualv8 or dualsab) else 0) | pf64 | ch2, no_u8, no_u8,
                             no_i32, post2)
            return
        g2(2, mt, stages, inter, row_route, bank["w2q" + sfx], bank["w2s" + sfx], bank["w2_gs"], active, n_active,
             exp_start, y, 512 // 8, grid)

    def post():
        if apost:   # done by GEMM2's last items
            return
        if split2:
            _post_norm_parts_kernel[(m,)](part2, norm_w, topk_weights, pos_of_route, out, eps, p, KP=kparts2, TOPK=topk,
                                          H=hidden, PDL=chain, num_warps=8, **pdl_kw)
            return
        _post_norm_rows_kernel[(m,)](y, norm_w, topk_weights, pos_of_route, out, eps, TOPK=topk, H=hidden, PDL=chain,
                                     num_warps=8, **pdl_kw)

    return (route, gemm1, gemm2, post), out


def moe_tiny_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, chain=True, topk=8, tiny_ring=4, stamps=False, **_):
    """Tiny batches (<= 8 rows per local expert, i.e. bs <= 8): route, then ONE persistent grid for
    both expert GEMMs (moe_tiny_kernel: GEMM2's first unit per CTA resident in shared memory from
    the start), then the parts post-norm (KP 2). Same numerics as the split-K v6 path."""
    m, hidden = x.shape
    assert m <= 8, "moe_tiny: <= 8 tokens"
    mod = moe_cuda_module()
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    p = m * topk
    count = bank["count"]
    device = x.device
    row_route = torch.empty((p,), dtype=torch.int32, device=device)
    pos_of_route = torch.empty((p,), dtype=torch.int32, device=device)
    exp_start = torch.empty((count + 1,), dtype=torch.int32, device=device)
    active = torch.empty((2 * count + 2 * p + 1,), dtype=torch.int32, device=device)
    n_active = torch.empty((2,), dtype=torch.int32, device=device)
    inter = torch.empty((p, 4096), dtype=torch.bfloat16, device=device)
    part1 = torch.empty((2, p, 8192), dtype=torch.float32, device=device)
    part2 = torch.empty((2, p, 4096), dtype=torch.float32, device=device)
    out = torch.empty((m, hidden), dtype=x.dtype, device=device)
    cnt = bank.setdefault("tiny_cnt", torch.zeros((4096 + 2 * 4 * 512,), dtype=torch.int32, device=device))

    def route():
        _route_compact_kernel[(count,)](ids, local_start, row_route, pos_of_route, exp_start, active, n_active, p, 8,
                                        NE=256, E=count, CHUNK=max(256, triton.next_power_of_2(p)), SPLIT=0, ROWS2=64,
                                        PDL=chain, num_warps=4)

    def tiny():
        mod.moe_tiny(x, row_route, exp_start, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], bank["w2q_sab"],
                     bank["w2s_sab"], bank["w2_gs"], inter, part1, part2, cnt,
                     (128 if chain else 0) | {4: 0, 16: 0x1000, 8: 0x2000, 6: 0x4000}[tiny_ring] | (0x8000 if stamps else 0))

    def post():
        _post_norm_parts_kernel[(m,)](part2, norm_w, topk_weights, pos_of_route, out, eps, p, KP=2, TOPK=topk, H=4096,
                                      PDL=chain, num_warps=8, **(dict(launch_pdl=True) if chain else {}))

    return [route, tiny, post], out


def moe_cuda_forward(x, ids, topk_weights, local_start, bank, norm_w, eps, **kw):
    """x [M, 4096] BF16, ids [M, 8] global expert ids, topk_weights [M, 8] FP32, local_start
    int32 [1] (device) -> [M, 4096] BF16: this GPU's experts' weighted, normed sum."""
    fns, out = moe_cuda_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, **kw)
    for fn in fns:
        fn()
    return out
