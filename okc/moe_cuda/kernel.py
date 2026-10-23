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

template <int MT>
struct SabCfg {
  static constexpr int A_ROW = 272;
  static constexpr int A_STAGE = MT * 8 * A_ROW;
  static constexpr int W_STAGE = 16 * 512;
  static constexpr int S_STAGE = 16 * 64;
  static constexpr int STAGE = A_STAGE + W_STAGE + S_STAGE;
};

template <int MODE, int MT, int STAGES, bool PROF, int ABL = 0>
__global__ void __launch_bounds__(288, MT == 1 ? 4 : 3) moe_gemm_sab_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks,
    long long* __restrict__ prof, int* __restrict__ sched, int kparts, float* __restrict__ part, int part_rows,
    int sk) {
  // Split K (kparts > 1, small batches: too few items for the SMs): item = (chunk entry,
  // n block, k part); each part streams NCHUNK / kparts chunks and writes its scaled fp32
  // partial to part[k part][row][column] (GEMM1: gate at n, up at 4096 + n); the parts are
  // summed in a fixed order afterwards (_splitk_silu_kernel, _post_norm_parts_kernel), so the
  // result stays bitwise repeatable.
  constexpr int abl = PROF ? ABL : 0;   // ablation bits as moe_gemm_ws_kernel's
  using C = SabCfg<MT>;
  constexpr int NCHUNK = KTOT / 128;
  static_assert(NCHUNK % STAGES == 0 && MT * 8 <= 32, "stages divide chunks; <= 32 token rows");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + STAGES * C::STAGE, bar_empty = bar_full + 8 * STAGES;
  volatile int* hdr = reinterpret_cast<volatile int*>(smem + STAGES * (C::STAGE + 16));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, 256);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
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
    // Producer (as v5): first item static, then the global counter; one end marker.
    uint32_t f = 0;
    int u = s0;
    for (bool first = true;; first = false) {
      int item = blockIdx.x, kb = 0, ke = nch;
      bool done;
      if (sk) {
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
        return;
      }
      const int it = sk ? item : item / kparts;
      const int nb = it % n_blocks;
      const int ch = active[it / n_blocks];
      const int e = ch & 255, mc = ch >> 8;
      const int row0 = exp_start[e] + mc * MT * 8;
      const int live = min(MT * 8, exp_start[e + 1] - row0);
      int src = 0;
      if (lane < live) src = MODE == 1 ? row_route[row0 + lane] / 8 : row0 + lane;
      const uint8_t* a_row = reinterpret_cast<const uint8_t*>(a_src + (size_t)src * KTOT);
      const uint8_t* w_blk = wq + ((size_t)e * n_blocks + nb) * NCHUNK * C::W_STAGE;
      const uint8_t* s_blk = ws + ((size_t)e * n_blocks + nb) * NCHUNK * C::S_STAGE;
      const uint32_t tx = (abl & 4) ? 0u : ((abl & 2) ? 0u : live * 256) + C::W_STAGE + C::S_STAGE;
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
        if (lane == 0) {
          hdr[st] = item;
          mbar_expect_tx(full, tx);
          if (!(abl & 4)) {
            bulk_g2s(sb + C::A_STAGE, w_blk + (size_t)kc * C::W_STAGE, C::W_STAGE, full);
            bulk_g2s(sb + C::A_STAGE + C::W_STAGE, s_blk + (size_t)kc * C::S_STAGE, C::S_STAGE, full);
          }
        }
        __syncwarp();
        if (!(abl & 6) && lane < live) bulk_g2s(sb + lane * C::A_ROW, a_row + kc * 256, 256, full);
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
  const uint32_t s_src = base + C::A_STAGE + C::W_STAGE + (warp * 2) * 64 + r * 8;
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
    // acc[x][t]: token group t, k16 steps of parity x (two chains per group).
    float acc[2][MT][4];
#pragma unroll
    for (int x = 0; x < 2; ++x)
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[x][t][i] = 0.f;

    auto chunks = [&](auto tt) {
      constexpr int T = decltype(tt)::value;
#pragma unroll 1
      for (int kc0 = 0; kc0 < nchi; kc0 += STAGES, f += STAGES) {
        const uint32_t parity = (f / STAGES) & 1;
#pragma unroll
        for (int st = 0; st < STAGES; ++st) {
          mbar_wait(bar_full + 8 * st, parity);
          if (abl & 1) {
            mbar_arrive(bar_empty + 8 * st);
            continue;
          }
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
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            uint32_t v[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) unpack8(w == 0 ? wv[j].x : w == 1 ? wv[j].y : w == 2 ? wv[j].z : wv[j].w, v[j]);
            uint32_t b[T][4];
#pragma unroll
            for (int t = 0; t < T; ++t)
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                           : "=r"(b[t][0]), "=r"(b[t][1]), "=r"(b[t][2]), "=r"(b[t][3])
                           : "r"(ldb[w] + st * C::STAGE + t * 8 * C::A_ROW));
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
                  acc[half][t][0] += __int_as_float((a[0] ^ a[3] ^ b[t][2 * half]) & 0x3f800000u);
                } else {
                  mma16816(acc[half][t], a, b[t][2 * half], b[t][2 * half + 1]);
                }
              }
            }
          }
          mbar_arrive(bar_empty + 8 * st);
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
    if (kparts > 1) {
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
    } else
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= groups) continue;
      float y[2][2];   // [row block 0 / 1 (MODE 2), or gate*up (MODE 1, block 0)][token 2c / 2c+1]
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float lo = (acc[0][t][i] + acc[1][t][i]) * gsc, hi = (acc[0][t][2 + i] + acc[1][t][2 + i]) * gsc;
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
        const int n = MODE == 1 ? 8 * (nb * 8 + warp) + (r & ~1) : 8 * (nb * 16 + 2 * warp + blk) + (r & ~1);
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
template <int MODE, int N, int STAGES, bool PROF, int ABL = 0, bool W0 = false, bool BIG = false, bool TMA = false>
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
      mbar_init(bar_empty + 8 * s, 256);
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
          mbar_arrive(bar_empty + 8 * st);
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
              wgmma_rs<N>(d, a, desc0 + ((uint64_t)off >> 4));
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
          mbar_arrive(bar_empty + 8 * st);   // all of this stage's wgmmas are complete
          continue;
        }
        if (st > 0) mbar_arrive(bar_empty + 8 * (st - 1));
        else if (kc0 > 0) mbar_arrive(bar_empty + 8 * (STAGES - 1));
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (!(abl & 1) && !SINGLE) mbar_arrive(bar_empty + 8 * (STAGES - 1));

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

template <int MODE, int MT, int STAGES, bool PROF, int ABL = 0>
void launch_sab(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
                torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
                int n_blocks, int grid, long long* prof, int* sched, int kparts, float* part, int part_rows, int sk) {
  size_t smem = (size_t)STAGES * (SabCfg<MT>::STAGE + 32);
  auto k = moe_gemm_sab_kernel<MODE, MT, STAGES, PROF, ABL>;
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
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks, prof, sched, kparts, part, part_rows, sk);
}

void moe_gemm_sab(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                  torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                  torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                  torch::Tensor sched, int64_t abl, int64_t kparts, torch::Tensor part, int64_t sk) {
  int* sp = sched.data_ptr<int32_t>();
  float* pt = part.numel() ? part.data_ptr<float>() : nullptr;
  const int prow = part.numel() ? (int)part.size(1) : 0;
  const int kp = (int)kparts;
  TORCH_CHECK(kp == 1 || pt, "moe_gemm_sab: split K needs a partial buffer");
  long long* pp = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  if (pp) {
#ifdef OKC_PROF
#define SPCASE(M_, T_, S_, A_) if (mode == M_ && mt == T_ && stages == S_ && abl == A_) { launch_sab<M_, T_, S_, true, A_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, pp, sp, kp, pt, prow, (int)sk); return; }
#define SPCASES(A_) SPCASE(1, 2, 4, A_) SPCASE(2, 2, 4, A_) SPCASE(1, 1, 4, A_) SPCASE(2, 1, 4, A_)
    SPCASES(0) SPCASES(1) SPCASES(2) SPCASES(3) SPCASES(4) SPCASES(12) SPCASES(20) SPCASES(28)
    SPCASE(1, 1, 8, 0) SPCASE(2, 1, 8, 0)
#endif
    TORCH_CHECK(false, "moe_gemm_sab: no profiling build for this configuration (needs -DOKC_PROF)");
  }
#define SCASE(M_, T_, S_) if (mode == M_ && mt == T_ && stages == S_) { launch_sab<M_, T_, S_, false>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid, nullptr, sp, kp, pt, prow, (int)sk); return; }
  SCASE(1, 1, 4) SCASE(2, 1, 4) SCASE(1, 2, 4) SCASE(2, 2, 4) SCASE(1, 1, 8) SCASE(2, 1, 8) SCASE(1, 2, 8) SCASE(2, 2, 8)
  TORCH_CHECK(false, "moe_gemm_sab: unsupported mode / token groups / stages");
}

template <int MODE, int N, int STAGES, bool PROF, int ABL = 0, bool W0 = false, bool BIG = false, bool TMA = false>
void launch_wg(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
               torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
               int n_blocks, int grid, long long* prof, int* sched, int pfd) {
  size_t smem = (size_t)STAGES * (WgCfg<N>::STAGE + 32) + 1024;   // + alignment slack
  auto k = moe_gemm_wg_kernel<MODE, N, STAGES, PROF, ABL, W0, BIG, TMA>;
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
"""

MOE_CUDA_SRC = MOE_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MOE_DEVICE_SRC, 1)

MOE_CUDA_CPP = """
void moe_gemm_wg(int64_t mode, int64_t n_tok, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                 torch::Tensor sched, int64_t abl, int64_t pfd);
void moe_gemm_sab(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                  torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                  torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, torch::Tensor prof,
                  torch::Tensor sched, int64_t abl, int64_t kparts, torch::Tensor part, int64_t sk);
void moe_gemm_ws(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, int64_t ks,
                 torch::Tensor prof, int64_t abl, torch::Tensor sched);
void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid);
"""

_MOE_MODULE = []


def moe_cuda_module(prof=False):
    """The extension; prof=True: a separate build that also holds the profiling / ablation
    instantiations (okc/moe_cuda/sim_src.py only; production never compiles them)."""
    while len(_MOE_MODULE) < 2:
        _MOE_MODULE.append(None)
    if _MOE_MODULE[prof] is None:
        from torch.utils.cpp_extension import load_inline
        _MOE_MODULE[prof] = load_inline(
            name="okc_moe_gemm_prof" if prof else "okc_moe_gemm", cpp_sources=MOE_CUDA_CPP, cuda_sources=MOE_CUDA_SRC,
            functions=["moe_gemm", "moe_gemm_ws", "moe_gemm_sab", "moe_gemm_wg"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + (["-DOKC_PROF"] if prof else []),
            verbose=False)
    return _MOE_MODULE[prof]


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
                          active_ptr, n_active_ptr, P, ROWS, NE: tl.constexpr, E: tl.constexpr, CHUNK: tl.constexpr):
    """One program per local expert e: places e's routes (in route order) at rows
    exp_start[e] .. exp_start[e+1]; row_route[row] = route p (= token * 8 + slot) and
    pos_of_route[p] = row (-1 for non-local routes, written by program 0). Program 0 also
    writes exp_start [E + 1] and the list of non-empty chunks of ROWS rows, e | (chunk << 8)."""
    e = tl.program_id(0)
    experts = tl.arange(0, E)
    offs = tl.arange(0, CHUNK)
    local_start = tl.load(local_start_ptr)
    counts = tl.zeros((E,), tl.int32)
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        loc = (tl.load(ids_ptr + p, mask=p < P, other=0) - local_start + NE) % NE
        loc = tl.where((p < P) & (loc < E), loc, E)
        counts += tl.sum((loc[:, None] == experts[None, :]).to(tl.int32), axis=0)
    ends = tl.cumsum(counts, axis=0)
    starts = ends - counts
    my_start = tl.sum(tl.where(experts == e, starts, 0), axis=0)
    running = my_start
    for c0 in range(0, P, CHUNK):
        p = c0 + offs
        in_range = p < P
        loc = (tl.load(ids_ptr + p, mask=in_range, other=0) - local_start + NE) % NE
        mine = (loc == e) & in_range
        rank = tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(row_route_ptr + running + rank, p, mask=mine)
        tl.store(pos_of_route_ptr + p, running + rank, mask=mine)
        running += tl.sum(mine.to(tl.int32), axis=0)
        if e == 0:
            tl.store(pos_of_route_ptr + p, -1, mask=in_range & (loc >= E))
    if e == 0:
        tl.store(exp_start_ptr + experts, starts)
        tl.store(exp_start_ptr + E, tl.sum(counts, axis=0))
        nch = (counts + ROWS - 1) // ROWS
        first = tl.cumsum(nch, axis=0) - nch
        for j in range(0, tl.max(nch, axis=0)):
            tl.store(active_ptr + first + j, experts | (j << 8), mask=j < nch)
        tl.store(n_active_ptr, tl.sum(nch, axis=0))


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
def _splitk_silu_kernel(part_ptr, exp_start_ptr, inter_ptr, P, E: tl.constexpr, KP: tl.constexpr, H: tl.constexpr):
    """GEMM1 split K: inter[row] = silu(sum_k gate parts) * sum_k up parts, parts summed in order."""
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
                            H: tl.constexpr):
    """_post_norm_rows_kernel reading GEMM2's split-K fp32 parts (summed in order) for y."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.load(pos_ptr + t * TOPK + k)
        if row >= 0:
            y = tl.zeros((H,), tl.float32)
            for kp in tl.static_range(KP):
                y += tl.load(part_ptr + (kp * P + row).to(tl.int64) * H + offs)
            y = y.to(tl.bfloat16).to(tl.float32) * w
            inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (inv * tl.load(s_ptr + t * TOPK + k))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _post_norm_rows_kernel(y_ptr, w_ptr, s_ptr, pos_ptr, out_ptr, eps, TOPK: tl.constexpr, H: tl.constexpr):
    """out[t] = sum over slots k with a local row of s[t, k] * RMSNorm(y[row] * w)."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.load(pos_ptr + t * TOPK + k)
        if row >= 0:
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * w
            inv = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (inv * tl.load(s_ptr + t * TOPK + k))
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
    bank["sched"] = torch.zeros(4, dtype=torch.int32, device=q.device)
    return bank


def moe_cuda_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, mt=1, ctas_per_sm=2, topk=8, stages=4,
                    ws=False, ks=1, prof=None, abl=0, sab=False, wg=False, stages2=None, kparts=1, pfd=0, sk=False):
    """The pipeline as four separately callable stages (route, gemm1, gemm2, post) sharing
    buffers, for per-stage timing; moe_cuda_forward runs them in order."""
    mod = moe_cuda_module(prof is not None)
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
    active = torch.empty((count + triton.cdiv(p, mt * 16),), dtype=torch.int32, device=device)
    n_active = torch.empty((1,), dtype=torch.int32, device=device)
    inter = torch.empty((p, 4096), dtype=torch.bfloat16, device=device)
    y = torch.empty((p, 4096), dtype=torch.bfloat16, device=device)
    out = torch.empty((m, hidden), dtype=x.dtype, device=device)
    grid = 132 * ctas_per_sm

    tma = wg and stages >= 300
    xs = torch.empty((p, 4096), dtype=torch.bfloat16, device=device) if tma else x

    def route():
        _route_compact_kernel[(count,)](ids, local_start, row_route, pos_of_route, exp_start, active, n_active, p, rows_per_chunk,
                                        NE=256, E=count, CHUNK=256, num_warps=4)
        if tma:
            _gather_rows_kernel[(triton.cdiv(p, 4),)](x, row_route, exp_start, xs, E=count, ROWS=4, H=4096, num_warps=4)

    # prof: None, or (prof1, prof2) int64 buffers [max items * 8] for the profiling build.
    no_prof = torch.empty((0,), dtype=torch.int64, device=device)
    no_part = torch.empty((0, 0, 0), dtype=torch.float32, device=device)
    prof = prof or (no_prof, no_prof)
    sched = bank.get("sched")
    gemm_ws = lambda which: (lambda *a: mod.moe_gemm_ws(*a, ks, prof[which], abl, sched[2 * which:2 * which + 2]))
    g1 = gemm_ws(0) if ws else mod.moe_gemm
    g2 = gemm_ws(1) if ws else mod.moe_gemm
    rows_per_chunk = 8 * mt if (sab or wg) else 16 * mt
    sfx = "_ws" if ws else ""

    split = sab and kparts > 1
    part1 = torch.empty((kparts, p, 8192), dtype=torch.float32, device=device) if split else no_part
    part2 = torch.empty((kparts, p, 4096), dtype=torch.float32, device=device) if split else no_part

    def gemm1():
        if wg:    # v7: mt = 8-token groups per item (N = 8 mt)
            mod.moe_gemm_wg(1, 8 * mt, stages, xs, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active,
                            n_active, exp_start, inter, 64, grid, prof[0], sched[0:2], abl, pfd)
            return
        if sab:   # 64 n blocks (GEMM1: 8 gate/up pairs per CTA), 32 for GEMM2 (16 n8 groups per CTA)
            mod.moe_gemm_sab(1, mt, stages, x, row_route, bank["w13q_ws"], bank["w13s_ws"], bank["w13_gs"], active,
                             n_active, exp_start, inter, 64, grid, prof[0], sched[0:2], abl, kparts, part1, int(sk))
            if split:
                _splitk_silu_kernel[(p,)](part1, exp_start, inter, p, E=count, KP=kparts, H=4096, num_warps=8)
            return
        # 64 work items per active expert per GEMM: 8 warps x (MODE 1: a gate/up pair, MODE 2: an n8 group).
        g1(1, mt, stages, x, row_route, bank["w13q" + sfx], bank["w13s" + sfx], bank["w13_gs"], active, n_active,
             exp_start, inter, 512 // 8, grid)

    def gemm2():
        if wg:
            mod.moe_gemm_wg(2, 8 * mt, stages2 or stages, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active,
                            n_active, exp_start, y, 32, grid, prof[1], sched[2:4], abl, pfd)
            return
        if sab:
            mod.moe_gemm_sab(2, mt, stages, inter, row_route, bank["w2q_sab"], bank["w2s_sab"], bank["w2_gs"], active,
                             n_active, exp_start, y, 32, grid, prof[1], sched[2:4], abl, kparts, part2, int(sk))
            return
        g2(2, mt, stages, inter, row_route, bank["w2q" + sfx], bank["w2s" + sfx], bank["w2_gs"], active, n_active,
             exp_start, y, 512 // 8, grid)

    def post():
        if split:
            _post_norm_parts_kernel[(m,)](part2, norm_w, topk_weights, pos_of_route, out, eps, p, KP=kparts, TOPK=topk,
                                          H=hidden, num_warps=8)
            return
        _post_norm_rows_kernel[(m,)](y, norm_w, topk_weights, pos_of_route, out, eps, TOPK=topk, H=hidden,
                                     num_warps=8)

    return (route, gemm1, gemm2, post), out


def moe_cuda_forward(x, ids, topk_weights, local_start, bank, norm_w, eps, **kw):
    """x [M, 4096] BF16, ids [M, 8] global expert ids, topk_weights [M, 8] FP32, local_start
    int32 [1] (device) -> [M, 4096] BF16: this GPU's experts' weighted, normed sum."""
    fns, out = moe_cuda_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, **kw)
    for fn in fns:
        fn()
    return out
