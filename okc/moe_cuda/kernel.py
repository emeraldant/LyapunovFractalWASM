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
template <int MODE, int MT, int NGW, int STAGES, int KS>
__global__ void __launch_bounds__(32 * (8 * KS + 1)) moe_gemm_ws_kernel(
    const __nv_bfloat16* __restrict__ a_src, const int32_t* __restrict__ row_route,
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, const float* __restrict__ gs,
    const int32_t* __restrict__ active, const int32_t* __restrict__ n_active_ptr,
    const int32_t* __restrict__ exp_start, __nv_bfloat16* __restrict__ out, int n_blocks) {
  using C = WsCfg<MODE, MT, NGW>;
  constexpr int NCHUNK = KTOT / 128;
  static_assert(NCHUNK % STAGES == 0 && STAGES % KS == 0, "chunks / stages / K groups must divide");
  static_assert(8 * KS + 1 <= 32, "at most 1024 threads per CTA");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bar_full = base + STAGES * C::STAGE, bar_empty = bar_full + 8 * STAGES;
  float* red = reinterpret_cast<float*>(smem + STAGES * (C::STAGE + 16));
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
    uint32_t f = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
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
      const uint32_t tx = live * 256 + C::W_STAGE + C::S_STAGE;
      for (int kc = 0; kc < NCHUNK; ++kc, ++f) {
        const uint32_t st = f % STAGES;
        if (f >= STAGES) mbar_wait(bar_empty + 8 * st, ((f / STAGES) - 1) & 1);
        const uint32_t sb = base + st * C::STAGE, full = bar_full + 8 * st;
        if (lane == 0) {
          mbar_expect_tx(full, tx);
          bulk_g2s(sb + C::A_STAGE, w_blk + (size_t)kc * C::W_STAGE, C::W_STAGE, full);
          bulk_g2s(sb + C::A_STAGE + C::W_STAGE, s_blk + (size_t)kc * C::S_STAGE, C::S_STAGE, full);
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int lr = lane + 32 * h;
          if (lr < live) bulk_g2s(sb + lr * C::A_ROW, a_row[h] + kc * 256, 256, full);
        }
      }
    }
    return;
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
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int nb = item % n_blocks;
    const int ch = active[item / n_blocks];
    const int e = ch & 255, mc = ch >> 8;
    const int row0 = exp_start[e] + mc * MT * 16;
    const int rows_e = exp_start[e + 1] - row0;
    const int tiles = min(MT, (rows_e + 15) / 16);
    const float gsc = gs[e] * 256.0f;
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

#pragma unroll 1
    for (int kc0 = 0; kc0 < NCHUNK; kc0 += STAGES, f += STAGES) {
      const uint32_t parity = (f / STAGES) & 1;
#pragma unroll
      for (int jst = 0; jst < STAGES / KS; ++jst) {
        const int st = jst * KS;   // + kg, folded into the base addresses
        mbar_wait(my_full + 8 * st, parity);
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
              if ((step % 4) < 2) {
                b0[j] = mul2h<0>(v[j][2 * half], sp);
                b1[j] = mul2h<0>(v[j][2 * half + 1], sp);
              } else {
                b0[j] = mul2h<1>(v[j][2 * half], sp);
                b1[j] = mul2h<1>(v[j][2 * half + 1], sp);
              }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              // Tile 0 always has rows; the guard (a WARPSYNC per ldmatrix) only for the others.
              if (mt == 0 || mt < tiles) {
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"(lds_off[step] + a_stage + mt * 16 * C::A_ROW));
#pragma unroll
                for (int j = 0; j < NGW; ++j) mma16816(acc[half % NACC][mt][j], a, b0[j], b1[j]);
              }
            }
          }
        }
        mbar_arrive(my_empty + 8 * st);
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
  }
}
"""

MOE_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

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

template <int MODE, int MT, int STAGES, int KS>
void launch_ws(torch::Tensor a_src, torch::Tensor row_route, torch::Tensor wq, torch::Tensor ws, torch::Tensor gs,
               torch::Tensor active, torch::Tensor n_active, torch::Tensor exp_start, torch::Tensor out,
               int n_blocks, int grid) {
  constexpr int NGW = MODE == 1 ? 2 : 1;
  constexpr int THREADS = 32 * (8 * KS + 1);
  size_t smem = (size_t)STAGES * (WsCfg<MODE, MT, NGW>::STAGE + 16) + (size_t)(KS - 1) * 8 * MT * NGW * 4 * 32 * 4;
  auto k = moe_gemm_ws_kernel<MODE, MT, NGW, STAGES, KS>;
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
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_blocks);
}

void moe_gemm_ws(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, int64_t ks) {
#define WCASE(M_, T_, S_, K_) if (mode == M_ && mt == T_ && stages == S_ && ks == K_) { launch_ws<M_, T_, S_, K_>(a_src, row_route, wq, ws, gs, active, n_active, exp_start, out, n_blocks, grid); return; }
  WCASE(1, 1, 4, 1) WCASE(2, 1, 4, 1) WCASE(1, 2, 4, 1) WCASE(2, 2, 4, 1) WCASE(1, 1, 8, 1) WCASE(2, 1, 8, 1)
  WCASE(1, 1, 8, 2) WCASE(2, 1, 8, 2) WCASE(1, 1, 4, 2) WCASE(2, 1, 4, 2)   // KS * 8 + 1 warps <= 32
  TORCH_CHECK(false, "moe_gemm_ws: unsupported mode / m tiles / stages");
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
void moe_gemm_ws(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
                 torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
                 torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid, int64_t ks);
void moe_gemm(int64_t mode, int64_t mt, int64_t stages, torch::Tensor a_src, torch::Tensor row_route,
              torch::Tensor wq, torch::Tensor ws, torch::Tensor gs, torch::Tensor active, torch::Tensor n_active,
              torch::Tensor exp_start, torch::Tensor out, int64_t n_blocks, int64_t grid);
"""

_MOE_MODULE = []


def moe_cuda_module():
    if not _MOE_MODULE:
        from torch.utils.cpp_extension import load_inline
        _MOE_MODULE.append(load_inline(
            name="okc_moe_gemm", cpp_sources=MOE_CUDA_CPP, cuda_sources=MOE_CUDA_SRC, functions=["moe_gemm", "moe_gemm_ws"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False))
    return _MOE_MODULE[0]


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
    return bank


def moe_cuda_stages(x, ids, topk_weights, local_start, bank, norm_w, eps, mt=1, ctas_per_sm=2, topk=8, stages=4,
                    ws=False, ks=1):
    """The pipeline as four separately callable stages (route, gemm1, gemm2, post) sharing
    buffers, for per-stage timing; moe_cuda_forward runs them in order."""
    mod = moe_cuda_module()
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

    def route():
        _route_compact_kernel[(count,)](ids, local_start, row_route, pos_of_route, exp_start, active, n_active, p, mt * 16,
                                        NE=256, E=count, CHUNK=256, num_warps=4)

    gemm = (lambda *a: mod.moe_gemm_ws(*a, ks)) if ws else mod.moe_gemm
    sfx = "_ws" if ws else ""

    def gemm1():
        # 64 work items per active expert per GEMM: 8 warps x (MODE 1: a gate/up pair, MODE 2: an n8 group).
        gemm(1, mt, stages, x, row_route, bank["w13q" + sfx], bank["w13s" + sfx], bank["w13_gs"], active, n_active,
             exp_start, inter, 512 // 8, grid)

    def gemm2():
        gemm(2, mt, stages, inter, row_route, bank["w2q" + sfx], bank["w2s" + sfx], bank["w2_gs"], active, n_active,
             exp_start, y, 512 // 8, grid)

    def post():
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
