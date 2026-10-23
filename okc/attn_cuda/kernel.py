"""Memory-first decode attention over a paged FP8 cache (CUDA via load_inline), one launch.

Why: on full-attention layers at B 16/32 the KV gather is everything (333-667 MB of random
128 B rows, page size 1). Random 128 B rows stream at 3.8-4.0 TB/s on the judge's H200
(probe 8433b3e3) while FA3 reaches 1.3-1.7 TB/s and our Triton v3 ~1.9 TB/s; the sliding
layers are a chain of dependent DRAM round trips plus launches.

Design (attn_dec_kernel, L = 1):
- Work balance: request b contributes v_b = n_b + 1 "virtual keys" (n_b cached keys in its
  window, plus the new token). The CTAs of a KV-head pair split the batch's total T = sum v_b
  evenly: CTA i takes [T i / N, T (i + 1) / N), whatever requests that crosses, so every CTA
  streams the same bytes regardless of the lengths' spread (one wave, no tail).
- Gather: tiles of 64 keys; per key one 128 B K row segment (both heads of the pair) and one V
  segment, cp.async 16 B granules into a STAGES-deep ring (XOR-swizzled so the fragment loads
  are conflict-free). Page-table entries are loaded two tiles ahead of their gathers.
- Math: f16 mma.sync. S = Q K^T with M = the 8 query heads of a KV head (rows 8-15 zero) and
  K in the B operand (one LDS.128 per key row gives all four k16 steps, d permuted to match);
  O^T = V^T P^T, so P leaves the S accumulators straight as the B operand and V's bytes are
  paired by PRMT and converted by cvt.f16x2.e4m3x2. Queries: RMSNorm x 5.473 (+ RoPE),
  rounded to BF16 like the reference (exact in f16); P in f16; FP32 accumulation.
- The new token: the CTA whose range holds request b's virtual key n_b normalizes it (+ RoPE),
  stores it to the cache in FP8 and attends to it from registers.
- Split merge: each CTA writes one partial (m, l, acc) per request it touched, slot i + b;
  the last to arrive for (b, pair) (atomic counter, reset by it) merges them and applies
  RMSNorm x sigmoid(gate).
"""
import torch

ATTN_DEVICE_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define ATT_TILE 64
#define ATT_INF __int_as_float(0x7f800000)
#define ATT_MAXB 64
#define ATT_STAGE_BYTES (8192 * 2 + 2048 + 512)

struct AttnParams {
  const __nv_bfloat16* qkv; const float* inv_freq; const int32_t* seq_lens; const int32_t* page_table;
  const int64_t* cache_loc; uint8_t* k_cache; uint8_t* v_cache; const float* k_scale; const float* v_scale;
  __nv_bfloat16* out; float* ws_acc; float* ws_ml; int32_t* counters;
  int qkv_w, pt_stride, B, HQ, HKV, page_shift, window, rope, n_ctas;
  float qk_factor, sm_scale_log2, eps;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t pair16) {
  uint32_t r;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}\n" : "=r"(r) : "r"(pair16));
  return r;
}
__device__ __forceinline__ uint32_t cvt_lo(uint32_t w) { return e4m3x2_f16x2(w & 0xFFFFu); }
__device__ __forceinline__ uint32_t cvt_hi(uint32_t w) { return e4m3x2_f16x2(w >> 16); }

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_f16(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 16 B chunk swizzle of a 128 B key row: conflict-free for the K reads (rows 2m, 2m+1 x four
// chunks of one head) and the V reads (rows 2t, t = 0..3, x two chunks).
__device__ __forceinline__ int att_swz(int key) {
  return ((((key >> 1) & 3) ^ ((key & 1) << 1)) << 1);
}

// The CTA owning virtual key u of T split over n CTAs: start(i) = floor(T i / n).
__device__ __forceinline__ int att_cta_of(long long u, long long T, int n) {
  return (int)(((u + 1) * n - 1) / T);
}

template <int STAGES>
__global__ void __launch_bounds__(128) attn_dec_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  // per stage: K [64][128] | V [64][128] | Q [16][64] bf16 | new K, V [2][2][64] bf16
  float* s_scr = reinterpret_cast<float*>(smem + STAGES * ATT_STAGE_BYTES);  // [4 warps][2][8][66]
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ float s_knew[2][64];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ;

  // Lengths: each request's attended cached keys n_b (its window) and the prefix of v_b = n_b + 1.
  if (warp == 0) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      int b = b0 + lane;
      int len = b < B ? p.seq_lens[b] : 0;
      int n = p.window > 0 ? min(len, p.window - 1) : len;
      int v = b < B ? n + 1 : 0;
      int x = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (b < B) { s_pref[b] = carry + x - v; s_first[b] = len - n; s_n[b] = n; }
      carry += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) s_pref[B] = carry;
  }
  __syncthreads();
  const long long T = s_pref[B];
  const int N = (int)min((long long)p.n_ctas, T);
  const int cta = blockIdx.x;
  if (cta >= N) return;
  const long long r0 = T * cta / N, r1 = T * (cta + 1) / N;

  // This CTA's segments: (request, cached range [u0, uc), holds the new token, tiles).
  if (tid == 0) {
    int ns = 0;
    for (int b = 0; b < B; ++b) {
      long long lo = s_pref[b], hi = s_pref[b + 1];
      if (hi <= r0 || lo >= r1) continue;
      int u0 = (int)max(r0 - lo, 0LL), u1 = (int)min(r1 - lo, hi - lo);
      int n = s_n[b];
      int uc = min(u1, n);
      s_sb[ns] = b; s_su0[ns] = u0; s_suc[ns] = max(uc, u0); s_snew[ns] = u1 > n;
      s_stiles[ns] = uc > u0 ? (uc - u0 + ATT_TILE - 1) / ATT_TILE : 1;
      ++ns;
    }
    s_nseg = ns;
  }
  __syncthreads();
  const int nseg = s_nseg;
  int ntiles = 0;
  for (int s = 0; s < nseg; ++s) ntiles += s_stiles[s];

  const uint32_t smem_u32 = (uint32_t)__cvta_generic_to_shared(smem);
  const int pmask = (1 << p.page_shift) - 1;
  const size_t row_bytes = (size_t)HKV * 64;
  const int my_key = tid >> 1, my_kv = tid & 1;   // gather role: key of the tile, K or V

  // Tile iterators (segment, tile within it) for: page-index loads, gathers, compute.
  int ix_s = 0, ix_k = 0, is_s = 0, is_k = 0;
  auto advance = [&](int& s, int& k) { if (s < nseg && ++k == s_stiles[s]) { ++s; k = 0; } };
  // A tile's page entry for this thread's key: loaded two tiles before its gather, and only
  // combined with the in-page offset at the gather (no register move waits on the load).
  auto load_page = [&](int s, int k, int& pg, int& off) {
    pg = -1; off = 0;
    if (s >= nseg) return;
    int b = s_sb[s];
    int u = s_su0[s] + k * ATT_TILE + my_key;
    if (u >= s_suc[s]) return;
    int pos = s_first[b] + u;
    off = pos & pmask;
    pg = __ldg(p.page_table + (size_t)b * p.pt_stride + (pos >> p.page_shift));
  };
  auto issue = [&](int s, int k, int pg, int off, int stage) {
    if (s < nseg) {
      long long row = pg < 0 ? -1 : ((long long)pg << p.page_shift) + off;
      uint32_t st = smem_u32 + stage * ATT_STAGE_BYTES;
      const uint8_t* src = (my_kv ? p.v_cache : p.k_cache) + (row < 0 ? 0 : row) * row_bytes + pair * 128;
      uint32_t dst = st + my_kv * 8192 + my_key * 128;
      uint32_t bytes = row < 0 ? 0u : 16u;
      int sw = att_swz(my_key);
#pragma unroll
      for (int j = 0; j < 8; ++j) cp_async16(dst + ((j ^ sw) << 4), src + j * 16, bytes);
      int b = s_sb[s];
      if (k == 0) {  // the request's 16 query heads of this pair (2 KB)
        const uint8_t* q = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + pair * 16 * 64);
        cp_async16(st + 16384 + tid * 16, q + tid * 16, 16u);
      }
      if (k == s_stiles[s] - 1 && s_snew[s] && tid < 32) {  // the new token's K and V rows (bf16)
        int kv = tid >> 4, c = tid & 15;
        const uint8_t* src2 = reinterpret_cast<const uint8_t*>(
            p.qkv + (size_t)b * p.qkv_w + (HQ + kv * HKV) * 64 + pair * 128);
        cp_async16(st + 16384 + 2048 + kv * 256 + c * 16, src2 + c * 16, 16u);
      }
    }
    cp_commit();
  };

  // Prologue: page rows of tiles 0..STAGES (gathers of 0..STAGES-2 issued now).
  for (int j = 0; j < STAGES - 1; ++j) {
    int pg, off;
    load_page(ix_s, ix_k, pg, off); advance(ix_s, ix_k);
    issue(is_s, is_k, pg, off, j); advance(is_s, is_k);
  }
  int pg0, off0, pg1, off1;
  load_page(ix_s, ix_k, pg0, off0); advance(ix_s, ix_k);
  load_page(ix_s, ix_k, pg1, off1); advance(ix_s, ix_k);

  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale;

  uint32_t qa[2][4][2];
  float m_r[2], l_r[2], acc[2][4][4];
  int cs = 0, ck = 0;  // compute iterator
  // Two tiles per iteration so the two page registers alternate without moves.
  auto step = [&](int j, int& pg, int& off) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    issue(is_s, is_k, pg, off, (j + STAGES - 1) % STAGES); advance(is_s, is_k);
    load_page(ix_s, ix_k, pg, off); advance(ix_s, ix_k);

    const uint8_t* st = smem + (j % STAGES) * ATT_STAGE_BYTES;
    const int b = s_sb[cs];
    if (ck == 0) {
      // Queries: RMSNorm x factor (+ RoPE at the token's position), rounded to BF16, as f16
      // A fragments: k-step c holds d = 16t + 4c + {0,1} (a0) and {2,3} (a2) of head row g.
      const __nv_bfloat16* qs = reinterpret_cast<const __nv_bfloat16*>(st + 16384);
      const int pos = s_first[b] + s_n[b];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float x[16];
        const uint4* src = reinterpret_cast<const uint4*>(qs + (8 * h + g) * 64 + 16 * t);
        uint4 w0 = src[0], w1 = src[1];
        const __nv_bfloat162* hw0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
        const __nv_bfloat162* hw1 = reinterpret_cast<const __nv_bfloat162*>(&w1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 f0 = __bfloat1622float2(hw0[i]), f1 = __bfloat1622float2(hw1[i]);
          x[2 * i] = f0.x; x[2 * i + 1] = f0.y; x[8 + 2 * i] = f1.x; x[8 + 2 * i + 1] = f1.y;
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        float sc = rsqrtf(ss / 64.f + p.eps) * p.qk_factor;
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] *= sc;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float sn, cn;
            sincosf((float)pos * __ldg(p.inv_freq + 8 * t + i), &sn, &cn);
            float e = x[2 * i], o = x[2 * i + 1];
            x[2 * i] = e * cn - o * sn;
            x[2 * i + 1] = o * cn + e * sn;
          }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          qa[h][c][0] = pack_h2(x[4 * c], x[4 * c + 1]);
          qa[h][c][1] = pack_h2(x[4 * c + 2], x[4 * c + 3]);
        }
        m_r[h] = -ATT_INF; l_r[h] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[h][i][0] = acc[h][i][1] = acc[h][i][2] = acc[h][i][3] = 0.f; }
      }
    }

    const int nvalid = min(s_suc[cs] - (s_su0[cs] + ck * ATT_TILE) - 16 * warp, 16);
    if (nvalid > 0) {
      const uint8_t* sK = st + (16 * warp) * 128;
      const uint8_t* sV = st + 8192 + (16 * warp) * 128;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        // S = Q K^T: two n8 tiles (keys g, g + 8 of this warp's 16)
        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          int key = 8 * nt + g;
          uint4 kw = *reinterpret_cast<const uint4*>(sK + key * 128 + (((4 * h + t) ^ att_swz(key)) << 4));
          uint32_t kws[4] = {kw.x, kw.y, kw.z, kw.w};
          s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
          for (int c = 0; c < 4; ++c)
            mma_f16(s[nt], qa[h][c][0], 0u, qa[h][c][1], 0u, cvt_lo(kws[c]), cvt_hi(kws[c]));
        }
        // online softmax on row g (keys 2t, 2t+1 of each n tile), log2 units
        float x0 = 2 * t < nvalid ? s[0][0] * c_log2 : -ATT_INF;
        float x1 = 2 * t + 1 < nvalid ? s[0][1] * c_log2 : -ATT_INF;
        float x2 = 8 + 2 * t < nvalid ? s[1][0] * c_log2 : -ATT_INF;
        float x3 = 9 + 2 * t < nvalid ? s[1][1] * c_log2 : -ATT_INF;
        float mt = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
        float mn = fmaxf(m_r[h], mt);
        float alpha = ex2(m_r[h] - mn);
        m_r[h] = mn;
        float p0 = ex2(x0 - mn), p1 = ex2(x1 - mn), p2 = ex2(x2 - mn), p3 = ex2(x3 - mn);
        l_r[h] = l_r[h] * alpha + (p0 + p1) + (p2 + p3);
        float al_a = __shfl_sync(0xffffffffu, alpha, 8 * t);
        float al_b = __shfl_sync(0xffffffffu, alpha, 8 * t + 4);
        uint32_t pb0 = pack_h2(p0, p1), pb1 = pack_h2(p2, p3);
        // O^T += V^T P^T: m tile i holds d = 8g + i (rows g) and 8g + 4 + i (rows g + 8)
        const int vo = ((4 * h + (g >> 1)) << 4) | ((g & 1) << 3);
        uint2 va = *reinterpret_cast<const uint2*>(sV + (2 * t) * 128 + (vo ^ (att_swz(2 * t) << 4)));
        uint2 vb = *reinterpret_cast<const uint2*>(sV + (2 * t + 1) * 128 + (vo ^ (att_swz(2 * t + 1) << 4)));
        uint2 vc = *reinterpret_cast<const uint2*>(sV + (2 * t + 8) * 128 + (vo ^ (att_swz(2 * t + 8) << 4)));
        uint2 vd = *reinterpret_cast<const uint2*>(sV + (2 * t + 9) * 128 + (vo ^ (att_swz(2 * t + 9) << 4)));
        uint32_t lo01 = __byte_perm(va.x, vb.x, 0x5140), lo23 = __byte_perm(va.x, vb.x, 0x7362);
        uint32_t hi01 = __byte_perm(va.y, vb.y, 0x5140), hi23 = __byte_perm(va.y, vb.y, 0x7362);
        uint32_t clo01 = __byte_perm(vc.x, vd.x, 0x5140), clo23 = __byte_perm(vc.x, vd.x, 0x7362);
        uint32_t chi01 = __byte_perm(vc.y, vd.y, 0x5140), chi23 = __byte_perm(vc.y, vd.y, 0x7362);
        uint32_t A0[4] = {cvt_lo(lo01), cvt_hi(lo01), cvt_lo(lo23), cvt_hi(lo23)};
        uint32_t A1[4] = {cvt_lo(hi01), cvt_hi(hi01), cvt_lo(hi23), cvt_hi(hi23)};
        uint32_t A2[4] = {cvt_lo(clo01), cvt_hi(clo01), cvt_lo(clo23), cvt_hi(clo23)};
        uint32_t A3[4] = {cvt_lo(chi01), cvt_hi(chi01), cvt_lo(chi23), cvt_hi(chi23)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[h][i][0] *= al_a; acc[h][i][1] *= al_b; acc[h][i][2] *= al_a; acc[h][i][3] *= al_b;
          mma_f16(acc[h][i], A0[i], A1[i], A2[i], A3[i], pb0, pb1);
        }
      }
    }

    if (ck == s_stiles[cs] - 1) {
      // ---- end of this request's share: the new token, then this CTA's partial ----
      if (s_snew[cs] && warp == 0) {
        const __nv_bfloat16* kn = reinterpret_cast<const __nv_bfloat16*>(st + 16384 + 2048);
        const int pos = s_first[b] + s_n[b];
        const long long loc = p.cache_loc[b];
        // lane: head lane >> 4, d = 4 (lane & 15) .. + 3
        const int hh = lane >> 4, d0 = 4 * (lane & 15);
        float k4[4], v4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          k4[i] = __bfloat162float(kn[hh * 64 + d0 + i]);
          v4[i] = __bfloat162float(kn[128 + hh * 64 + d0 + i]);
        }
        float ss = k4[0] * k4[0] + k4[1] * k4[1] + k4[2] * k4[2] + k4[3] * k4[3];
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        float inv = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) k4[i] *= inv;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            float sn, cn;
            sincosf((float)pos * __ldg(p.inv_freq + d0 / 2 + i), &sn, &cn);
            float e = k4[2 * i], o = k4[2 * i + 1];
            k4[2 * i] = e * cn - o * sn;
            k4[2 * i + 1] = o * cn + e * sn;
          }
        }
        uint32_t kq = 0, vq = 0;
        float kd[4], vdq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          __nv_fp8_e4m3 a(k4[i] / k_scale), c(v4[i] / v_scale);
          kq |= (uint32_t)a.__x << (8 * i);
          vq |= (uint32_t)c.__x << (8 * i);
          kd[i] = (float)a; vdq[i] = (float)c;
        }
        size_t off = ((size_t)loc * HKV + 2 * pair + hh) * 64 + d0;
        *reinterpret_cast<uint32_t*>(p.k_cache + off) = kq;
        *reinterpret_cast<uint32_t*>(p.v_cache + off) = vq;
#pragma unroll
        for (int i = 0; i < 4; ++i) { s_knew[hh][d0 + i] = kd[i]; }
        __syncwarp();
        // scores of rows g (query heads 8h + g) against the new key: q holds d = 16t + 4c + {0..3}
        float sc[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a = 0.f;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float2 q01 = __half22float2(*reinterpret_cast<__half2*>(&qa[h][c][0]));
            float2 q23 = __half22float2(*reinterpret_cast<__half2*>(&qa[h][c][1]));
            const float* kk = &s_knew[h][16 * t + 4 * c];
            a += q01.x * kk[0] + q01.y * kk[1] + q23.x * kk[2] + q23.y * kk[3];
          }
          a += __shfl_xor_sync(0xffffffffu, a, 1);
          a += __shfl_xor_sync(0xffffffffu, a, 2);
          sc[h] = a * c_log2;
        }
        __syncwarp();
        // reuse s_knew for the new value (dequantized FP8, v_scale applied at the end)
#pragma unroll
        for (int i = 0; i < 4; ++i) s_knew[hh][d0 + i] = vdq[i];
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float mn = fmaxf(m_r[h], sc[h]);
          float alpha = ex2(m_r[h] - mn), pn = ex2(sc[h] - mn);
          m_r[h] = mn;
          l_r[h] = l_r[h] * alpha + (t == 0 ? pn : 0.f);
          float al_a = __shfl_sync(0xffffffffu, alpha, 8 * t), al_b = __shfl_sync(0xffffffffu, alpha, 8 * t + 4);
          float pa = __shfl_sync(0xffffffffu, pn, 8 * t), pbb = __shfl_sync(0xffffffffu, pn, 8 * t + 4);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float v0 = s_knew[h][8 * g + i], v1 = s_knew[h][8 * g + 4 + i];
            acc[h][i][0] = acc[h][i][0] * al_a + pa * v0;
            acc[h][i][1] = acc[h][i][1] * al_b + pbb * v0;
            acc[h][i][2] = acc[h][i][2] * al_a + pa * v1;
            acc[h][i][3] = acc[h][i][3] * al_b + pbb * v1;
          }
        }
      }
      // warps -> scratch [warp][h][row][66]: m, l, acc[64]
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float l = l_r[h];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        float* sw = s_scr + ((warp * 2 + h) * 8) * 66;
        if (t == 0) { sw[g * 66] = m_r[h]; sw[g * 66 + 1] = l; }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sw[(2 * t) * 66 + 2 + 8 * g + i] = acc[h][i][0];
          sw[(2 * t + 1) * 66 + 2 + 8 * g + i] = acc[h][i][1];
          sw[(2 * t) * 66 + 2 + 8 * g + 4 + i] = acc[h][i][2];
          sw[(2 * t + 1) * 66 + 2 + 8 * g + 4 + i] = acc[h][i][3];
        }
      }
      __syncthreads();
      {
        const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
        float M = -ATT_INF;
#pragma unroll
        for (int w = 0; w < 4; ++w) M = fmaxf(M, s_scr[((w * 2 + h) * 8 + r) * 66]);
        float L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float* sw = s_scr + ((w * 2 + h) * 8 + r) * 66;
          float f = ex2(sw[0] - M);
          L += sw[1] * f;
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] += sw[2 + 8 * dc + i] * f;
        }
        const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
        float* wa = p.ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc;
        reinterpret_cast<float4*>(wa)[0] = make_float4(o[0], o[1], o[2], o[3]);
        reinterpret_cast<float4*>(wa)[1] = make_float4(o[4], o[5], o[6], o[7]);
        if (dc == 0) reinterpret_cast<float2*>(p.ws_ml)[(slot * 2 + h) * 8 + r] = make_float2(M, L);
      }
      __threadfence();
      __syncthreads();
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      if (tid == 0) {
        int old = atomicAdd(p.counters + b * gridDim.y + pair, 1);
        int last = old == i_last - i_first;
        if (last) p.counters[b * gridDim.y + pair] = 0;
        s_last = last;
      }
      __syncthreads();
      if (s_last) {
        __threadfence();
        const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
        const int qh = (2 * pair + h) * 8 + r;
        const uint4 gw = *reinterpret_cast<const uint4*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + qh * 64 + 8 * dc);
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = i_first; i <= i_last; ++i) {
          const size_t slot = (size_t)pair * (p.n_ctas + B) + i + b;
          float2 ml = __ldcg(reinterpret_cast<const float2*>(p.ws_ml) + (slot * 2 + h) * 8 + r);
          const float4* wa = reinterpret_cast<const float4*>(p.ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc);
          float4 a0 = __ldcg(wa), a1 = __ldcg(wa + 1);
          float mn = fmaxf(M, ml.x);
          float fa = ex2(M - mn), fb = ex2(ml.x - mn);
          M = mn;
          L = L * fa + ml.y * fb;
          o[0] = o[0] * fa + a0.x * fb; o[1] = o[1] * fa + a0.y * fb; o[2] = o[2] * fa + a0.z * fb;
          o[3] = o[3] * fa + a0.w * fb; o[4] = o[4] * fa + a1.x * fb; o[5] = o[5] * fa + a1.y * fb;
          o[6] = o[6] * fa + a1.z * fb; o[7] = o[7] * fa + a1.w * fb;
        }
        float sc = v_scale / L, ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[i] *= sc; ss += o[i] * o[i]; }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        ss += __shfl_xor_sync(0xffffffffu, ss, 4);
        float inv = rsqrtf(ss / 64.f + p.eps);
        const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw);
        uint4 ow;
        __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 gf = __bfloat1622float2(gh[i]);
          float y0 = o[2 * i] * inv / (1.f + __expf(-gf.x));
          float y1 = o[2 * i + 1] * inv / (1.f + __expf(-gf.y));
          oh[i] = __floats2bfloat162_rn(y0, y1);
        }
        *reinterpret_cast<uint4*>(p.out + (size_t)b * HQ * 64 + qh * 64 + 8 * dc) = ow;
      }
      __syncthreads();
      ++cs; ck = 0;
    } else {
      ++ck;
    }
  };
  for (int j = 0; j < ntiles; j += 2) {
    step(j, pg0, off0);
    if (j + 1 < ntiles) step(j + 1, pg1, off1);
  }
  cp_wait<0>();
}
"""

ATTN_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <int S>
static void launch_dec(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * ATT_STAGE_BYTES + 4 * 2 * 8 * 66 * sizeof(float);
  auto kern = attn_dec_kernel<S>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor counters, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages) {
  AttnParams p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.inv_freq = inv_freq.data_ptr<float>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.page_table = page_table.data_ptr<int32_t>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_ml = ws_ml.data_ptr<float>();
  p.counters = counters.data_ptr<int32_t>();
  p.qkv_w = (int)qkv.size(-1);
  p.pt_stride = (int)page_table.stride(0);
  p.B = (int)B; p.HQ = (int)HQ; p.HKV = (int)HKV; p.page_shift = (int)page_shift; p.window = (int)window;
  p.rope = (int)rope; p.n_ctas = (int)n_ctas;
  p.qk_factor = (float)qk_factor; p.sm_scale_log2 = (float)sm_scale_log2; p.eps = (float)eps;
  int pairs = (int)HKV / 2;
  if (stages == 3) launch_dec<3>(p, (int)n_ctas, pairs);
  else if (stages == 4) launch_dec<4>(p, (int)n_ctas, pairs);
  else if (stages == 6) launch_dec<6>(p, (int)n_ctas, pairs);
  else TORCH_CHECK(false, "attn_dec: unsupported stages");
}
"""

ATTN_CUDA_SRC = ATTN_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n",
                                      "#include <ATen/cuda/CUDAContext.h>\n" + ATTN_DEVICE_SRC, 1)

ATTN_CUDA_CPP = """
void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor counters, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages);
"""

_ATTN_MODULES = {}
_ATTN_COUNTERS = {}


def attn_cuda_module():
    if "m" not in _ATTN_MODULES:
        from torch.utils.cpp_extension import load_inline
        _ATTN_MODULES["m"] = load_inline(
            name="okc_attn_dec", cpp_sources=ATTN_CUDA_CPP, cuda_sources=ATTN_CUDA_SRC, functions=["attn_dec"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _ATTN_MODULES["m"]


def attn_cuda_supported(cfg, page_table):
    """Shapes the CUDA path covers (others take the Triton / SGLang paths)."""
    page = cfg["page_size"]
    return (cfg["q_len"] == 1 and cfg["head_dim"] == 64 and cfg["num_kv_heads"] % 2 == 0
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["batch"] <= 64
            and page & (page - 1) == 0)


def attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=264, stages=4, min_keys=256):
    """qkv [B, W] bf16 -> [B, Hq * 64] bf16; writes the new cache entries. n_ctas: CTAs per
    KV-head pair at most (fewer when the batch has fewer than min_keys keys per CTA)."""
    batch = cfg["batch"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    pairs = hkv // 2
    page = cfg["page_size"]
    window = cfg["window"]
    page_table = data["page_table"]
    capacity = page_table.shape[1] * page
    span = min(window, capacity) if window else capacity
    n = max(1, min(max(1, n_ctas // pairs), -(-batch * span // min_keys)))
    device = qkv.device
    key = (device, batch * pairs)
    counters = _ATTN_COUNTERS.get(key)
    if counters is None:
        counters = _ATTN_COUNTERS[key] = torch.zeros(batch * pairs, dtype=torch.int32, device=device)
    slots = pairs * (n + batch)
    ws_acc = torch.empty((slots, 2, 8, 64), dtype=torch.float32, device=device)
    ws_ml = torch.empty((slots, 2, 8, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, hq * 64), dtype=torch.bfloat16, device=device)
    attn_cuda_module().attn_dec(
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], out, ws_acc, ws_ml, counters, batch, hq, hkv, page.bit_length() - 1,
        window, int(bool(cfg["rope_theta"])), cfg["qk_scale_factor"], cfg["sm_scale"] * 1.4426950408889634, 1e-5,
        n, stages)
    return out
