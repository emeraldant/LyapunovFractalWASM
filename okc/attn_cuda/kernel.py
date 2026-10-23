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
  __nv_bfloat16* out; float* ws_acc; float* ws_ml; float* ws2_acc; float* ws2_ml; int32_t* counters; long long* prof;
  int qkv_w, pt_stride, B, HQ, HKV, page_shift, window, rope, n_ctas, max_groups;
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

// sin, cos of an fp32 angle (up to ~1e6 rad): Cody-Waite reduction by 2 pi in two parts, then
// the fast intrinsics on [-pi, pi] (abs error ~1e-6, far below the BF16 rounding that follows).
__device__ __forceinline__ void att_sincos(float a, float& sn, float& cn) {
  const float k = rintf(a * 0.15915494309189535f);
  float r = fmaf(-k, 6.28318548202514648f, a);
  r = fmaf(-k, -1.7484555314695172e-07f, r);
  __sincosf(r, &sn, &cn);
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

#define ATT_GROUP 16

// Optional phase stamps (sim only): prof[cta][k] = clock64 at event k (62: globaltimer at the end,
// 63: at the start).
#define ATT_STAMP(k) do { if (p.prof != nullptr && threadIdx.x == 0) \
  p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + (k)] = clock64(); } while (0)
__device__ __forceinline__ long long att_gtime() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

// Merges `count` partials (m, l, acc) at slots slot0 .. slot0 + count - 1 for this thread's
// (head h, row r, 8 d values from 8 dc): loads four partials ahead of their merges.
__device__ __forceinline__ void att_merge(const float* ws_acc, const float* ws_ml, size_t slot0, int count, int h,
                                          int r, int dc, float& M, float& L, float o[8]) {
  for (int k0 = 0; k0 < count; k0 += 4) {
    float2 ml[4];
    float4 a0[4], a1[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t slot = slot0 + min(k0 + u, count - 1);
      ml[u] = __ldcg(reinterpret_cast<const float2*>(ws_ml) + (slot * 2 + h) * 8 + r);
      const float4* wa = reinterpret_cast<const float4*>(ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc);
      a0[u] = __ldcg(wa); a1[u] = __ldcg(wa + 1);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (k0 + u < count) {
        float mn = fmaxf(M, ml[u].x);
        float fa = ex2(M - mn), fb = ex2(ml[u].x - mn);
        M = mn;
        L = L * fa + ml[u].y * fb;
        o[0] = o[0] * fa + a0[u].x * fb; o[1] = o[1] * fa + a0[u].y * fb; o[2] = o[2] * fa + a0[u].z * fb;
        o[3] = o[3] * fa + a0[u].w * fb; o[4] = o[4] * fa + a1[u].x * fb; o[5] = o[5] * fa + a1[u].y * fb;
        o[6] = o[6] * fa + a1[u].z * fb; o[7] = o[7] * fa + a1[u].w * fb;
      }
    }
  }
}

__device__ __forceinline__ void att_store_partial(float* ws_acc, float* ws_ml, size_t slot, int h, int r, int dc,
                                                  float M, float L, const float o[8]) {
  float* wa = ws_acc + ((slot * 2 + h) * 8 + r) * 64 + 8 * dc;
  reinterpret_cast<float4*>(wa)[0] = make_float4(o[0], o[1], o[2], o[3]);
  reinterpret_cast<float4*>(wa)[1] = make_float4(o[4], o[5], o[6], o[7]);
  if (dc == 0) reinterpret_cast<float2*>(ws_ml)[(slot * 2 + h) * 8 + r] = make_float2(M, L);
}

template <int STAGES>
__global__ void __launch_bounds__(128) attn_dec_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  // per stage: K [64][128] | V [64][128] | Q [16][64] bf16 | new K, V [2][2][64] bf16
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];
  __shared__ __align__(16) __nv_bfloat16 s_q0[16 * 64];
  __shared__ float s_knew[2][64];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ;
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 63] = att_gtime();
  ATT_STAMP(0);

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
  ATT_STAMP(1);
  if (cta >= N) return;
  const long long r0 = T * cta / N, r1 = T * (cta + 1) / N;

  // This CTA's segments: (request, cached range [u0, uc), holds the new token, tiles); warp 0,
  // a lane per request.
  if (warp == 0) {
    int ns = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long lo = 0, hi = 0;
      bool hit = false;
      if (b < B) { lo = s_pref[b]; hi = s_pref[b + 1]; hit = hi > r0 && lo < r1; }
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int i = ns + __popc(m & ((1u << lane) - 1u));
        const int u0 = (int)max(r0 - lo, 0LL), u1 = (int)min(r1 - lo, hi - lo);
        const int n = s_n[b], uc = min(u1, n);
        s_sb[i] = b; s_su0[i] = u0; s_suc[i] = max(uc, u0); s_snew[i] = u1 > n;
        s_stiles[i] = uc > u0 ? (uc - u0 + ATT_TILE - 1) / ATT_TILE : 1;
      }
      ns += __popc(m);
    }
    if (lane == 0) s_nseg = ns;
  }
  __syncthreads();
  ATT_STAMP(2);
  const int nseg = s_nseg;
  int ntiles = 0;
  for (int s = 0; s < nseg; ++s) ntiles += s_stiles[s];

  const uint32_t smem_u32 = (uint32_t)__cvta_generic_to_shared(smem);
  const int pmask = (1 << p.page_shift) - 1;
  const size_t row_bytes = (size_t)HKV * 64;
  // Gather roles, coalesced: lanes 8r..8r+7 copy the eight 16 B granules of one 128 B row, so a
  // warp instruction moves four whole rows (a row per thread made every instruction touch 32
  // rows with 16 B each: DRAM served scattered sectors at ~0.8 TB/s). Thread rows: K and V of
  // keys rs + 16q, q = 0..3.
  const int rs = tid >> 3, gr = tid & 7;

  // Tile iterators (segment, tile within it): page-entry copies (STAGES - 1 tiles ahead of the
  // gathers, into an 8-tile smem ring by 4 B cp.async: nothing waits on them in registers),
  // gathers, compute. One commit group per tile: its gather plus a later tile's page entries.
  int ip_s = 0, ip_k = 0, ip_t = 0, is_s = 0, is_k = 0, is_t = 0;
  auto advance = [&](int& s, int& k, int& tt) { ++tt; if (s < nseg && ++k == s_stiles[s]) { ++s; k = 0; } };
  auto copy_pages = [&]() {
    if (tid < ATT_TILE && ip_s < nseg) {
      const int b = s_sb[ip_s];
      const int u = s_su0[ip_s] + ip_k * ATT_TILE + tid;
      const bool ok = u < s_suc[ip_s];
      const int pos = s_first[b] + (ok ? u : 0);
      const int32_t* src = p.page_table + (size_t)b * p.pt_stride + (pos >> p.page_shift);
      const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_pg[ip_t & 7][tid]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 4 : 0));
    }
    advance(ip_s, ip_k, ip_t);
  };
  auto gather = [&](int stage) {
    if (is_s < nseg) {
      const int s = is_s, k = is_k, b = s_sb[s];
      const uint32_t st = smem_u32 + stage * ATT_STAGE_BYTES;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int key = rs + 16 * q;
        const int u = s_su0[s] + k * ATT_TILE + key;
        const bool ok = u < s_suc[s];
        const int pos = s_first[b] + u;
        const long long row = ok ? ((long long)s_pg[is_t & 7][key] << p.page_shift) + (pos & pmask) : 0;
        const size_t go = row * row_bytes + pair * 128 + gr * 16;
        const uint32_t dst = st + key * 128 + ((gr ^ att_swz(key)) << 4);
        cp_async16(dst, p.k_cache + go, ok ? 16u : 0u);
        cp_async16(dst + 8192, p.v_cache + go, ok ? 16u : 0u);
      }
      if (k == 0 && s > 0) {  // a later request's 16 query heads of this pair (2 KB); its gate rows to L2
        const uint8_t* q = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + pair * 16 * 64);
        cp_async16(st + 16384 + tid * 16, q + tid * 16, 16u);
        if (tid < 16) {
          const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + pair * 16 * 64);
          asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + tid * 128));
        }
      }
      if (k == s_stiles[s] - 1 && s_snew[s] && tid < 32) {  // the new token's K and V rows (bf16)
        const int kv = tid >> 4, c = tid & 15;
        const uint8_t* src2 = reinterpret_cast<const uint8_t*>(
            p.qkv + (size_t)b * p.qkv_w + (HQ + kv * HKV) * 64 + pair * 128);
        cp_async16(st + 16384 + 2048 + kv * 256 + c * 16, src2 + c * 16, 16u);
      }
    }
    advance(is_s, is_k, is_t);
  };

  // Prologue: the first tiles' page entries and the first request's queries in one round trip,
  // then the first STAGES - 1 gathers; the queries are prepared while those are in flight.
  for (int m = 0; m < STAGES - 1; ++m) copy_pages();
  if (nseg > 0) {
    const int b = s_sb[0];
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + pair * 16 * 64);
    cp_async16((uint32_t)__cvta_generic_to_shared(s_q0) + tid * 16, q + tid * 16, 16u);
    if (tid < 16) {
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + pair * 16 * 64);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + tid * 128));
    }
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int m = 0; m < STAGES - 1; ++m) { gather(m); copy_pages(); cp_commit(); }

  ATT_STAMP(3);
  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale;

  uint32_t qa[2][4][2];
  float m_r[2], l_r[2], acc[2][4][4];
  // Queries: RMSNorm x factor (+ RoPE at the token's position), rounded to BF16, as f16 A
  // fragments: k-step c holds d = 16t + 4c + {0,1} (a0) and {2,3} (a2) of head row g.
  auto prep_q = [&](const __nv_bfloat16* qs, int b) {
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
          att_sincos((float)pos * __ldg(p.inv_freq + 8 * t + i), sn, cn);
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
  };
  if (nseg > 0) prep_q(s_q0, s_sb[0]);

  int cs = 0, ck = 0;  // compute iterator
  for (int j = 0; j < ntiles; ++j) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (j < 16) ATT_STAMP(4 + 2 * j);
    gather((j + STAGES - 1) % STAGES);
    copy_pages();
    cp_commit();

    const uint8_t* st = smem + (j % STAGES) * ATT_STAGE_BYTES;
    const int b = s_sb[cs];
    if (ck == 0 && cs > 0) prep_q(reinterpret_cast<const __nv_bfloat16*>(st + 16384), b);

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

    if (j < 16) ATT_STAMP(5 + 2 * j);
    if (ck == s_stiles[cs] - 1) {
      ATT_STAMP(40);
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
            att_sincos((float)pos * __ldg(p.inv_freq + d0 / 2 + i), sn, cn);
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
      // warps -> scratch [warp][h][row][66]: m, l, acc[64], in this tile's stage (16.9 of its
      // 18.9 KB; the stage is reissued only after the next iteration's barrier)
      __syncthreads();
      float* s_scr = reinterpret_cast<float*>(smem + (j % STAGES) * ATT_STAGE_BYTES);
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
        att_store_partial(p.ws_acc, p.ws_ml, slot, h, r, dc, M, L, o);
      }
      __threadfence();
      __syncthreads();
      // Merge: the last of each group of up to 16 partials merges its group; the last group
      // (or, with one group, the last partial) merges into the output.
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      const int c = i_last - i_first + 1, ng = (c + ATT_GROUP - 1) / ATT_GROUP;
      const int gi = (cta - i_first) / ATT_GROUP, gs = min(ATT_GROUP, c - gi * ATT_GROUP);
      int* cnt_final = p.counters + b * gridDim.y + pair;
      int* cnt_group = p.counters + B * gridDim.y + (size_t)(b * gridDim.y + pair) * p.max_groups + gi;
      const size_t g_slot0 = ((size_t)pair * B + b) * p.max_groups;
      if (tid == 0) {
        int mode = 0;
        if (ng == 1) {
          if (atomicAdd(cnt_final, 1) == c - 1) { *cnt_final = 0; mode = 2; }
        } else if (atomicAdd(cnt_group, 1) == gs - 1) {
          *cnt_group = 0; mode = 1;
        }
        s_last = mode;
      }
      __syncthreads();
      ATT_STAMP(41);
      const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
      if (s_last == 1) {
        __threadfence();
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        att_merge(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + gi * ATT_GROUP + b, gs, h, r, dc, M, L, o);
        att_store_partial(p.ws2_acc, p.ws2_ml, g_slot0 + gi, h, r, dc, M, L, o);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          int mode = 0;
          if (atomicAdd(cnt_final, 1) == ng - 1) { *cnt_final = 0; mode = 3; }
          s_last = mode;
        }
        __syncthreads();
      }
      if (s_last >= 2) {
        __threadfence();
        const int qh = (2 * pair + h) * 8 + r;
        const uint4 gw = *reinterpret_cast<const uint4*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + qh * 64 + 8 * dc);
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (s_last == 2) att_merge(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, r, dc, M, L, o);
        else att_merge(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, r, dc, M, L, o);
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
      ATT_STAMP(42);
      ++cs; ck = 0;
    } else {
      ++ck;
    }
  }
  cp_wait<0>();
  ATT_STAMP(61);
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 62] = att_gtime();
}
"""

ATTN_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <int S>
static void launch_dec(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * ATT_STAGE_BYTES;
  auto kern = attn_dec_kernel<S>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
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
  p.ws2_acc = ws2_acc.data_ptr<float>();
  p.ws2_ml = ws2_ml.data_ptr<float>();
  p.counters = counters.data_ptr<int32_t>();
  p.max_groups = (int)max_groups;
  p.prof = prof.numel() > 0 ? reinterpret_cast<long long*>(prof.data_ptr()) : nullptr;
  p.qkv_w = (int)qkv.size(-1);
  p.pt_stride = (int)page_table.stride(0);
  p.B = (int)B; p.HQ = (int)HQ; p.HKV = (int)HKV; p.page_shift = (int)page_shift; p.window = (int)window;
  p.rope = (int)rope; p.n_ctas = (int)n_ctas;
  p.qk_factor = (float)qk_factor; p.sm_scale_log2 = (float)sm_scale_log2; p.eps = (float)eps;
  int pairs = (int)HKV / 2;
  if (stages == 3) launch_dec<3>(p, (int)n_ctas, pairs);
  else if (stages == 4) launch_dec<4>(p, (int)n_ctas, pairs);
  else if (stages == 5) launch_dec<5>(p, (int)n_ctas, pairs);
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
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages);
"""

_ATTN_MODULES = {}
_ATTN_COUNTERS = {}
_ATTN_EMPTY = {}


def _no_prof(device):
    t = _ATTN_EMPTY.get(device)
    if t is None:
        t = _ATTN_EMPTY[device] = torch.empty(0, dtype=torch.int64, device=device)
    return t


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


def attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=264, stages=4, min_keys=256, prof=None):
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
    max_groups = n // 16 + 2
    key = (device, batch * pairs * (1 + max_groups))
    counters = _ATTN_COUNTERS.get(key)
    if counters is None:
        counters = _ATTN_COUNTERS[key] = torch.zeros(key[1], dtype=torch.int32, device=device)
    slots = pairs * (n + batch)
    ws_acc = torch.empty((slots, 2, 8, 64), dtype=torch.float32, device=device)
    ws_ml = torch.empty((slots, 2, 8, 2), dtype=torch.float32, device=device)
    ws2_acc = torch.empty((pairs * batch * max_groups, 2, 8, 64), dtype=torch.float32, device=device)
    ws2_ml = torch.empty((pairs * batch * max_groups, 2, 8, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, hq * 64), dtype=torch.bfloat16, device=device)
    attn_cuda_module().attn_dec(
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], out, ws_acc, ws_ml, ws2_acc, ws2_ml, counters, prof if prof is not None else _no_prof(device), max_groups, batch, hq, hkv, page.bit_length() - 1,
        window, int(bool(cfg["rope_theta"])), cfg["qk_scale_factor"], cfg["sm_scale"] * 1.4426950408889634, 1e-5,
        n, stages)
    return out
