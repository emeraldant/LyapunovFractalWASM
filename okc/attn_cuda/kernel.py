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
  int qkv_w, pt_stride, B, HQ, HKV, page_shift, window, rope, n_ctas, max_groups, L;
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

// Thread-block cluster helpers (CL kernels): rank, cluster barrier, DSMEM mapping, mbarriers.
__device__ __forceinline__ void att_cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t att_mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void att_mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void att_mbar_arrive_remote(uint32_t cl_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(cl_addr) : "memory");
}
__device__ __forceinline__ void att_mbar_wait0(uint32_t addr) {
  asm volatile("{\n .reg .pred P;\n WAIT%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n"
               " @!P bra WAIT%=;\n}\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ float2 att_ld_cl2(uint32_t a) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float4 att_ld_cl4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}

// CL: CTAs form clusters of 8 (consecutive blockIdx.x). A request's CTAs inside one cluster merge
// their partials over DSMEM into the request's last CTA there (remote mbarrier arrive, no global
// stores, fences or atomics); only one partial per cluster goes through global memory, and a
// request inside one cluster never touches it. Each CTA keeps up to 2 segment partials in
// shared memory slots ([2][2 heads][8 rows][72]: M, L, 6 pad, acc[64]) until the cluster's
// final barrier. A request whose first CTA would need a third slot takes the per-CTA global path.
#define ATT_SLOT_FLOATS (2 * 8 * 72)

template <int STAGES, bool CL>
__global__ void __launch_bounds__(128) attn_dec_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  // per stage: K [64][128] | V [64][128] | Q [16][64] bf16 | new K, V [2][2][64] bf16
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];
  __shared__ __align__(16) __nv_bfloat16 s_q0[16 * 64];
  __shared__ float s_knew[2][64];
  __shared__ __align__(8) unsigned long long s_mbar[2];

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
  if (!CL && cta >= N) return;   // CL: idle CTAs still take part in the cluster barriers
  const long long r0 = cta < N ? T * cta / N : T, r1 = cta < N ? T * (cta + 1) / N : T;

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
  float* s_slots = reinterpret_cast<float*>(smem + STAGES * ATT_STAGE_BYTES);
  // The request holding virtual key u (u < T).
  auto req_of = [&](long long u) { int b = 0; while (b + 1 < B && s_pref[b + 1] <= u) ++b; return b; };
  // Cluster mode for request b: every cluster it touches can merge it over DSMEM (only its first
  // CTA can hold it at a segment index >= 2).
  auto cl_mode = [&](int b, int i_first, int i_last) {
    if (!CL) return false;
    if (min(i_last, (i_first / 8) * 8 + 7) == i_first) return true;
    return b - req_of(T * i_first / N) < 2;
  };
  if (CL) {
    if (tid == 0) {
      for (int s = 0; s < min(nseg, 2); ++s) {
        const int b = s_sb[s];
        const int i_first = att_cta_of(s_pref[b], T, N), i_last = att_cta_of(s_pref[b + 1] - 1, T, N);
        const int k = cta / 8, cl_lo = max(i_first, 8 * k), cl_hi = min(i_last, 8 * k + 7);
        const int cnt = (cta == cl_hi && cl_mode(b, i_first, i_last)) ? cl_hi - cl_lo : 0;
        att_mbar_init((uint32_t)__cvta_generic_to_shared(&s_mbar[s]), (uint32_t)max(cnt, 1));
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    att_cl_sync();
  }
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
      float mM, mL, mo[8];
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
        mM = M; mL = L;
#pragma unroll
        for (int i = 0; i < 8; ++i) mo[i] = o[i];
      }
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      const int h = tid >> 6, r = (tid >> 3) & 7, dc = tid & 7;
      // out = RMSNorm(o / L * v_scale) * sigmoid(gate) for this thread's (head, row, 8 d)
      auto emit = [&](float M, float L, float o[8]) {
        const int qh = (2 * pair + h) * 8 + r;
        const uint4 gw = *reinterpret_cast<const uint4*>(p.qkv + (size_t)b * p.qkv_w + (HQ + 2 * HKV) * 64 + qh * 64 + 8 * dc);
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
      };
      if (cl_mode(b, i_first, i_last)) {
        const int k = cta / 8, cl_lo = max(i_first, 8 * k), cl_hi = min(i_last, 8 * k + 7);
        if (cta != cl_hi) {
          // contributor: partial -> own slot cs, then arrive on the owner's barrier 0
          float* sl = s_slots + (size_t)cs * ATT_SLOT_FLOATS + (h * 8 + r) * 72;
          if (dc == 0) { sl[0] = mM; sl[1] = mL; }
#pragma unroll
          for (int i = 0; i < 8; ++i) sl[8 + 8 * dc + i] = mo[i];
          asm volatile("fence.acq_rel.cluster;\n" ::: "memory");
          __syncthreads();
          if (tid == 0)
            att_mbar_arrive_remote(att_mapa((uint32_t)__cvta_generic_to_shared(&s_mbar[0]), (uint32_t)(cl_hi & 7)));
        } else {
          float M = mM, L = mL, o[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = mo[i];
          if (cl_hi > cl_lo) {
            att_mbar_wait0((uint32_t)__cvta_generic_to_shared(&s_mbar[cs]));
            const int idx_lo = cl_lo > i_first ? 0 : b - req_of(T * cl_lo / N);
            for (int jc = cl_lo; jc < cl_hi; ++jc) {
              const int idx = jc == cl_lo ? idx_lo : 0;
              const uint32_t base = att_mapa((uint32_t)__cvta_generic_to_shared(
                  s_slots + (size_t)idx * ATT_SLOT_FLOATS + (h * 8 + r) * 72), (uint32_t)(jc & 7));
              const float2 ml = att_ld_cl2(base);
              const float4 a0 = att_ld_cl4(base + 32 + 32 * dc), a1 = att_ld_cl4(base + 48 + 32 * dc);
              const float mn = fmaxf(M, ml.x);
              const float fa = ex2(M - mn), fb = ex2(ml.x - mn);
              M = mn;
              L = L * fa + ml.y * fb;
              o[0] = o[0] * fa + a0.x * fb; o[1] = o[1] * fa + a0.y * fb; o[2] = o[2] * fa + a0.z * fb;
              o[3] = o[3] * fa + a0.w * fb; o[4] = o[4] * fa + a1.x * fb; o[5] = o[5] * fa + a1.y * fb;
              o[6] = o[6] * fa + a1.z * fb; o[7] = o[7] * fa + a1.w * fb;
            }
          }
          const int k0 = i_first / 8, nclu = i_last / 8 - k0 + 1;
          if (nclu == 1) {
            emit(M, L, o);
          } else {
            const size_t base2 = (size_t)gridDim.y * (p.n_ctas + B) + (size_t)pair * (p.n_ctas / 8 + B) + b;
            att_store_partial(p.ws_acc, p.ws_ml, base2 + k, h, r, dc, M, L, o);
            __threadfence();
            __syncthreads();
            int* cnt_final = p.counters + b * gridDim.y + pair;
            if (tid == 0) {
              int last = atomicAdd(cnt_final, 1) == nclu - 1;
              if (last) *cnt_final = 0;
              s_last = last;
            }
            __syncthreads();
            if (s_last) {
              __threadfence();
              float M2 = -ATT_INF, L2 = 0.f, o2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
              att_merge(p.ws_acc, p.ws_ml, base2 + k0, nclu, h, r, dc, M2, L2, o2);
              emit(M2, L2, o2);
            }
          }
        }
      } else {
      {
        const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
        att_store_partial(p.ws_acc, p.ws_ml, slot, h, r, dc, mM, mL, mo);
      }
      __threadfence();
      __syncthreads();
      // Merge: the last of each group of up to 16 partials merges its group; the last group
      // (or, with one group, the last partial) merges into the output.
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
        float M = -ATT_INF, L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (s_last == 2) att_merge(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, r, dc, M, L, o);
        else att_merge(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, r, dc, M, L, o);
        emit(M, L, o);
      }
      }
      __syncthreads();
      ATT_STAMP(42);
      ++cs; ck = 0;
    } else {
      ++ck;
    }
  }
  cp_wait<0>();
  if (CL) att_cl_sync();   // contributors' slots stay readable until every owner has merged
  ATT_STAMP(61);
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 62] = att_gtime();
}
"""

ATTN_VER_SRC = r"""
// ---------------------------------------------------------------------------------------
// Verify blocks (L = 2..8 new tokens per request, causal inside the block): the same work split,
// gather and merge as attn_dec_kernel, with 8 L rows per KV head. Warp w owns KV head w >> 1
// and rows 32 (w & 1) .. + 31 (4 tokens x 8 query heads) over all of a tile's keys, so its
// online-softmax state is final for its rows (no cross-warp merge). S = Q K^T in E4M3 (queries
// rounded to E4M3 as FA3 does, the cached K bytes straight from shared memory as the B
// operand: mma m16n8k32); O^T = V^T P^T in f16 as in the decode kernel. The block's own keys
// are normalized, stored, and attended as one extra 16-key step with the causal mask.

__device__ __forceinline__ void mma_e4m3(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t att_e4m3x4(float a, float b, float c, float d) {
  __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Partials of the verify kernel: acc [slot][2 heads][64 d][64 rows], ml [slot][2][64 rows][2].
__device__ __forceinline__ void att_merge_v(const float* ws_acc, const float* ws_ml, size_t slot0, int count, int h,
                                            int row, float& M, float& Lsum, float o[64]) {
  for (int k = 0; k < count; ++k) {
    const size_t slot = slot0 + k;
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(ws_ml) + (slot * 2 + h) * 64 + row);
    const float* wa = ws_acc + (slot * 2 + h) * 4096 + row;
    const float mn = fmaxf(M, ml.x);
    const float ms = mn == -ATT_INF ? 0.f : mn;
    const float fa = ex2(M - ms), fb = ex2(ml.x - ms);
    M = mn;
    Lsum = Lsum * fa + ml.y * fb;
#pragma unroll
    for (int d0 = 0; d0 < 64; d0 += 16) {
      float a[16];
#pragma unroll
      for (int d = 0; d < 16; ++d) a[d] = __ldcg(wa + (d0 + d) * 64);
#pragma unroll
      for (int d = 0; d < 16; ++d) o[d0 + d] = o[d0 + d] * fa + a[d] * fb;
    }
  }
}

__device__ __forceinline__ void att_store_partial_v(float* ws_acc, float* ws_ml, size_t slot, int h, int row,
                                                    float M, float Lsum, const float o[64]) {
  float* wa = ws_acc + (slot * 2 + h) * 4096 + row;
#pragma unroll
  for (int d = 0; d < 64; ++d) wa[d * 64] = o[d];
  reinterpret_cast<float2*>(ws_ml)[(slot * 2 + h) * 64 + row] = make_float2(M, Lsum);
}

template <int STAGES>
__global__ void __launch_bounds__(128, 3) attn_ver_kernel(AttnParams p) {
  extern __shared__ __align__(128) uint8_t smem[];   // per stage: K [64][128] | V [64][128]
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ, L = p.L;
  const int hw = warp >> 1, rh = warp & 1;   // this warp's KV head (of the pair) and row half
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 63] = att_gtime();
  ATT_STAMP(0);

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
  const int rs = tid >> 3, gr = tid & 7;
  const size_t W = p.qkv_w;

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
  // The request's gate rows and new K / V rows to L2 when its first tile is gathered.
  auto prefetch_req = [&](int b) {
    if (tid < 16 * L) {
      const int j = tid >> 4, c = tid & 15;
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + 2 * HKV) * 64 + pair * 1024);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + c * 128));
      if (c < 4) {
        const uint8_t* kp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + (c >> 1) * HKV) * 64 + pair * 128);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(kp + (c & 1) * 128));
      }
    }
  };
  auto gather = [&](int stage) {
    if (is_s < nseg) {
      const int s = is_s, k = is_k, b = s_sb[s];
      const uint32_t st = smem_u32 + stage * 16384;
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
      if (k == 0 && s > 0) prefetch_req(b);
    }
    advance(is_s, is_k, is_t);
  };

  for (int m = 0; m < STAGES - 1; ++m) copy_pages();
  if (nseg > 0) prefetch_req(s_sb[0]);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int m = 0; m < STAGES - 1; ++m) { gather(m); copy_pages(); cp_commit(); }
  ATT_STAMP(3);

  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale * p.qk_factor;  // queries are E4M3 of q / factor

  // Per thread: rows (mi, hf) = warp rows 16 mi + 8 hf + g: token 4 rh + 2 mi + hf, query head g.
  uint32_t qw[2][2][4];
  float m_r[2][2], l_r[2][2], o[4][4][4];
  int seq = 0;
  auto prep_q = [&](int b) {
    seq = s_first[b] + s_n[b];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int j = 4 * rh + 2 * mi + hf;
        float x[16];
        if (j < L) {
          const uint4* src = reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + j) * W + (pair * 16 + 8 * hw + g) * 64 + 16 * t);
          uint4 w0 = __ldg(src), w1 = __ldg(src + 1);
          const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
          const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&w1);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float2 f0 = __bfloat1622float2(h0[i]), f1 = __bfloat1622float2(h1[i]);
            x[2 * i] = f0.x; x[2 * i + 1] = f0.y; x[8 + 2 * i] = f1.x; x[8 + 2 * i + 1] = f1.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) x[i] = 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float sc = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] *= sc;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float sn, cn;
            att_sincos((float)(seq + j) * __ldg(p.inv_freq + 8 * t + i), sn, cn);
            float e = x[2 * i], od = x[2 * i + 1];
            x[2 * i] = e * cn - od * sn;
            x[2 * i + 1] = od * cn + e * sn;
          }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) qw[mi][hf][c] = att_e4m3x4(x[4 * c], x[4 * c + 1], x[4 * c + 2], x[4 * c + 3]);
        m_r[mi][hf] = -ATT_INF; l_r[mi][hf] = 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) o[i][ni][0] = o[i][ni][1] = o[i][ni][2] = o[i][ni][3] = 0.f;
  };
  if (nseg > 0) prep_q(s_sb[0]);

  // One 16-key step of this warp over keys kb .. kb + 15 of a K / V tile. nvalid: valid keys;
  // mode 1: the block's own keys (causal); wpos >= 0: absolute position of key kb, for the
  // window's lower edge (row j sees positions > seq + j - window).
  auto substep = [&](const uint8_t* sK, const uint8_t* sV, int kb, int nvalid, int mode, int wpos) {
    float s[2][2][4];
    uint4 kw[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int key = kb + 8 * nt + g;
      kw[nt] = *reinterpret_cast<const uint4*>(sK + key * 128 + (((4 * hw + t) ^ att_swz(key)) << 4));
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        s[mi][nt][0] = s[mi][nt][1] = s[mi][nt][2] = s[mi][nt][3] = 0.f;
        mma_e4m3(s[mi][nt], qw[mi][0][0], qw[mi][1][0], qw[mi][0][1], qw[mi][1][1], kw[nt].x, kw[nt].y);
        mma_e4m3(s[mi][nt], qw[mi][0][2], qw[mi][1][2], qw[mi][0][3], qw[mi][1][3], kw[nt].z, kw[nt].w);
      }
    float al[2][2];
    bool grow = false;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int j = 4 * rh + 2 * mi + hf;
        float mt = -ATT_INF;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int k = 8 * nt + 2 * t + e;
            bool ok = k < nvalid;
            if (mode == 1) ok = ok && k <= j;
            if (wpos >= 0) ok = ok && wpos + k > seq + j - p.window;
            const float x = ok ? s[mi][nt][2 * hf + e] * c_log2 : -ATT_INF;
            s[mi][nt][2 * hf + e] = x;
            mt = fmaxf(mt, x);
          }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
        const float mn = fmaxf(m_r[mi][hf], mt);
        const float ms = mn == -ATT_INF ? 0.f : mn;
        al[mi][hf] = ex2(m_r[mi][hf] - ms);
        grow |= mn > m_r[mi][hf];
        m_r[mi][hf] = mn;
        float ls = 0.f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float pv = ex2(s[mi][nt][2 * hf + e] - ms);
            s[mi][nt][2 * hf + e] = pv;
            ls += pv;
          }
        l_r[mi][hf] = l_r[mi][hf] * al[mi][hf] + ls;
      }
    if (__any_sync(0xffffffffu, grow)) {
      // O^T column (row) 8 ni + 2t (+1) belongs to S row g = 2t (+1) of (mi, hf) = (ni >> 1, ni & 1)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const float a0 = __shfl_sync(0xffffffffu, al[ni >> 1][ni & 1], 8 * t);
        const float a1 = __shfl_sync(0xffffffffu, al[ni >> 1][ni & 1], 8 * t + 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) { o[i][ni][0] *= a0; o[i][ni][1] *= a1; o[i][ni][2] *= a0; o[i][ni][3] *= a1; }
      }
    }
    uint32_t pb0[4], pb1[4];
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int mi = ni >> 1, hf = ni & 1;
      pb0[ni] = pack_h2(s[mi][0][2 * hf], s[mi][0][2 * hf + 1]);
      pb1[ni] = pack_h2(s[mi][1][2 * hf], s[mi][1][2 * hf + 1]);
    }
    const int vo = ((4 * hw + (g >> 1)) << 4) | ((g & 1) << 3);
    const uint8_t* sVb = sV + kb * 128;
    uint2 va = *reinterpret_cast<const uint2*>(sVb + (2 * t) * 128 + (vo ^ (att_swz(2 * t) << 4)));
    uint2 vb = *reinterpret_cast<const uint2*>(sVb + (2 * t + 1) * 128 + (vo ^ (att_swz(2 * t + 1) << 4)));
    uint2 vc = *reinterpret_cast<const uint2*>(sVb + (2 * t + 8) * 128 + (vo ^ (att_swz(2 * t + 8) << 4)));
    uint2 vd = *reinterpret_cast<const uint2*>(sVb + (2 * t + 9) * 128 + (vo ^ (att_swz(2 * t + 9) << 4)));
    uint32_t lo01 = __byte_perm(va.x, vb.x, 0x5140), lo23 = __byte_perm(va.x, vb.x, 0x7362);
    uint32_t hi01 = __byte_perm(va.y, vb.y, 0x5140), hi23 = __byte_perm(va.y, vb.y, 0x7362);
    uint32_t clo01 = __byte_perm(vc.x, vd.x, 0x5140), clo23 = __byte_perm(vc.x, vd.x, 0x7362);
    uint32_t chi01 = __byte_perm(vc.y, vd.y, 0x5140), chi23 = __byte_perm(vc.y, vd.y, 0x7362);
    uint32_t A0[4] = {cvt_lo(lo01), cvt_hi(lo01), cvt_lo(lo23), cvt_hi(lo23)};
    uint32_t A1[4] = {cvt_lo(hi01), cvt_hi(hi01), cvt_lo(hi23), cvt_hi(hi23)};
    uint32_t A2[4] = {cvt_lo(clo01), cvt_hi(clo01), cvt_lo(clo23), cvt_hi(clo23)};
    uint32_t A3[4] = {cvt_lo(chi01), cvt_hi(chi01), cvt_lo(chi23), cvt_hi(chi23)};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) mma_f16(o[i][ni], A0[i], A1[i], A2[i], A3[i], pb0[ni], pb1[ni]);
  };

  int cs = 0, ck = 0;
  for (int j = 0; j < ntiles; ++j) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (j < 16) ATT_STAMP(4 + 2 * j);
    gather((j + STAGES - 1) % STAGES);
    copy_pages();
    cp_commit();

    uint8_t* st = smem + (j % STAGES) * 16384;
    const int b = s_sb[cs];
    if (ck == 0 && cs > 0) prep_q(b);
    const int u0 = s_su0[cs] + ck * ATT_TILE;
    const int nv_tile = s_suc[cs] - u0;
    const int pos0 = s_first[b] + u0;
    const int wpos = (p.window > 0 && pos0 < seq + L - p.window) ? pos0 : -1;
#pragma unroll 1
    for (int sub = 0; sub < 4; ++sub) {
      const int nv = min(nv_tile - 16 * sub, 16);
      if (nv > 0) substep(st, st + 8192, 16 * sub, nv, 0, wpos >= 0 ? wpos + 16 * sub : -1);
    }
    if (j < 16) ATT_STAMP(5 + 2 * j);

    if (ck == s_stiles[cs] - 1) {
      ATT_STAMP(40);
      if (s_snew[cs]) {
        // The block's own keys: normalized (+ RoPE), FP8, stored to the cache and to rows 0..15
        // of this (consumed) stage, then one causal step. Thread: token tid >> 4, head
        // (tid >> 3) & 1, d 8 (tid & 7) .. + 7.
        __syncthreads();
        const int jt = tid >> 4, hh = (tid >> 3) & 1, dc = tid & 7;
        float kx[8], vx[8];
        if (jt < L) {
          const uint4 kr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + 2 * pair + hh) * 64 + 8 * dc));
          const uint4 vr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + HKV + 2 * pair + hh) * 64 + 8 * dc));
          const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kr);
          const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vr);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float2 a = __bfloat1622float2(kh[i]), c = __bfloat1622float2(vh[i]);
            kx[2 * i] = a.x; kx[2 * i + 1] = a.y; vx[2 * i] = c.x; vx[2 * i + 1] = c.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) { kx[i] = 0.f; vx[i] = 0.f; }
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += kx[i] * kx[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        ss += __shfl_xor_sync(0xffffffffu, ss, 4);
        const float inv = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) kx[i] *= inv;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float sn, cn;
            att_sincos((float)(seq + jt) * __ldg(p.inv_freq + 4 * dc + i), sn, cn);
            float e = kx[2 * i], od = kx[2 * i + 1];
            kx[2 * i] = e * cn - od * sn;
            kx[2 * i + 1] = od * cn + e * sn;
          }
        }
        const float ik = 1.f / k_scale, iv = 1.f / v_scale;
        uint2 kq, vq;
        kq.x = att_e4m3x4(kx[0] * ik, kx[1] * ik, kx[2] * ik, kx[3] * ik);
        kq.y = att_e4m3x4(kx[4] * ik, kx[5] * ik, kx[6] * ik, kx[7] * ik);
        vq.x = att_e4m3x4(vx[0] * iv, vx[1] * iv, vx[2] * iv, vx[3] * iv);
        vq.y = att_e4m3x4(vx[4] * iv, vx[5] * iv, vx[6] * iv, vx[7] * iv);
        if (jt < L) {
          const long long loc = p.cache_loc[(size_t)b * L + jt];
          const size_t off = ((size_t)loc * HKV + 2 * pair + hh) * 64 + 8 * dc;
          *reinterpret_cast<uint2*>(p.k_cache + off) = kq;
          *reinterpret_cast<uint2*>(p.v_cache + off) = vq;
        } else {
          kq = make_uint2(0u, 0u); vq = make_uint2(0u, 0u);
        }
        // rows jt (0..7) and jt + 8 (zero) of the stage
        const int so = ((((4 * hh + (dc >> 1)) ^ att_swz(jt)) << 4) | ((dc & 1) << 3));
        *reinterpret_cast<uint2*>(st + jt * 128 + so) = kq;
        *reinterpret_cast<uint2*>(st + 8192 + jt * 128 + so) = vq;
        const int so8 = ((((4 * hh + (dc >> 1)) ^ att_swz(jt + 8)) << 4) | ((dc & 1) << 3));
        *reinterpret_cast<uint2*>(st + (jt + 8) * 128 + so8) = make_uint2(0u, 0u);
        *reinterpret_cast<uint2*>(st + 8192 + (jt + 8) * 128 + so8) = make_uint2(0u, 0u);
        __syncthreads();
        substep(st, st + 8192, 0, L, 1, -1);
      }
      // this warp's rows -> partial slot (cta + b): acc [d][row], ml [row]
      const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
      {
        float* wa = p.ws_acc + (slot * 2 + hw) * 4096 + 32 * rh;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int rr = 8 * ni + 2 * t;
            *reinterpret_cast<float2*>(wa + (8 * g + i) * 64 + rr) = make_float2(o[i][ni][0], o[i][ni][1]);
            *reinterpret_cast<float2*>(wa + (8 * g + 4 + i) * 64 + rr) = make_float2(o[i][ni][2], o[i][ni][3]);
          }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            float l = l_r[mi][hf];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            if (t == 0)
              reinterpret_cast<float2*>(p.ws_ml)[(slot * 2 + hw) * 64 + 32 * rh + 16 * mi + 8 * hf + g] = make_float2(m_r[mi][hf], l);
          }
      }
      __threadfence();
      __syncthreads();
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
      const int h = tid >> 6, row = tid & 63;
      if (s_last == 1) {
        __threadfence();
        float M = -ATT_INF, Ls = 0.f, oo[64];
#pragma unroll
        for (int d = 0; d < 64; ++d) oo[d] = 0.f;
        att_merge_v(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + gi * ATT_GROUP + b, gs, h, row, M, Ls, oo);
        att_store_partial_v(p.ws2_acc, p.ws2_ml, g_slot0 + gi, h, row, M, Ls, oo);
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
        float M = -ATT_INF, Ls = 0.f, oo[64];
#pragma unroll
        for (int d = 0; d < 64; ++d) oo[d] = 0.f;
        if (s_last == 2) att_merge_v(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, row, M, Ls, oo);
        else att_merge_v(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, row, M, Ls, oo);
        const int jt = row >> 3, qh = (2 * pair + h) * 8 + (row & 7);
        if (jt < L) {
          const float sc = v_scale / Ls;
          float ss = 0.f;
#pragma unroll
          for (int d = 0; d < 64; ++d) { oo[d] *= sc; ss += oo[d] * oo[d]; }
          const float inv = rsqrtf(ss / 64.f + p.eps);
          const size_t tok = (size_t)b * L + jt;
          const uint4* gp = reinterpret_cast<const uint4*>(p.qkv + tok * W + (HQ + 2 * HKV) * 64 + qh * 64);
          uint4* op = reinterpret_cast<uint4*>(p.out + tok * HQ * 64 + qh * 64);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint4 gw = gp[q];
            const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw);
            uint4 ow;
            __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 gf = __bfloat1622float2(gh[i]);
              oh[i] = __floats2bfloat162_rn(oo[8 * q + 2 * i] * inv / (1.f + __expf(-gf.x)),
                                            oo[8 * q + 2 * i + 1] * inv / (1.f + __expf(-gf.y)));
            }
            op[q] = ow;
          }
        }
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

ATTN_VWG_SRC = r"""
// ---------------------------------------------------------------------------------------
// Verify blocks on wgmma (attn_vwg_kernel). mma.sync could not keep up with 64 rows per KV head:
// Hopper has no native E4M3 mma.sync (ptxas emulates it as two f16 HMMAs plus unpacks), and a
// 16-key step was 596 instructions. Here a CTA = two warpgroups, one per KV head of the pair:
//   S = Q K^T: wgmma m64n64k32 E4M3, A = the head's 64 rows (8 tokens x 8 query heads) in
//     registers, B = the cached K tile straight from shared memory (128 B swizzle, K-major:
//     the gather writes rows in that layout; head h's k32 steps start at 64 h + 32 c).
//   O^T = V^T P^T: wgmma m64n64k16 f16, A = V^T from registers (warp w holds d = 8 g + w and
//     8 g + 4 + w: the decode kernel's byte pairing), B = P in shared memory ([row][key] f16,
//     128 B swizzle). The online-softmax rescale goes through shared memory (alpha per row).

__device__ __forceinline__ void wg_e4m3_64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
               "%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
}

__device__ __forceinline__ void wg_f16_64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
               "%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
}

__device__ __forceinline__ uint64_t att_desc128(uint32_t addr) {
  // K-major, 128 B swizzle: 8-row atoms 1024 B apart (SBO); LBO unused (16)
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}

// Partials of attn_vwg_kernel, row-major: acc [slot][2 heads][64 rows][64 d], ml [slot][2][64][2].
// The merge gives each of 256 threads one (head, row, half of d): 8 x 16 B loads per partial,
// the next partial's loads issued before the current one's math.
__device__ __forceinline__ void att_merge_h(const float* ws_acc, const float* ws_ml, size_t slot0, int count, int h,
                                            int row, int half, float& M, float& Lsum, float o[32]) {
  for (int k = 0; k < count; k += 2) {
    float2 ml[2];
    float4 cur[2][8];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const size_t slot = slot0 + min(k + u, count - 1);
      ml[u] = __ldcg(reinterpret_cast<const float2*>(ws_ml) + (slot * 2 + h) * 64 + row);
      const float4* wa = reinterpret_cast<const float4*>(ws_acc + ((slot * 2 + h) * 64 + row) * 64 + 32 * half);
#pragma unroll
      for (int i = 0; i < 8; ++i) cur[u][i] = __ldcg(wa + i);
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      if (k + u < count) {
        const float mn = fmaxf(M, ml[u].x);
        const float ms = mn == -ATT_INF ? 0.f : mn;
        const float fa = ex2(M - ms), fb = ex2(ml[u].x - ms);
        M = mn;
        Lsum = Lsum * fa + ml[u].y * fb;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          o[4 * i] = o[4 * i] * fa + cur[u][i].x * fb; o[4 * i + 1] = o[4 * i + 1] * fa + cur[u][i].y * fb;
          o[4 * i + 2] = o[4 * i + 2] * fa + cur[u][i].z * fb; o[4 * i + 3] = o[4 * i + 3] * fa + cur[u][i].w * fb;
        }
      }
    }
  }
}

__device__ __forceinline__ void att_store_partial_h(float* ws_acc, float* ws_ml, size_t slot, int h, int row, int half,
                                                    float M, float Lsum, const float o[32]) {
  float4* wa = reinterpret_cast<float4*>(ws_acc + ((slot * 2 + h) * 64 + row) * 64 + 32 * half);
#pragma unroll
  for (int i = 0; i < 8; ++i) wa[i] = make_float4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
  if (half == 0) reinterpret_cast<float2*>(ws_ml)[(slot * 2 + h) * 64 + row] = make_float2(M, Lsum);
}

// The cooperative merge's work for one CTA (rank of c): its slice of the 128 (head, row) rows.
// Out of line: it runs once per segment, and inlined its registers spilled the main loop.
__device__ __forceinline__ void att_coop_merge_v(const AttnParams& p, int rank, int c, size_t s0, int b, int pair, int L,
                                              float v_scale) {
  const int tid = threadIdx.x;
  const int HQ = p.HQ, HKV = p.HKV;
  const size_t W = p.qkv_w;
  const int r_lo = rank * 128 / c, r_hi = (rank + 1) * 128 / c;
  const int d4 = tid & 15;
  for (int r0 = r_lo; r0 < r_hi; r0 += 16) {
    const int hr = r0 + (tid >> 4);
    const bool live = hr < r_hi;
    float M = -ATT_INF, Ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k0 = 0; k0 < c; k0 += 2) {   // 2 partials in flight per round
      float4 a[2];
      float2 ml[2];
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        if (k0 + k < c && live) {
          a[k] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc) + (s0 + k0 + k) * 2048 + hr * 16 + d4);
          ml[k] = __ldcg(reinterpret_cast<const float2*>(p.ws_ml) + (s0 + k0 + k) * 128 + hr);
        }
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        if (k0 + k < c && live) {
          const float mn = fmaxf(M, ml[k].x);
          const float ms = mn == -ATT_INF ? 0.f : mn;
          const float fa = ex2(M - ms), fb = ex2(ml[k].x - ms);
          M = mn;
          Ls = Ls * fa + ml[k].y * fb;
          acc.x = acc.x * fa + a[k].x * fb; acc.y = acc.y * fa + a[k].y * fb;
          acc.z = acc.z * fa + a[k].z * fb; acc.w = acc.w * fa + a[k].w * fb;
        }
      }
    }
    const float sc = live ? v_scale / Ls : 0.f;
    acc.x *= sc; acc.y *= sc; acc.z *= sc; acc.w *= sc;
    float ss = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    ss += __shfl_xor_sync(0xffffffffu, ss, 8);
    const float inv = rsqrtf(ss / 64.f + p.eps);
    const int hh = hr >> 6, row = hr & 63, jt = row >> 3, qh = (2 * pair + hh) * 8 + (row & 7);
    if (live && jt < L) {
      const size_t tok = (size_t)b * L + jt;
      const uint2 gw = *reinterpret_cast<const uint2*>(p.qkv + tok * W + (HQ + 2 * HKV) * 64 + qh * 64 + 4 * d4);
      const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw);
      const float2 g0 = __bfloat1622float2(gh[0]), g1 = __bfloat1622float2(gh[1]);
      uint2 ow;
      __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
      oh[0] = __floats2bfloat162_rn(acc.x * inv / (1.f + __expf(-g0.x)), acc.y * inv / (1.f + __expf(-g0.y)));
      oh[1] = __floats2bfloat162_rn(acc.z * inv / (1.f + __expf(-g1.x)), acc.w * inv / (1.f + __expf(-g1.y)));
      *reinterpret_cast<uint2*>(p.out + tok * HQ * 64 + qh * 64 + 4 * d4) = ow;
    }
  }
}

template <int STAGES, bool COOP>
__global__ void __launch_bounds__(256, 2) attn_vwg_kernel(AttnParams p) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ int s_pref[ATT_MAXB + 1], s_first[ATT_MAXB], s_n[ATT_MAXB];
  __shared__ int s_sb[ATT_MAXB], s_su0[ATT_MAXB], s_suc[ATT_MAXB], s_snew[ATT_MAXB], s_stiles[ATT_MAXB];
  __shared__ int s_nseg, s_last;
  __shared__ int s_pg[8][ATT_TILE];
  __shared__ __align__(16) float s_alpha[2][64];
  __shared__ float s_ss[256];

  const int tid = threadIdx.x, lane = tid & 31;
  const int hw = tid >> 7, wt = tid & 127, w = wt >> 5;   // warpgroup = KV head of the pair; warp in it
  const int g = lane >> 2, t = lane & 3;
  const int pair = blockIdx.y;
  const int B = p.B, HKV = p.HKV, HQ = p.HQ, L = p.L;
  const uint32_t raw_u32 = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t base = (raw_u32 + 1023u) & ~1023u;            // 1024-aligned for the swizzle atoms
  uint8_t* smem = smem_raw + (base - raw_u32);
  const uint32_t pbuf = base + STAGES * 16384 + hw * 8192;     // this head's P [64 rows][64 keys] f16
  uint8_t* pbuf_g = smem + STAGES * 16384 + hw * 8192;
  if (p.prof != nullptr && tid == 0) p.prof[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 64 + 63] = att_gtime();
  ATT_STAMP(0);

  if (tid < 32) {
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

  if (tid < 32) {
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

  const int pmask = (1 << p.page_shift) - 1;
  const size_t row_bytes = (size_t)HKV * 64;
  const int rs = tid >> 3, gr = tid & 7;     // gather: keys rs and rs + 32, granule gr
  const size_t W = p.qkv_w;

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
  auto prefetch_req = [&](int b) {
    if (tid < 16 * L) {
      const int j = tid >> 4, c = tid & 15;
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + 2 * HKV) * 64 + pair * 1024);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(gp + c * 128));
      if (c < 4) {
        const uint8_t* kp = reinterpret_cast<const uint8_t*>(p.qkv + ((size_t)b * L + j) * W + (HQ + (c >> 1) * HKV) * 64 + pair * 128);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(kp + (c & 1) * 128));
      }
    }
  };
  auto gather = [&](int stage) {
    if (is_s < nseg) {
      const int s = is_s, k = is_k, b = s_sb[s];
      const uint32_t st = base + stage * 16384;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int key = rs + 32 * q;
        const int u = s_su0[s] + k * ATT_TILE + key;
        const bool ok = u < s_suc[s];
        const int pos = s_first[b] + u;
        const long long row = ok ? ((long long)s_pg[is_t & 7][key] << p.page_shift) + (pos & pmask) : 0;
        const size_t go = row * row_bytes + pair * 128 + gr * 16;
        const uint32_t dst = st + key * 128 + ((gr ^ (key & 7)) << 4);
        cp_async16(dst, p.k_cache + go, ok ? 16u : 0u);
        cp_async16(dst + 8192, p.v_cache + go, ok ? 16u : 0u);
      }
      if (k == 0 && s > 0) prefetch_req(b);
    }
    advance(is_s, is_k, is_t);
  };

  for (int m = 0; m < STAGES - 1; ++m) copy_pages();
  if (nseg > 0) prefetch_req(s_sb[0]);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int m = 0; m < STAGES - 1; ++m) { gather(m); copy_pages(); cp_commit(); }
  ATT_STAMP(3);

  const float k_scale = *p.k_scale, v_scale = *p.v_scale;
  const float c_log2 = p.sm_scale_log2 * k_scale * p.qk_factor;   // queries are E4M3 of q / factor

  // rows of this thread in its warpgroup's 64: ra = 16 w + g (token 2w, query head g), rb = ra + 8
  // (token 2w + 1); Q E4M3 words per row: d 16 c + 4 t .. + 3, c = 0..3.
  uint32_t qa[2][4];
  float m_r[2], l_r[2], o[32];
  int seq = 0;
  auto prep_q = [&](int b) {
    seq = s_first[b] + s_n[b];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int j = 2 * w + hf;
      float x[16];
      if (j < L) {
        const __nv_bfloat16* src = p.qkv + ((size_t)b * L + j) * W + (pair * 16 + 8 * hw + g) * 64 + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint2 u = __ldg(reinterpret_cast<const uint2*>(src + 16 * c));
          const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&u);
          float2 f0 = __bfloat1622float2(hh[0]), f1 = __bfloat1622float2(hh[1]);
          x[4 * c] = f0.x; x[4 * c + 1] = f0.y; x[4 * c + 2] = f1.x; x[4 * c + 3] = f1.y;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = 0.f;
      }
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float sc = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] *= sc;
      if (p.rope) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            float sn, cn;
            att_sincos((float)(seq + j) * __ldg(p.inv_freq + 8 * c + 2 * t + i), sn, cn);
            float e = x[4 * c + 2 * i], od = x[4 * c + 2 * i + 1];
            x[4 * c + 2 * i] = e * cn - od * sn;
            x[4 * c + 2 * i + 1] = od * cn + e * sn;
          }
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) qa[hf][c] = att_e4m3x4(x[4 * c], x[4 * c + 1], x[4 * c + 2], x[4 * c + 3]);
      m_r[hf] = -ATT_INF; l_r[hf] = 0.f;
    }
  };
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;   // zeroed again after each partial is stored
  if (nseg > 0) prep_q(s_sb[0]);

  // One 64-key tile for this warpgroup's head. mode 0: cached keys (nvalid valid; wpos >= 0: the
  // window's lower edge applies, wpos = position of key 0); mode 1: the block's own keys.
  int tcount = 0;   // phase stamps 44 + 4 i .. (QK done, softmax done, PV issued, PV done) of tiles i < 4
  auto tile = [&](uint32_t st, const uint8_t* stg, int nvalid, int mode, int wpos, bool masked) {
    const int tci = tcount++;
    float sacc[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) sacc[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t dk = att_desc128(st);
    wg_e4m3_64(sacc, qa[0][0], qa[1][0], qa[0][1], qa[1][1], dk + ((uint64_t)(64 * hw) >> 4));
    wg_e4m3_64(sacc, qa[0][2], qa[1][2], qa[0][3], qa[1][3], dk + ((uint64_t)(64 * hw + 32) >> 4));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (tci < 3) ATT_STAMP(36 + tci);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (tci < 3) ATT_STAMP(44 + 4 * tci);
    // softmax on rows ra (e = 0, 1) and rb (e = 2, 3); key of sacc[4 jj + e] = 8 jj + 2 t + (e & 1)
    float al[2];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int jrow = 2 * w + hf;
      float mt = -ATT_INF;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float x = sacc[4 * jj + 2 * hf + e] * c_log2;
          if (masked) {
            const int k = 8 * jj + 2 * t + e;
            bool ok = k < nvalid;
            if (mode == 1) ok = ok && k <= jrow;
            if (wpos >= 0) ok = ok && wpos + k > seq + jrow - p.window;
            x = ok ? x : -ATT_INF;
          }
          sacc[4 * jj + 2 * hf + e] = x;
          mt = fmaxf(mt, x);
        }
      mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
      mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
      const float mn = fmaxf(m_r[hf], mt);
      const float ms = mn == -ATT_INF ? 0.f : mn;
      al[hf] = ex2(m_r[hf] - ms);
      m_r[hf] = mn;
      float ls = 0.f;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(sacc[4 * jj + 2 * hf + e] - ms);
          sacc[4 * jj + 2 * hf + e] = pv;
          ls += pv;
        }
      l_r[hf] = l_r[hf] * al[hf] + ls;
    }
    // P -> shared [row][key] f16, 128 B swizzle (rows ra, rb share ra & 7 = g); alphas per row
    const int ra = 16 * w + g;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const int co = ((jj ^ g) << 4) + 4 * t;
      *reinterpret_cast<uint32_t*>(pbuf_g + ra * 128 + co) = pack_h2(sacc[4 * jj], sacc[4 * jj + 1]);
      *reinterpret_cast<uint32_t*>(pbuf_g + (ra + 8) * 128 + co) = pack_h2(sacc[4 * jj + 2], sacc[4 * jj + 3]);
    }
    if (t == 0) { s_alpha[hw][ra] = al[0]; s_alpha[hw][ra + 8] = al[1]; }
    if (tci < 3) ATT_STAMP(45 + 4 * tci);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + hw) : "memory");
    // rescale O^T: column (row) 8 jj + 2 t (+1)
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const float2 a2 = *reinterpret_cast<const float2*>(&s_alpha[hw][8 * jj + 2 * t]);
      o[4 * jj] *= a2.x; o[4 * jj + 1] *= a2.y; o[4 * jj + 2] *= a2.x; o[4 * jj + 3] *= a2.y;
    }
    // V^T A fragments: warp w's d rows 8 g + w (g) and 8 g + 4 + w (g + 8), keys 16 kk + {2t, 2t+1, 2t+8, 2t+9}
    const uint8_t* sV = stg + 8192;
    const int vc = 4 * hw + (g >> 1), vh = (g & 1) << 3;
    const uint32_t sel = (uint32_t)w | ((uint32_t)(4 + w) << 4);
    uint32_t va[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int k0 = 16 * kk + 2 * t;
      const uint2 x0 = *reinterpret_cast<const uint2*>(sV + k0 * 128 + (((vc ^ (k0 & 7)) << 4) | vh));
      const uint2 x1 = *reinterpret_cast<const uint2*>(sV + (k0 + 1) * 128 + (((vc ^ ((k0 + 1) & 7)) << 4) | vh));
      const uint2 x2 = *reinterpret_cast<const uint2*>(sV + (k0 + 8) * 128 + (((vc ^ ((k0 + 8) & 7)) << 4) | vh));
      const uint2 x3 = *reinterpret_cast<const uint2*>(sV + (k0 + 9) * 128 + (((vc ^ ((k0 + 9) & 7)) << 4) | vh));
      va[kk][0] = e4m3x2_f16x2(__byte_perm(x0.x, x1.x, sel) & 0xFFFFu);
      va[kk][1] = e4m3x2_f16x2(__byte_perm(x0.y, x1.y, sel) & 0xFFFFu);
      va[kk][2] = e4m3x2_f16x2(__byte_perm(x2.x, x3.x, sel) & 0xFFFFu);
      va[kk][3] = e4m3x2_f16x2(__byte_perm(x2.y, x3.y, sel) & 0xFFFFu);
    }
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t dp = att_desc128(pbuf);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) wg_f16_64(o, va[kk][0], va[kk][1], va[kk][2], va[kk][3], dp + ((uint64_t)(32 * kk) >> 4));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (tci < 3) ATT_STAMP(46 + 4 * tci);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (tci < 3) ATT_STAMP(47 + 4 * tci);
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + hw) : "memory");   // P / alpha reused by the next tile
  };

  int cs = 0, ck = 0;
  for (int j = 0; j < ntiles; ++j) {
    cp_wait<STAGES - 2>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (j < 16) ATT_STAMP(4 + 2 * j);
    gather((j + STAGES - 1) % STAGES);
    copy_pages();
    cp_commit();

    const int stage = j % STAGES;
    const uint32_t st = base + stage * 16384;
    uint8_t* stg = smem + stage * 16384;
    const int b = s_sb[cs];
    if (ck == 0 && cs > 0) prep_q(b);
    const int u0 = s_su0[cs] + ck * ATT_TILE;
    const int nv = min(s_suc[cs] - u0, ATT_TILE);
    const int pos0 = s_first[b] + u0;
    const int wpos = (p.window > 0 && pos0 < seq + L - p.window) ? pos0 : -1;
    if (nv > 0) {
      if (nv < ATT_TILE || wpos >= 0) tile(st, stg, nv, 0, wpos, true);
      else tile(st, stg, nv, 0, -1, false);
    }
    if (j < 16) ATT_STAMP(5 + 2 * j);

    if (ck == s_stiles[cs] - 1) {
      ATT_STAMP(40);
      if (s_snew[cs]) {
        __syncthreads();
        const int jt = tid >> 4, hh = (tid >> 3) & 1, dc = tid & 7;
        float kx[8], vx[8];
        if (jt < L) {
          const uint4 kr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + 2 * pair + hh) * 64 + 8 * dc));
          const uint4 vr = __ldg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + jt) * W + (HQ + HKV + 2 * pair + hh) * 64 + 8 * dc));
          const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kr);
          const __nv_bfloat162* vh2 = reinterpret_cast<const __nv_bfloat162*>(&vr);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float2 a = __bfloat1622float2(kh[i]), c = __bfloat1622float2(vh2[i]);
            kx[2 * i] = a.x; kx[2 * i + 1] = a.y; vx[2 * i] = c.x; vx[2 * i + 1] = c.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) { kx[i] = 0.f; vx[i] = 0.f; }
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += kx[i] * kx[i];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        ss += __shfl_xor_sync(0xffffffffu, ss, 4);
        const float inv = rsqrtf(ss / 64.f + p.eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) kx[i] *= inv;
        if (p.rope) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float sn, cn;
            att_sincos((float)(seq + jt) * __ldg(p.inv_freq + 4 * dc + i), sn, cn);
            float e = kx[2 * i], od = kx[2 * i + 1];
            kx[2 * i] = e * cn - od * sn;
            kx[2 * i + 1] = od * cn + e * sn;
          }
        }
        const float ik = 1.f / k_scale, iv = 1.f / v_scale;
        uint2 kq, vq;
        kq.x = att_e4m3x4(kx[0] * ik, kx[1] * ik, kx[2] * ik, kx[3] * ik);
        kq.y = att_e4m3x4(kx[4] * ik, kx[5] * ik, kx[6] * ik, kx[7] * ik);
        vq.x = att_e4m3x4(vx[0] * iv, vx[1] * iv, vx[2] * iv, vx[3] * iv);
        vq.y = att_e4m3x4(vx[4] * iv, vx[5] * iv, vx[6] * iv, vx[7] * iv);
        if (jt < L) {
          const long long loc = p.cache_loc[(size_t)b * L + jt];
          const size_t off = ((size_t)loc * HKV + 2 * pair + hh) * 64 + 8 * dc;
          *reinterpret_cast<uint2*>(p.k_cache + off) = kq;
          *reinterpret_cast<uint2*>(p.v_cache + off) = vq;
        } else {
          kq = make_uint2(0u, 0u); vq = make_uint2(0u, 0u);
        }
        if (jt < 16) {  // rows 0..15 of the consumed stage (0..7 the block, 8..15 zero)
          const int so = ((((4 * hh + (dc >> 1)) ^ (jt & 7)) << 4) | ((dc & 1) << 3));
          *reinterpret_cast<uint2*>(stg + jt * 128 + so) = kq;
          *reinterpret_cast<uint2*>(stg + 8192 + jt * 128 + so) = vq;
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        tile(st, stg, L, 1, -1, true);
      }
      ATT_STAMP(56);
      // partial (slot cta + b): acc [slot][h][d][row], d = 8 g + w (rows g) / 8 g + 4 + w (g + 8)
      const size_t slot = (size_t)pair * (p.n_ctas + B) + cta + b;
      {
        // O^T fragments -> shared [row][d] (head 0 in this consumed stage, head 1 in the P
        // buffers), then coalesced 16 B rows to the partial slot
        float* so = reinterpret_cast<float*>(hw == 0 ? stg : smem + STAGES * 16384);
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int rr = 8 * jj + 2 * t;
          so[rr * 64 + ((8 * g + w) ^ (rr & 31))] = o[4 * jj];
          so[(rr + 1) * 64 + ((8 * g + w) ^ ((rr + 1) & 31))] = o[4 * jj + 1];
          so[rr * 64 + ((8 * g + 4 + w) ^ (rr & 31))] = o[4 * jj + 2];
          so[(rr + 1) * 64 + ((8 * g + 4 + w) ^ ((rr + 1) & 31))] = o[4 * jj + 3];
        }
        __syncthreads();
        {
          // flat, coalesced: float4 q * 256 + tid of the slot = (head, row) 16 q + (tid >> 4), d 4 (tid & 15)
          float4* dst = reinterpret_cast<float4*>(p.ws_acc) + slot * 2048;
          const int d0 = 4 * (tid & 15);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int hr = 16 * q + (tid >> 4), row = hr & 63, x = row & 31;
            const float* src = reinterpret_cast<const float*>(hr < 64 ? stg : smem + STAGES * 16384) + row * 64;
            dst[q * 256 + tid] = make_float4(src[d0 ^ x], src[(d0 + 1) ^ x], src[(d0 + 2) ^ x], src[(d0 + 3) ^ x]);
          }
        }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          float l = l_r[hf];
          l += __shfl_xor_sync(0xffffffffu, l, 1);
          l += __shfl_xor_sync(0xffffffffu, l, 2);
          if (t == 0) reinterpret_cast<float2*>(p.ws_ml)[(slot * 2 + hw) * 64 + 16 * w + 8 * hf + g] = make_float2(m_r[hf], l);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = 0.f;
      }
      ATT_STAMP(57);
      __threadfence();
      __syncthreads();
      ATT_STAMP(58);
      const long long vb0 = s_pref[b], vb1 = s_pref[b + 1];
      const int i_first = att_cta_of(vb0, T, N), i_last = att_cta_of(vb1 - 1, T, N);
      const int c = i_last - i_first + 1, ng = (c + ATT_GROUP - 1) / ATT_GROUP;
      const int gi = (cta - i_first) / ATT_GROUP, gs = min(ATT_GROUP, c - gi * ATT_GROUP);
      int* cnt_final = p.counters + b * gridDim.y + pair;
      int* cnt_group = p.counters + B * gridDim.y + (size_t)(b * gridDim.y + pair) * p.max_groups + gi;
      const size_t g_slot0 = ((size_t)pair * B + b) * p.max_groups;
      if (COOP && c <= ATT_GROUP) {
        // Cooperative merge: every CTA of the request waits for all partials (the grid is never
        // larger than what is resident, and CTA ranges are contiguous, so no wait is circular),
        // then merges its slice of the 128 (head, row) rows: 16 lanes per row, coalesced loads,
        // all of the request's partials in flight at once.
        int* cnt_done = p.counters + B * gridDim.y + (size_t)(b * gridDim.y + pair) * p.max_groups;
        if (tid == 0) {
          atomicAdd(cnt_final, 1);
          int v;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt_final) : "memory");
            if (v < c) __nanosleep(32);
          } while (v < c);
        }
        __syncthreads();
        ATT_STAMP(41);
        att_coop_merge_v(p, cta - i_first, c, (size_t)pair * (p.n_ctas + B) + i_first + b, b, pair, L, v_scale);
        ATT_STAMP(59);
        __syncthreads();
        if (tid == 0 && atomicAdd(cnt_done, 1) == c - 1) { *cnt_final = 0; *cnt_done = 0; }
      } else {
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
      const int h = tid >> 7, half = (tid >> 6) & 1, row = tid & 63;
      if (s_last == 1) {
        __threadfence();
        float M = -ATT_INF, Ls = 0.f, oo[32];
#pragma unroll
        for (int d = 0; d < 32; ++d) oo[d] = 0.f;
        att_merge_h(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + gi * ATT_GROUP + b, gs, h, row, half, M, Ls, oo);
        att_store_partial_h(p.ws2_acc, p.ws2_ml, g_slot0 + gi, h, row, half, M, Ls, oo);
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
        const int qh = (2 * pair + h) * 8 + (row & 7), jt = row >> 3;
        const size_t tok = (size_t)b * L + jt;
        uint4 gw[4];
        if (jt < L) {
          const uint4* gp = reinterpret_cast<const uint4*>(p.qkv + tok * W + (HQ + 2 * HKV) * 64 + qh * 64 + 32 * half);
#pragma unroll
          for (int q = 0; q < 4; ++q) gw[q] = gp[q];
        }
        float M = -ATT_INF, Ls = 0.f, oo[32];
#pragma unroll
        for (int d = 0; d < 32; ++d) oo[d] = 0.f;
        if (s_last == 2) att_merge_h(p.ws_acc, p.ws_ml, (size_t)pair * (p.n_ctas + B) + i_first + b, c, h, row, half, M, Ls, oo);
        else att_merge_h(p.ws2_acc, p.ws2_ml, g_slot0, ng, h, row, half, M, Ls, oo);
        ATT_STAMP(59);
        const float sc = v_scale / Ls;
        float ss = 0.f;
#pragma unroll
        for (int d = 0; d < 32; ++d) { oo[d] *= sc; ss += oo[d] * oo[d]; }
        s_ss[tid] = ss;
        __syncthreads();
        ss += s_ss[tid ^ 64];
        const float inv = rsqrtf(ss / 64.f + p.eps);
        if (jt < L) {
          uint4* op = reinterpret_cast<uint4*>(p.out + tok * HQ * 64 + qh * 64 + 32 * half);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gw[q]);
            uint4 ow;
            __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ow);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 gf = __bfloat1622float2(gh[i]);
              oh[i] = __floats2bfloat162_rn(oo[8 * q + 2 * i] * inv / (1.f + __expf(-gf.x)),
                                            oo[8 * q + 2 * i + 1] * inv / (1.f + __expf(-gf.y)));
            }
            op[q] = ow;
          }
        }
      }
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

template <int S, bool CL>
static void launch_dec(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * ATT_STAGE_BYTES + (CL ? 2 * ATT_SLOT_FLOATS * sizeof(float) : 0);
  auto kern = attn_dec_kernel<S, CL>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (!CL) {
    kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_x, grid_y);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 8;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, p);
}

#ifndef ATT_DEC_ONLY
template <int S, bool COOP>
static void launch_vwg(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * 16384 + 2 * 8192 + 1024;
  auto kern = attn_vwg_kernel<S, COOP>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 256, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

template <int S>
static void launch_ver(const AttnParams& p, int grid_x, int grid_y) {
  size_t smem = (size_t)S * 16384;
  auto kern = attn_ver_kernel<S>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<dim3(grid_x, grid_y), 128, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

#endif  // ATT_DEC_ONLY

void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages,
              int64_t L) {
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
  p.rope = (int)rope; p.n_ctas = (int)n_ctas; p.L = (int)L;
  p.qk_factor = (float)qk_factor; p.sm_scale_log2 = (float)sm_scale_log2; p.eps = (float)eps;
  int pairs = (int)HKV / 2;
#ifdef ATT_DEC_ONLY
  TORCH_CHECK(L == 1, "attn_dec: this build has the decode kernel only");
#else
  if (L > 1 && stages >= 100) {
    if (stages == 103) launch_vwg<3, false>(p, (int)n_ctas, pairs);
    else if (stages == 104) launch_vwg<4, false>(p, (int)n_ctas, pairs);
    else if (stages == 113) launch_vwg<3, true>(p, (int)n_ctas, pairs);
    else TORCH_CHECK(false, "attn_dec: unsupported wgmma stages");
    return;
  }
  if (L > 1) {
    if (stages == 3) launch_ver<3>(p, (int)n_ctas, pairs);
    else if (stages == 4) launch_ver<4>(p, (int)n_ctas, pairs);
    else TORCH_CHECK(false, "attn_dec: unsupported stages for L > 1");
    return;
  }
#endif
  if (stages == 3) launch_dec<3, false>(p, (int)n_ctas, pairs);
  else if (stages == 4) launch_dec<4, false>(p, (int)n_ctas, pairs);
  else if (stages == 203) launch_dec<3, true>(p, (int)n_ctas, pairs);
  else if (stages == 204) launch_dec<4, true>(p, (int)n_ctas, pairs);
  else TORCH_CHECK(false, "attn_dec: unsupported stages");
}
"""

ATTN_CUDA_SRC = ATTN_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n",
                                      "#include <ATen/cuda/CUDAContext.h>\n" + ATTN_DEVICE_SRC + ATTN_VER_SRC + ATTN_VWG_SRC, 1)

ATTN_CUDA_CPP = """
void attn_dec(torch::Tensor qkv, torch::Tensor inv_freq, torch::Tensor seq_lens, torch::Tensor page_table,
              torch::Tensor cache_loc, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale,
              torch::Tensor v_scale, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
              torch::Tensor ws2_acc, torch::Tensor ws2_ml, torch::Tensor counters, torch::Tensor prof, int64_t max_groups, int64_t B, int64_t HQ, int64_t HKV, int64_t page_shift, int64_t window,
              int64_t rope, double qk_factor, double sm_scale_log2, double eps, int64_t n_ctas, int64_t stages,
              int64_t L);
"""

_ATTN_MODULES = {}
_ATTN_COUNTERS = {}
_ATTN_EMPTY = {}


def _no_prof(device):
    t = _ATTN_EMPTY.get(device)
    if t is None:
        t = _ATTN_EMPTY[device] = torch.empty(0, dtype=torch.int64, device=device)
    return t


def attn_cuda_module(dec_only=False):
    """dec_only: a build with only the decode kernel (shorter compile, for the decode layer's
    prepare, which has a time budget)."""
    key = "dec" if dec_only else "m"
    if key not in _ATTN_MODULES:
        from torch.utils.cpp_extension import load_inline
        src = ATTN_CUDA_SRC
        if dec_only:
            src = ATTN_HOST_SRC.replace("#include <ATen/cuda/CUDAContext.h>\n",
                                        "#include <ATen/cuda/CUDAContext.h>\n#define ATT_DEC_ONLY\n" + ATTN_DEVICE_SRC, 1)
        _ATTN_MODULES[key] = load_inline(
            name="okc_attn_dec" + ("_only" if dec_only else ""), cpp_sources=ATTN_CUDA_CPP, cuda_sources=src,
            functions=["attn_dec"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _ATTN_MODULES[key]


def attn_cuda_supported(cfg, page_table):
    """Shapes the CUDA path covers (others take the Triton / SGLang paths)."""
    page = cfg["page_size"]
    return (1 <= cfg["q_len"] <= 8 and cfg["head_dim"] == 64 and cfg["num_kv_heads"] % 2 == 0
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["batch"] <= 64
            and page & (page - 1) == 0)


def attn_cuda_decode(qkv, data, cfg, inv_freq, n_ctas=264, stages=4, min_keys=256, prof=None, dec_only=False):
    """qkv [B * L, W] bf16 -> [B * L, Hq * 64] bf16; writes the new cache entries. n_ctas: CTAs
    per KV-head pair at most (fewer when the batch has fewer than min_keys keys per CTA)."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    pairs = hkv // 2
    page = cfg["page_size"]
    window = cfg["window"]
    page_table = data["page_table"]
    capacity = page_table.shape[1] * page
    span = min(window, capacity) if window else capacity
    n = max(1, min(max(1, n_ctas // pairs), -(-batch * span // min_keys)))
    cl = length == 1 and stages >= 200
    if cl:
        n = -(-n // 8) * 8   # whole clusters of 8
    device = qkv.device
    max_groups = n // 16 + 2
    key = (device, batch * pairs * (1 + max_groups))
    counters = _ATTN_COUNTERS.get(key)
    if counters is None:
        counters = _ATTN_COUNTERS[key] = torch.zeros(key[1], dtype=torch.int32, device=device)
    slots = pairs * (n + batch) + (pairs * (n // 8 + batch) if cl else 0)
    rows = 8 if length == 1 else 64
    acc_shape = (2, 8, 64) if length == 1 else (2, 64, 64)
    ws_acc = torch.empty((slots,) + acc_shape, dtype=torch.float32, device=device)
    ws_ml = torch.empty((slots, 2, rows, 2), dtype=torch.float32, device=device)
    ws2_acc = torch.empty((pairs * batch * max_groups,) + acc_shape, dtype=torch.float32, device=device)
    ws2_ml = torch.empty((pairs * batch * max_groups, 2, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch * length, hq * 64), dtype=torch.bfloat16, device=device)
    attn_cuda_module(dec_only).attn_dec(
        qkv, inv_freq, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], out, ws_acc, ws_ml, ws2_acc, ws2_ml, counters,
        prof if prof is not None else _no_prof(device), max_groups, batch, hq, hkv, page.bit_length() - 1,
        window, int(bool(cfg["rope_theta"])), cfg["qk_scale_factor"], cfg["sm_scale"] * 1.4426950408889634, 1e-5,
        n, stages, length)
    return out
