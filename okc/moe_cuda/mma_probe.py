"""GPU unit test for the memory-first MoE kernel's building blocks, run in a host's prepare().

Design (see okc/ROOFLINES.md, EXACT.md): the experts must stream NVFP4 weights near HBM
bandwidth (4.06 TB/s measured), which needs ~7.2 T weights/s unpacked. The 64-lane INT pipe
gives ~8.4 T/s at 2 ops per weight, so unpacking must stay ~2 INT ops per weight and the
unpacked values must go straight into mma.sync register fragments (no shared-memory round
trip). Hence:

  - weights repacked in prepare() so each lane's 16-byte load holds exactly its B-fragment
    nibbles for 8 k16 steps of one n8 group (lossless: still E2M1 codes);
  - Marlin's unpack: Out = (q & 0x80008000) | ((q & 0x70007000) >> 6) gives bf16x2 holding
    e2m1 * 2^-126 for the nibbles at bits 12 and 28; q << 4, and q >> 8, give the others;
  - the E4M3 block scale becomes bf16 by placing its bits ((b & 0x7F) << 4 | sign << 8),
    i.e. scale * 2^-120, then an integer add of 238 to the exponent field: scale * 2^118
    (valid for normal scales; prepare() checks there are no subnormal ones);
  - one bf16x2 multiply gives e2m1 * scale * 2^-8 exactly (2 + 4 significant bits);
  - mma.sync m16n8k16 bf16 -> fp32; the global scale and 2^8 apply to the fp32 result.

This probe checks, on the judge's GPU: nvcc via torch.utils.cpp_extension.load_inline (and
how long it takes), the architecture, and that one warp's fragments (A from x rows, B from
repacked weights) reproduce x @ dequant(W).T for one 16 x 8 x 128 tile.
"""
import time

import torch

_PROBE_CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Unpack 8 E2M1 codes (one 32-bit word, Marlin's nibble order) to 4 bf16x2 = e2m1 * 2^-126.
__device__ __forceinline__ void unpack8(uint32_t q, uint32_t out[4]) {
  const uint32_t MASK = 0x70007000u, SIGN = 0x80008000u;
  uint32_t a = q;
  out[0] = (a & SIGN) | ((a & MASK) >> 6);   // nibbles at bits 12, 28
  a <<= 4;
  out[1] = (a & SIGN) | ((a & MASK) >> 6);   // bits 8, 24
  uint32_t b = q << 8;
  out[2] = (b & SIGN) | ((b & MASK) >> 6);   // bits 4, 20
  b <<= 4;
  out[3] = (b & SIGN) | ((b & MASK) >> 6);   // bits 0, 16
}

// E4M3 scale byte -> bf16x2 (both halves) = scale * 2^118.
__device__ __forceinline__ uint32_t scale2(uint32_t s) {
  uint32_t bits = ((s & 0x80u) << 8) | ((s & 0x7Fu) << 4);
  bits += (238u << 7);
  return bits | (bits << 16);
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One warp: out[16, 8] = x[16, 128] @ W[8, 128]^T, W from the repacked layout:
// wq [32 lanes][4 words]: word w holds steps 2w and 2w+1; ws [8 rows][8 steps] E4M3 scales.
__global__ void tile_kernel(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ wq,
                            const uint8_t* __restrict__ ws, float* __restrict__ out) {
  int lane = threadIdx.x;
  int r = lane / 4, c = (lane % 4) * 2;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint4 q4 = reinterpret_cast<const uint4*>(wq)[lane];
  uint32_t qs[4] = {q4.x, q4.y, q4.z, q4.w};
  for (int w = 0; w < 4; ++w) {
    uint32_t v[4];
    unpack8(qs[w], v);
    for (int half = 0; half < 2; ++half) {
      int step = 2 * w + half;
      uint32_t s2 = scale2(ws[r * 8 + step]);
      uint32_t b[2] = {mul2(v[2 * half], s2), mul2(v[2 * half + 1], s2)};
      int k0 = step * 16;
      const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(x + r * 128 + k0);
      const uint32_t* xr8 = reinterpret_cast<const uint32_t*>(x + (r + 8) * 128 + k0);
      uint32_t a[4] = {xr0[c / 2], xr8[c / 2], xr0[c / 2 + 4], xr8[c / 2 + 4]};
      mma16816(acc, a, b);
    }
  }
  out[r * 8 + c] = acc[0];
  out[r * 8 + c + 1] = acc[1];
  out[(r + 8) * 8 + c] = acc[2];
  out[(r + 8) * 8 + c + 1] = acc[3];
}

torch::Tensor tile(torch::Tensor x, torch::Tensor wq, torch::Tensor ws) {
  auto out = torch::zeros({16, 8}, x.options().dtype(torch::kFloat32));
  tile_kernel<<<1, 32>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                         reinterpret_cast<const uint32_t*>(wq.data_ptr()), ws.data_ptr<uint8_t>(),
                         out.data_ptr<float>());
  return out;
}
"""

_PROBE_CPP = "torch::Tensor tile(torch::Tensor x, torch::Tensor wq, torch::Tensor ws);"

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def repack_tile(codes):
    """codes [8 rows, 128 k] uint8 (0..15) -> wq [32 lanes, 4 words] int32 in the lane layout:
    lane (r = lane/4, c = (lane%4)*2) word w holds, for steps 2w and 2w+1, the codes at
    k = step*16 + c, c+1 (b[0]) and c+8, c+9 (b[1]) at Marlin's nibble positions."""
    # unpack8 order: out[0] <- bits 12 (low half) / 28 (high half), out[1] <- 8 / 24,
    # out[2] <- 4 / 20, out[3] <- 0 / 16. out[2*half] is b[0] (k c, c+1), out[2*half+1] is b[1].
    pos = {(0, 0, 0): 12, (0, 0, 1): 28, (0, 1, 0): 8, (0, 1, 1): 24,
           (1, 0, 0): 4, (1, 0, 1): 20, (1, 1, 0): 0, (1, 1, 1): 16}
    out = [[0] * 4 for _ in range(32)]
    for lane in range(32):
        r, c = lane // 4, (lane % 4) * 2
        for w in range(4):
            word = 0
            for half in range(2):
                step = 2 * w + half
                for bi in range(2):           # b[0]: k c, c+1; b[1]: k c+8, c+9
                    for e in range(2):        # bf16x2 element: low half k, high half k+1
                        k = step * 16 + c + 8 * bi + e
                        word |= int(codes[r, k]) << pos[(half, bi, e)]
            out[lane][w] = word - (1 << 32) if word >= (1 << 31) else word
    return torch.tensor(out, dtype=torch.int32)


def run_mma_probe():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    cap = torch.cuda.get_device_capability()
    print(f"cuda probe: capability {cap}, torch {torch.__version__}, cuda {torch.version.cuda}")
    try:
        mod = load_inline(name="okc_mma_probe", cpp_sources=_PROBE_CPP, cuda_sources=_PROBE_CUDA,
                          functions=["tile"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          verbose=False)
    except Exception as e:
        print(f"cuda probe: load_inline failed after {time.time() - t0:.1f} s: {type(e).__name__}: {str(e)[-1500:]}")
        return
    print(f"cuda probe: load_inline compiled in {time.time() - t0:.1f} s")
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for trial in range(4):
        codes = torch.randint(0, 16, (8, 128), generator=gen, dtype=torch.uint8)
        # Normal E4M3 scales across their range (exponent 1..15).
        sf = torch.exp2(torch.randint(-6, 9, (8, 8), generator=gen).float()) * (1 + torch.randint(0, 8, (8, 8), generator=gen).float() / 8)
        sf8 = sf.to(torch.float8_e4m3fn)
        grid = torch.tensor(_E2M1)
        vals = torch.where(codes >= 8, -grid[(codes & 7).long()], grid[(codes & 7).long()])
        w = vals * sf8.float().repeat_interleave(16, dim=1)            # [8, 128]
        x = (torch.randn(16, 128, generator=gen) * (10.0 if trial == 3 else 1.0)).to(torch.bfloat16)
        ref = (x.double() @ w.double().T) * 2.0 ** -8
        wq = repack_tile(codes).cuda()
        ws = sf8.view(torch.uint8).contiguous().cuda()
        out = mod.tile(x.cuda(), wq, ws).double().cpu()
        err = float((out - ref).norm() / ref.norm())
        worst = max(worst, err)
        print(f"cuda probe: tile trial {trial}: rel err {err:.2e} (max |ref| {float(ref.abs().max()):.3g})")
    print(f"cuda probe: {'PASS' if worst < 1e-5 else 'FAIL'} (worst {worst:.2e}); total {time.time() - t0:.1f} s")
