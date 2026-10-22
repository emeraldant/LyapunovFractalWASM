# Rooflines: Muse Spark Open kernel problems on one H200

Generated tables come from `python3 tools/roofline.py`; the reasoning is here. The "practical"
rates below are assumptions until calibrated against measured runs (see "Calibration" at the end).
Exact byte and FLOP counts, peak-rate floors per resource and their validation are in
[EXACT.md](EXACT.md) (`tools/exact.py --validate`).

## Machine model

| quantity | value | used for |
|---|---|---|
| HBM peak | 4.8 TB/s | the judge's floor |
| HBM, long streaming reads, cold L2 | 4.3 TB/s (assumed ~90%) | weights |
| HBM, one random 128 B row per key (page size 1) | 3.6 TB/s (assumed ~75%) | the paged KV cache |
| BF16 tensor peak / achievable | 989 / 700 TFLOP/s | GEMMs (wgmma) |
| FP8 tensor peak / achievable | 1979 / 1300 TFLOP/s | prefill attention (FP8 queries are allowed) |
| INT32 / LOP3 / shift | 15.5 T ops/s (132 SM x 64 x 1.83 GHz) | NVFP4 dequant |
| NVFP4 dequant | 1.875 INT ops per weight (Marlin's sequence, counted in EXACT.md) | MoE |
| each dependent kernel stage | ~1 µs beyond the first (launch gap, ramp, tail) | fusion value |
| small-kernel critical path | ~3 µs (3 dependent DRAM round trips) | tiny attention |

The judge times every shape with a cold L2 and subtracts ~4 µs of graph replay cost, so
nothing is cached across calls: each call pays for its bytes from HBM.

## Headroom (per decode step, geometric mean over loads, as the judge reports)

| problem | SGLang µs | leader | judge floor µs (max) | practical µs (max) | packed µs (max) |
|---|---|---|---|---|---|
| muse_bf16_proj | 4831 | 1.20x | 2390 (2.02x) | 2769 (1.74x) | 2204 (2.19x) |
| muse_attention_decode | 3766 | 1.26x | 599 (6.29x) | 1169 (3.22x) | 1169 (3.22x) |
| muse_moe_w4a16 | 13448 | 1.31x | 4094 (3.28x) | 4822 (2.79x) | 4553 (2.95x) |
| muse_decode_layer | 22620 | 1.24x | 7761 (2.91x) | 9512 (2.38x) | 8333 (2.71x) |
| muse_prefill_layer | 387404 | 1.10x | 126122 (3.07x) | 168211 (2.30x) | 168211 (2.30x) |

The leaders (day 1) are at 1.10-1.31x. The practical ceilings are 1.7-3.2x, so all five
boards are wide open. "packed" adds lossless weight compression (below).

## What bounds each problem, and the levers

### muse_bf16_proj: HBM-bound up to m 128, tensor-bound at m 256

- FLOP/B is ~m: below the ~160 FLOP/B knee (700 T / 4.3 TB/s) every shape but m 256 is
  pure weight streaming. Time = weight bytes / achieved bandwidth + fixed cost.
- **Lossless 12-bit weights.** The judge's BF16 weights (randn / sqrt K, as real weights are
  roughly) have exponent entropy 2.54 bits: 15 exponent values cover 99.986%. Storing
  sign + 7 mantissa bits + a 4-bit exponent code (one code = escape, with a sparse list for the
  0.014%) is exact and moves 25% fewer bytes. Decoding is ~4 lane ops per weight, against ~15
  available per weight at full bandwidth. The rules allow it explicitly ("compress them
  losslessly"). This is the single biggest lever on the memory-bound shapes: 1.74x -> 2.19x
  ceiling. Entropy coding (10.5 bits) would add ~12% more but costs a serial decode.
- FP32 router: same trick, 28 bits (12.5% fewer bytes).
- m 256 is tensor-bound at ~70% wgmma: cuBLAS is close to optimal there, so leave it unless
  measured otherwise.
- The small shapes (o, unpooler-slice, router: 1.7-3.6 µs floors) are dominated by fixed
  cost: one launch, no memset, no separate cast or reduce kernel (split-K reductions must be
  in-kernel, e.g. last-CTA-reduces).

### muse_attention_decode: latency-bound on small batches, gather-bound on full layers

- Sliding layers run 46 times a step and move 0.5-19 MB: at B 1 and 16 the floor is 0.1-2 µs
  of bandwidth but ~3 µs of dependent DRAM round trips (qkv, then page table, then KV rows).
  SGLang spends 5-6 launches here. The target is one launch, ~4 µs per call.
- Decode 1 and verify 1 loads are almost entirely this latency (practical 292 µs/step vs the
  judge's 75).
- Full layers at B 16/32 read 333-667 MB of KV as random 128 B rows (page size 1, scattered
  pages). The real ceiling is the gather bandwidth: measure it.
- One CTA should cover both KV heads, so each key is one full 128 B row per cache (a CTA per
  head reads half-rows: two DRAM transactions for one row's worth of data).
- Prefetch the page table a block ahead so the KV gather never waits on it.
- Keep queries BF16 (more accurate than FA3's FP8 path, at no cost: decode isn't tensor-bound).

### muse_moe_w4a16: HBM- and INT-bound together at every decode shape

- The busiest GPU touches 2.5 experts at bs 1 and 24 at bs 256 (simulated with the judge's
  generator; its floor counts only 75% of these). 28.3 MB per expert.
- FLOP/B <= 56: memory-bound throughout, so time = touched experts x 28.3 MB / bandwidth.
- Dequant runs on the 64-lane INT pipe: Marlin's 1.875 ops per weight decode 8.25 T weights/s
  against HBM's 8.53 T, so the experts are co-bound by INT and HBM. A naive Triton dequant
  (~6-8 ops) would run at a third of bandwidth. This is why Marlin is hard to beat on the GEMMs
  themselves, and why the glue (align, eager post-norm below 32 tokens, padding over 257
  buckets) is the first target.
- At bs 1 only 2-3 experts are touched: both GEMMs must still spread each expert across all
  132 SMs (split-K / stream-K), or bandwidth collapses.
- Block scales have 3.7 bits of entropy: coding them in 4 bits saves 5.5% of the bytes. The
  E2M1 codes are uniform random in the judge's weights: incompressible.

### muse_decode_layer: the sum, dominated by dense weights at small batch

- B 1: 210 MB per call, of which 130 MB dense BF16 + 8 MB router: the 12-bit packing alone
  is worth ~16% here.
- Stage count matters at small batch: ~13 dependent stages as SGLang runs it, ~7 achievable
  with fusion (norms into GEMM epilogues via last-CTA reductions, routing into the router
  GEMM, silu into GEMM1). ~1 µs each, x62 calls per step.
- Attention and MoE as above.

### muse_prefill_layer: compute-bound; the experts run on an Ampere-era kernel

- Dense GEMMs ~55% of FLOPs, experts ~40%, attention up to 49% on full layers at 80K.
- SGLang runs the experts on Marlin (mma.sync, block M <= 64) even at 16K tokens: a
  Hopper-native grouped GEMM (wgmma, NVFP4 dequantized to BF16 by producer warps, TMA) is the
  big lever.
- Attention: FA3 FP8 (queries may be FP8) at ~1300 T; already SGLang's path.
- Doesn't count toward the overall board.

## Calibration and the hill-climb loop

1. `okc check baseline.py` once per problem: SGLang's own time per shape, and our ~1.00x
   noise floor. This replaces the guessed per-shape SGLang times.
2. Every `okc check` returns the last 16,000 characters of what the run prints. prepare()
   already times dozens of candidate kernels on the judge's GPU (CUDA graphs, cold L2) and
   prints them, so one check run is hundreds of micro-benchmarks. Use it to calibrate the
   constants above: achieved streaming and gather bandwidth, stage cost, dequant rate.
3. Then per problem: rank shapes by (weight x (our time - practical floor)), attack the
   largest gap, re-check. 20 check runs and 10 scored runs per problem per day.

## Per-shape tables

### muse_bf16_proj

| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| m 1   qkv_gate | 62 | 37.8 | 0.04 | 1 | 7.9 | 8.8 | 6.6 | mem |  |
| m 1   o | 62 | 16.8 | 0.02 | 1 | 3.5 | 3.9 | 2.9 | mem |  |
| m 1   pooler | 62 | 67.1 | 0.07 | 1 | 14.0 | 15.6 | 11.7 | mem |  |
| m 1   unpooler | 62 | 8.4 | 0.01 | 1 | 1.7 | 2.0 | 1.5 | mem |  |
| m 1   router | 62 | 8.4 | 0.00 | 0 | 1.8 | 2.0 | 1.7 | mem |  |
| m 16  qkv_gate | 62 | 38.1 | 0.60 | 16 | 7.9 | 8.9 | 6.7 | mem |  |
| m 16  o | 62 | 17.1 | 0.27 | 16 | 3.6 | 4.0 | 3.0 | mem |  |
| m 16  pooler | 62 | 67.5 | 1.07 | 16 | 14.1 | 15.7 | 11.8 | mem |  |
| m 16  unpooler | 62 | 67.5 | 1.07 | 16 | 14.1 | 15.7 | 11.8 | mem |  |
| m 16  router | 62 | 8.7 | 0.07 | 8 | 1.8 | 2.0 | 1.8 | mem |  |
| m 32  qkv_gate | 62 | 38.4 | 1.21 | 31 | 8.0 | 8.9 | 6.7 | mem |  |
| m 32  o | 62 | 17.4 | 0.54 | 31 | 3.6 | 4.0 | 3.1 | mem |  |
| m 32  pooler | 62 | 67.9 | 2.15 | 32 | 14.1 | 15.8 | 11.9 | mem |  |
| m 32  unpooler | 62 | 67.9 | 2.15 | 32 | 14.1 | 15.8 | 11.9 | mem |  |
| m 32  router | 62 | 8.9 | 0.13 | 15 | 1.9 | 2.1 | 1.8 | mem |  |
| m 8   qkv_gate | 62 | 37.9 | 0.30 | 8 | 7.9 | 8.8 | 6.6 | mem |  |
| m 8   o | 62 | 16.9 | 0.13 | 8 | 3.5 | 3.9 | 3.0 | mem |  |
| m 8   pooler | 62 | 67.3 | 0.54 | 8 | 14.0 | 15.7 | 11.8 | mem |  |
| m 8   unpooler | 62 | 8.5 | 0.07 | 8 | 1.8 | 2.0 | 1.5 | mem |  |
| m 8   router | 62 | 8.5 | 0.03 | 4 | 1.8 | 2.0 | 1.7 | mem |  |
| m 128 qkv_gate | 62 | 40.4 | 4.83 | 119 | 8.4 | 9.4 | 7.2 | mem |  |
| m 128 o | 62 | 19.1 | 2.15 | 112 | 4.0 | 4.5 | 3.5 | mem |  |
| m 128 pooler | 62 | 70.3 | 8.59 | 122 | 14.6 | 16.3 | 12.4 | mem |  |
| m 128 unpooler | 62 | 70.3 | 8.59 | 122 | 14.6 | 16.3 | 12.4 | mem |  |
| m 128 router | 62 | 10.6 | 0.54 | 51 | 2.2 | 2.5 | 2.2 | mem |  |
| m 256 qkv_gate | 62 | 43.1 | 9.66 | 224 | 9.8 | 13.8 | 13.8 | tensor |  |
| m 256 o | 62 | 21.5 | 4.29 | 200 | 4.5 | 6.1 | 6.1 | tensor |  |
| m 256 pooler | 62 | 73.4 | 17.18 | 234 | 17.4 | 24.5 | 24.5 | tensor |  |
| m 256 unpooler | 62 | 73.4 | 17.18 | 234 | 17.4 | 24.5 | 24.5 | tensor |  |
| m 256 router | 62 | 12.8 | 1.07 | 84 | 2.7 | 3.0 | 2.7 | mem |  |

### muse_attention_decode

| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| B 1  L 1 sliding | 46 | 0.5 | 0.01 | 16 | 0.1 | 4.0 | 4.0 | latency |  |
| B 1  L 1 full | 16 | 20.8 | 0.33 | 16 | 4.3 | 6.8 | 6.8 | gather |  |
| B 16 L 1 sliding | 46 | 8.6 | 0.13 | 16 | 1.8 | 4.0 | 4.0 | latency |  |
| B 16 L 1 full | 16 | 332.9 | 5.24 | 16 | 69.4 | 93.5 | 93.5 | gather |  |
| B 32 L 1 sliding | 46 | 17.3 | 0.27 | 16 | 3.6 | 5.8 | 5.8 | gather |  |
| B 32 L 1 full | 16 | 665.8 | 10.49 | 16 | 138.7 | 185.9 | 185.9 | gather |  |
| B 1  L 8 sliding | 46 | 0.6 | 0.07 | 115 | 0.1 | 4.0 | 4.0 | latency |  |
| B 1  L 8 full | 16 | 20.9 | 2.62 | 126 | 4.3 | 6.8 | 6.8 | gather |  |
| B 16 L 8 sliding | 46 | 9.4 | 1.08 | 115 | 2.0 | 4.0 | 4.0 | latency |  |
| B 16 L 8 full | 16 | 333.7 | 41.95 | 126 | 69.5 | 93.7 | 93.7 | gather |  |
| B 32 L 8 sliding | 46 | 18.8 | 2.15 | 115 | 3.9 | 6.1 | 6.1 | gather |  |
| B 32 L 8 full | 16 | 667.4 | 83.89 | 126 | 139.0 | 186.3 | 186.3 | gather |  |

### muse_moe_w4a16

| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| bs 1   | 60 | 71.0 | 0.25 | 4 | 14.8 | 19.5 | 18.5 | mem | 2 routes, 2.5 experts |
| bs 16  | 60 | 336.0 | 2.57 | 8 | 70.0 | 81.1 | 76.5 | mem | 26 routes, 11.8 experts |
| bs 32  | 60 | 462.5 | 5.11 | 11 | 96.4 | 110.6 | 104.2 | mem | 51 routes, 16.2 experts |
| bs 8   | 60 | 244.5 | 1.42 | 6 | 50.9 | 59.9 | 56.5 | mem | 14 routes, 8.6 experts |
| bs 128 | 60 | 637.7 | 19.64 | 31 | 132.9 | 151.3 | 142.7 | mem | 195 routes, 22.0 experts |
| bs 256 | 60 | 718.3 | 40.08 | 56 | 149.6 | 170.0 | 160.5 | mem | 398 routes, 24.3 experts |

### muse_decode_layer

| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| B 1  L 1 sliding | 46 | 210.0 | 0.39 | 2 | 43.8 | 57.9 | 46.1 | mem |  |
| B 1  L 1 full | 16 | 230.3 | 0.71 | 3 | 48.0 | 63.5 | 51.7 | mem |  |
| B 16 L 1 sliding | 46 | 543.2 | 5.79 | 11 | 113.2 | 135.7 | 116.9 | mem |  |
| B 16 L 1 full | 16 | 867.5 | 10.90 | 13 | 180.7 | 225.8 | 207.0 | mem |  |
| B 32 L 1 sliding | 46 | 679.8 | 11.56 | 17 | 141.6 | 167.9 | 147.3 | mem |  |
| B 32 L 1 full | 16 | 1328.4 | 21.77 | 16 | 276.7 | 348.0 | 327.5 | mem |  |
| B 1  L 8 sliding | 46 | 384.3 | 2.56 | 7 | 80.1 | 98.4 | 84.2 | mem |  |
| B 1  L 8 full | 16 | 404.5 | 5.11 | 13 | 84.3 | 104.0 | 89.8 | mem |  |
| B 16 L 8 sliding | 46 | 856.0 | 45.41 | 53 | 178.3 | 208.5 | 185.6 | mem |  |
| B 16 L 8 full | 16 | 1180.3 | 86.28 | 73 | 245.9 | 298.5 | 275.7 | mem |  |
| B 32 L 8 sliding | 46 | 957.7 | 91.63 | 96 | 199.5 | 232.5 | 208.7 | mem |  |
| B 32 L 8 full | 16 | 1606.2 | 173.37 | 108 | 334.6 | 412.6 | 388.9 | mem |  |

### muse_prefill_layer

| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |
|---|---|---|---|---|---|---|---|---|---|
| B 1 ctx 80000 q 753 sliding | 46 | 1056.2 | 261.02 | 247 | 263.9 | 405.9 | 405.9 | att 2% moe 42% dense 56% |  |
| B 1 ctx 80000 q 753 full | 16 | 1076.2 | 502.49 | 467 | 508.1 | 591.6 | 591.6 | att 49% moe 22% dense 29% |  |
| B 4 ctx 80000 q 753 sliding | 46 | 1280.4 | 1080.80 | 844 | 1092.8 | 1536.0 | 1536.0 | att 2% moe 44% dense 54% |  |
| B 4 ctx 80000 q 753 full | 16 | 1360.2 | 2046.66 | 1505 | 2069.4 | 2279.0 | 2279.0 | att 48% moe 23% dense 28% |  |
| B 1 ctx 0 q 16384 full | 16 | 2596.3 | 6049.97 | 2330 | 6117.3 | 8289.3 | 8289.3 | att 9% moe 39% dense 52% |  |
| B 1 ctx 32768 q 16384 sliding | 138 | 2596.8 | 5637.62 | 2171 | 5700.3 | 7972.1 | 7972.1 | att 2% moe 41% dense 56% |  |
| B 1 ctx 32768 q 16384 full | 16 | 2604.7 | 8248.99 | 3167 | 8340.7 | 9980.9 | 9980.9 | att 33% moe 28% dense 38% |  |
| B 1 ctx 65536 q 16384 full | 16 | 2613.1 | 10448.01 | 3998 | 10564.2 | 11672.4 | 11672.4 | att 47% moe 22% dense 30% |  |

