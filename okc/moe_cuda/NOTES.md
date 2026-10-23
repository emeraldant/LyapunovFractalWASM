# MoE kernel: first-principles log

Target: stream the touched experts' NVFP4 weights at the measured 4.06 TB/s (bs 1: 75 MB ->
~18.5 µs + ~2 µs ramp; bs 16: 338 MB -> ~85 µs; bs 256: 690 MB -> ~172 µs). Every gap to
that is explained before the next change. Tools: `tools/sass.py moe` (exact SASS per main
loop, per pipe), `tools/moe_model.py` (work decomposition x pipes x latency), the sim
(`moe_cuda/sim_src.py`: correctness on held-out cases + per-stage timing on the judge's GPU).

## Dream roofline (the target every number below is compared against)

Busiest GPU of 8 (judge's routing), all touched experts' NVFP4 codes + E4M3 scales read once
(28.31 MB per expert: GEMM1 18.87, GEMM2 9.44) at the measured streaming rate 4.06 TB/s,
plus ~5 us of launches/ramp/route/post (CUDA graph, ~1 us per extra kernel):

| bs | experts touched | weight MB | dream GEMMs (us) | dream total (us) | SGLang (us) | dream speedup |
|---|---|---|---|---|---|---|
| 1 | 2.58 | 73 | 18.0 | 23 | 76.8 | 3.3x |
| 8 | 8.43 | 239 | 58.8 | 64 | 144.1 | 2.25x |
| 16 | 11.86 | 336 | 82.7 | 88 | 196.9 | 2.24x |
| 32 | 15.98 | 452 | 111.4 | 116 | 206.9 | 1.78x |
| 128 | 22.80 | 646 | 159.0 | 164 | 468.7 | 2.86x |
| 256 | 24.42 | 691 | 170.3 | 175 | 618.3 | 3.53x |

Geomean dream speedup ~2.6x (current submission 1.34x). The sim prints, per stage, the
achieved fraction of this dream (`gemm1 193.1 (58% of 112)`), so a gap is always located to
a stage before anything is changed.

## The roofline has three terms, not one (found 2026-10-23, run 6c4f8d07)

Per byte of weights (codes + scales) streamed from HBM, a design also moves bytes through
shared memory (bulk-copy writes + ldmatrix / ld.shared reads), at 128 B/clk/SM (~33.4 TB/s),
and issues instructions at 4 warp-instructions/clk/SM:

    t >= max( HBM bytes / 4.06 TB/s,  smem bytes / 33.4 TB/s,  instructions / issue rate )

| design | smem B per HBM B | smem ceiling | issue ceiling |
|---|---|---|---|
| v5 GEMM1 (A = 16 token rows, read by all 8 warps) | 6.4 | 5.2 TB/s | 7.6 TB/s |
| v5 GEMM2 | 10.4 | **3.2 TB/s (below HBM)** | 6.9 TB/s |
| v6 swap-AB, 8 tokens (weights = A from registers, tokens = B) | 4.25 | 7.9 TB/s | ~8 TB/s |

Evidence: the "loop only" ablation (no loads, no decode, no MMA; only the smem reads) runs
at 113-115 B/clk/SM in both GEMMs and both batch sizes = 88-90% of the smem port, and costs
95 of GEMM2's 154 us at bs 256. Earlier "consumer starvation" was this, not DRAM.

Measurement lesson: a first profiling build with runtime ablation branches spilled 348 B and
ran 2x slow (invalid data). Ablations are now compile-time instantiations in a separate
module (-DOKC_PROF), and `tools/sass.py moe_ws prof` checks the profiling build's main loop
equals production's before any profile is trusted.

### v6 swap-AB measured (runs 26eb50a3, c4a3842d)

Weights = A (16 n-rows from registers, a warp's two n8 groups), tokens = B (8 per group,
ldmatrix.x4 per two k16 steps); GEMM1's pair is (gate, up) so silu*up stays in-lane; the
epilogue pairs columns with one __shfl_xor. SASS: 551 instructions per 4 chunks for 2048
weights per warp in both GEMMs (GEMM2 was 1024). Gate: correct and repeatable on all 19
shapes; s2 (16 tokens/item) fails only skewed bs 32 (x0.86, not scored).
Judge check c4a3842d: **1.617** (bs 1 1.65 Marlin, 8 1.57, 16 1.68, 32 1.38, 128 1.78, 256 1.67).

Why not 2x: at 16 tokens each of the 8 warps still reads the same B fragments (4 KB per warp
per chunk). bs 256 GEMM1 bounds: smem 140 us, HBM 112 us, issue 85 us, measured 222 us:
smem and HBM are co-limiting. Next structural step: wgmma (a warpgroup's tensor core reads
the token tile from smem once: 4x fewer activation reads), as Humming / fractalyze do.

### v7 wgmma and pipeline depth (runs daa8882c probe, e8d6df4f, c2c158d0, fb5afbd8)

Probe (in a bf16_proj host, to save MoE checks): wgmma's register A fragment = mma.sync's per
warp; D layout = mma.sync's C per 8 columns; K-major no-swizzle B: K stride in LBO, 8-row
block stride in SBO (the other way: 99% error at N = 16). v7 (wgmma m64nNk16, B tile via
16-byte cp.async in the canonical layout, fence.proxy.async after the stage wait, A double-
buffered by half chunk): correct and repeatable on all shapes; +5-6% at bs 128/256 only.
bs 256 GEMM1 stayed ~220 us: smem was no longer the binding term after v6.

Pipeline depth hypothesis (data + math behaved like a sum, not a max): 8 stages at 2 CTAs/SM
helps small batches (bs 16 x1.09-1.11, bs 1 x1.04 vs plan) and hurts large ones (x0.73-0.90):
at large batch, halving consumer warps costs more than the deeper prefetch gains, so the large
batch limit is consumer-side latency (too few warps to hide decode -> MMA chains).
Judge check fb5afbd8: **1.638** (bs 1 1.64 Marlin, 8 1.68 v6 8-stage, 16 1.66, 32 1.37,
128 1.77, 256 1.74 v7).

## Hopper facts used (per SM sub-partition = SMSP, per clock)

| pipe | warp instructions / clk | what runs there |
|---|---|---|
| issue | 1 | everything |
| ALU | 1/2 | LOP3, SHF, IADD3, PRMT, LEA, ISETP, MOV |
| FMA-heavy | 1/2 | IMAD (incl. IMAD.SHL/.MOV/.X) |
| FMA (lite+heavy) | 1 | FFMA, HMUL2/HFMA2 (shares issue with IMAD) |
| tensor | 1/8 | HMMA m16n8k16 |
| uniform | own pipe | U* scalar ops |

## Iterations

| version | change | SASS (GEMM1, 1 tile, per warp-iteration) | measured bs 1 GEMM1 / GEMM2 / total | explanation |
|---|---|---|---|---|
| v1 | K-split warps, register prefetch, global A gathers | n/a | total 45 µs; collapses at bs 256 (1957 µs) | — |
| v2 | 4-stage cp.async, ldmatrix A, no K split | 382 issue, 200 ALU, 91 IMAD | total 70.6 µs | 60 INT ops/chunk of 64-bit address math, ~8 INT per scale decode; only 3 x 9.2 KB in flight per CTA |
| v3 | hoisted pointers, PRMT+IMAD scale decode (scales verified >= 0, normal) | 278 issue, 127 ALU, 64 IMAD | 4 stages, 2 CTA/SM: 33.3 / 19.8 / 60.2 µs; 8 stages, 1 CTA/SM: 21.7 / 14.7 / 44.3 µs (1.85x SGLang) | see below |

| v3.1 | compact work list: only non-empty (expert, token chunk) items | unchanged | bs 256: 2627 -> 386 us (mt1, 4 st, 4 CTA/SM); see below | — |

### v3.1: the large-batch collapse was load imbalance, not the tile path

Items were (active expert, n block, m chunk) with m_chunks = cdiv(bs, 16 MT) for every expert,
so at bs 256 / MT 1, 15 of 16 items were empty; worse, with a grid stride of 264 (a multiple of
8), the non-empty mc = 0 items all landed on CTAs with blockIdx % 16 in {0, 8}: 1/8 of the GPU
did all the work (measured 8-13x slow, matching). The route kernel now emits the list of
non-empty chunks (e | chunk << 8). Run 278fbdb2 (per stage, % of dream):

| bs | best variant | GEMM1 | GEMM2 | total | Marlin plan | SGLang |
|---|---|---|---|---|---|---|
| 1 | mt1, 8 st, 1 CTA/SM | 22.2 (42%) | 14.9 (31%) | 44.8 | 46.6 | 82 |
| 8 | mt1, 4 st, 4 CTA/SM | 62.4 (60%) | 34.7 (54%) | 106.0 | 105.1 | 147 |
| 16 | mt1 4/4 | 87.7 (53%) | 48.7 (48%) | 144.6 | 125.2 | 171 |
| 32 | mt1 4/4 | 146.7 (57%) | 80.1 (52%) | 235.4 | 228.7 | 236 |
| 128 | mt1 4/4 | 193.1 (58%) | 106.6 (52%) | 311.1 | 295.1 | 476 |
| 256 | mt2 4/4 | 223.2 (48%) | 138.4 (39%) | 378.1 | 386.7 | 575 |

MT 4 loses everywhere (it is tensor-bound on paper: 512 tensor clocks per MODE 1 iteration).

### What bounds v3.1 at 50-60%: instruction issue

MODE 2 main loop (SASS, per warp, 1024 weights): 150 instructions = unpack ~60 (sign and
magnitude need different shifts: 4 ops per bf16x2, 3 for the aligned one), scale build 16,
HMUL2 16, LDSM 8, HMMA 8, and ~40 of cp.async address / stage / barrier bookkeeping. One issue
per clock per sub-partition caps it at 4 x 1024 / 150 weights per clock per SM = ~3.8 TB/s
(MODE 1: 276 per 2048 weights -> ~4.1 TB/s). Measured at full occupancy: ~2.35 TB/s, i.e.
~57% of the issue ceiling, the rest being dependent-chain stalls (8 HMMAs chained on one
accumulator, a barrier per chunk). Both point the same way: fewer instructions per weight.

Plan (v4), each with its predicted saving per MODE 2 iteration:
1. Unpack layout with shared shifted copies of each word (searched exhaustively,
   `scratchpad layout.py`): ~60 -> ~32.
2. Unroll the chunk loop by STAGES: stage offsets and pointer steps become immediates,
   pointer increments once per STAGES chunks: ~40 -> ~12.
3. Two accumulators (even/odd k16 steps) to halve the HMMA dependency chain.
Target ~100 instructions per 1024 weights (MODE 2), ~185 per 2048 (MODE 1): issue ceiling
~5.7-6.2 TB/s, so DRAM, not issue, binds.

### v4 / v5 measured (runs b873639f, 640a6584; % of dream per GEMM)

v4 = 8-op unpack layout (exhaustive search) + loop unrolled by stages; v5 = warp-specialized
(producer warp, cp.async.bulk, full/empty mbarriers, no CTA barrier) + paired scale decode.
SASS per 128-k chunk at MT 1: GEMM2 150 (v3) -> 112 (v4) -> 83 (v5); GEMM1 276 -> 197 -> 150.

| bs | v5 ws mt1 4 st 4 CTA: GEMM1 / GEMM2 (% dream) | total | v4 total | Marlin plan |
|---|---|---|---|---|
| 1 | 24.4 (38%) / 16.0 (29%) | 47.8 | 52.5 | 45.8 |
| 8 | 51.5 (72%) / 26.6 (70%) | 86.5 | 103.6 | 104.4 |
| 16 | 68.7 (68%) / 41.5 (56%) | 117.8 | 130.8 | 124.9 |
| 32 | 126.5 (66%) / 66.8 (63%) | 202.6 | 207.7 | 225.5 |
| 128 | 163.1 (68%) / 88.0 (63%) | 263.4 | 267.8 | 293.1 |
| 256 | 199.2 (54%) / 131.6 (41%) | 348.2 | 339.0 | 382.5 |

Sim geomean vs SGLang: v5 1.564, v4 1.463, Marlin plan 1.434. Fewer instructions helped
small/medium batches (bs 8: 59% -> 72%) but not bs 256 (still 41-54%), so at large batch the
binding constraint is no longer issue. Open questions, to answer before the next change:
- bs 256 has 23 experts and ~300 local routes: GEMM items with >1 chunk per expert re-read
  the same weights (MT 1 = 16 rows); are those re-reads hitting L2 or DRAM?
- bs 1 (2-3 experts, ~165 items on 132 SMs x 4 CTAs) is parallelism-bound: split K.

### The gate (sim_src.py) and what it caught

Every candidate now runs on 19 cases (6 scored, 11 nearby / skewed, 5 odd tests): correct
(judge bounds vs FP32), bitwise repeatable (second run equal: catches pipeline races), and
timed paired against the CURRENT plan per shape; one regression fails the gate.
- v5 (k1): correct and repeatable everywhere; regressions bs 1 (x0.94-0.97), bs 3, bs 128
  skewed (x0.97). Skewed big batches are weak for everything we have (x0.67-0.79 vs SGLang):
  hot experts with 60+ rows re-stream weights per 16-row chunk.
- In-CTA split K (k2: 2 groups of 8 consumer warps, partials met in smem): bs 1 38.2 us
  (x1.20 vs plan, x2.17 vs SGLang) but regresses every other shape (as low as x0.41): at 1
  CTA/SM it has 16 consumer warps vs k1's 32, which only pays when items < SMs.
- KS 4 (33 warps) is illegal: 1056 threads > 1024.

Integration: prepare() adds CUDA configs as candidates per bucket; adopted only if correct on
the bucket routing + odd smaller batch + 2 fresh routings, bitwise repeatable, and <= 98% of
the incumbent on TWO routings. Judge check 7ec92eff: 1.427 (current submission 1.337). bs 1
kept Marlin: k2 won the first routing (43 vs 52 us) and lost the second (71 vs 60 us), the
gate doing its job; bs 1's best plan depends on how many experts the routing touches, which
only the device knows -> next: choose split K on the device from the item count.

### Why v3 is still ~1.8x off the roofline

Measured per-CTA chunk times (0.5-1.2 µs per 9.2 KB chunk) would need ~3.6 µs of DRAM
latency to be latency-bound: implausible (a model fit with DRAM latency as the free
parameter also failed, rms log error 0.63, worst at large batch). Instruction latency at low
occupancy explains it: with 8-16 warps per SM, each scheduler holds 2-4 warps, and each
warp's 278-instruction iteration is a dependent chain (LDS -> unpack -> HMUL2 -> HMMA)
behind a per-chunk __syncthreads; there aren't enough warps to hide those latencies. Same
mechanism explains why 8 stages at 1 CTA/SM wins at bs 1 (more bytes in flight, items fit in
one wave) but loses at bs 16 (2 warps per scheduler on far more items).

### Next (predicted, then measured)

1. Occupancy: 4 CTAs/SM (8 warps per scheduler) with 3-4 small stages each (~110 KB in
   flight per SM). Grid capped by the occupancy calculator so no CTA queues behind others.
2. bs <= 16: split K across CTAs so items fill the 528 resident CTAs (bs 1 has only 170
   items per GEMM): fp32 atomics + a small silu*up pass. Predicted bs 1: ~15 + ~8 µs GEMMs,
   ~30 µs total (SGLang 82 µs).
3. Large batch (tiles > 1): mma.sync tops out near ~535 TFLOPS here; wgmma (swap-AB, as in
   fractalyze/sglang PR #5 and Humming) if tensor time binds.
