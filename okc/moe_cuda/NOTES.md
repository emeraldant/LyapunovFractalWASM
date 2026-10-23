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

### Consumer-side decomposition at bs 256 (synthetic-weight profiler, run f11e4b95)

`run_moe_profile_synth` builds random NVFP4 weights for the 32 local experts and profiles in
ANY problem's host (kernel speed does not depend on weight values): experiments no longer
spend MoE checks. v7 (16 tokens/item) GEMM1 at bs 256:

| mode | us | | mode | us |
|---|---|---|---|---|
| loop only (handshake + weight/scale LDS) | 46 | | weights + activations, no math | 133 |
| loop + decode | 94 | | math only (no memory) | 161 |
| loop + MMA | 93 | | full | 216 |

HBM dream: 112 us. (1) Compute alone exceeds the HBM floor: loop / decode / tensor cost ~47 us
each on different units, yet add up (6 consumer warps per SMSP, warpgroup-lockstepped wgmma:
latency not hidden). (2) Tensor work is real at bs 256: 16-token items = 38.6 GFLOP padded in
GEMM1, ~39 us at peak. (3) Memory alone is above dream: hot experts re-streamed per 16-token
chunk (2304 items vs 1536 unique). Next levers: 256-k stages (half the handshakes per byte),
more independent work per warp (decode overlapping wgmma), 32-token tiles without spills.

### Measured unit rates and overlap (rate_probe.py, H200)

- wgmma m64nNk16 RS (A from registers): ~14.5 clk/instr/SM for N 8 and 16, 17.8 clk at N 32
  (90% of the 1024 dense bf16 MAC/clk/SM peak only at N 32). mma.sync m16n8k16 ~1200 MAC/clk/SM.
- LOP3 0.48 warp-instr/clk/SMSP, HMUL2.BF16 0.87.
- Overlap: RS wgmma and mma.sync do NOT overlap the ALU: ALU+wgmma took 1616 clk vs 1586 for
  the sum of the parts; LOP3+wgmma additive, HMUL2+wgmma worse than additive; LOP3+HMUL2
  overlap partly (~85%). So the consumer is an ADDITIVE cost model, in SM-cycles: ALU op 0.52,
  HMUL2 0.29, wgmma N16 14.5, HMMA 1.7. It predicts bs 256 t2 GEMM1 at 203 us (measured 193).
- Decode microbenchmark (registers only, no memory): 0.0212 SM-clk per weight; with mma.sync
  N8 0.0235 = 65% of the HBM budget (0.0362 SM-clk per weight at 4.06 TB/s). The full kernel
  costs ~1.85x that, so the gap at small batch is the memory pipeline, not the math.
  (IMAD.WIDE rotation variant of the decode: slower, refuted.)

### Small-batch ablations and the bytes-in-flight hypothesis

bs 16/32 on s1 (v6): weights-only (no math) reaches only 3.3-3.6 TB/s; math-only is under the
dream. Little's law: 4.06 TB/s x ~1 us loaded latency = ~4 MB in flight chip-wide = ~31 KB per
SM... at the measured per-chunk latencies (several us under load) the smem-held stages cap
bytes in flight. Fix that does not cost smem: cp.async.bulk.prefetch.L2 PFD chunks ahead
(PFD 6 -> ~220 KB in flight per SM, L2 footprint ~29 MB < 50 MB). `pfd` kernel parameter.

Split K (v6 kparts, fp32 partial slices summed in fixed order, deterministic): only bs 1 wins,
45 -> 38.6 us (route 1.4, GEMM1+silu 19.3, GEMM2 10.7, post 3.9 us).

**Refuted (run 7667fa12): L2 prefetch.** pfd 4 / 8 chunks ahead made bs 8 weights-only WORSE
(3.27 -> 2.77 -> 2.37 TB/s): bytes in flight were never the limit (see the bandwidth probe).

**HBM ceiling, measured (bw_probe.py, runs after 7667fa12):** 1 GB streamed by LDG.128: 4.39-4.44
TB/s; by cp.async.bulk 8 KB + 1 KB x 4 stages x 4 CTAs/SM (v6's exact shape): 4.49 TB/s; every
TMA shape from 72 to 216 KB/SM in flight: 4.42-4.52. So the roofline is ~4.49 TB/s, not 4.06
(the dream times above are ~10% generous to us). Short runs: time = bytes / 4.49 TB/s + ~6 us
per kernel (launch + fill + drain), for 528 / 448 / 264 / 224 equal CTAs alike: the CTA -> SM
spread does not cost bandwidth (refuted: "per-SM fair share makes 4-item SMs slow").

**Refuted (run after bw probe): stream-K for compute balance.** Equal-bytes CTA ranges (sk,
partial slots) are correct and bitwise repeatable, but GEMM1 at bs 8 stays 48 us balanced or
not; plain split K4 is as good or better everywhere (bs 1 39.2, bs 8 80.3, bs 32 160.3 vs s1
46.5 / 87.8 / 163.6; s1 best at bs 16). Kept as the `sk` flag (off).

**Refuted (run 9bbdc0bc): the activation rows' small copies.** bs 8 GEMM1 without act rows 47.4
us vs 48.1 full; memory-only with them 39.2 vs 39.9 without.

**Tensor and ALU do not overlap even across warp-specialized warpgroups (spec probe a314237e).**
Even WGs issue wgmma, odd WGs run the decode-like ALU block (clocks per iteration):
RS N16: both 380-400 vs wgmma 120 / ALU 300 (sum 420); RS N64: 757-796 vs 287 / 600 (sum 880);
SS N64: 766-812 (sum 880); SS N64 with 2x ALU: 1410 vs 287 / 1204 (sum 1491). At most ~25% of
the tensor time hides. So the large-batch compute floor is decode + tensor, additive. Units
fix: wgmma m64n64k16 reached 1827 MAC/clk/SM = 89% of the 2048 dense bf16 peak (not 1024);
N16 only ~55%.

Judge check 21259fd1 (split K4 config added): 1.688 (bs 1 1.845 with s1k4 adopted, 44.9 us);
bs 8 88.1 us because the gate kept s1 over s1d this time (86.2/94.7 us on one routing,
108.2/108.7 on the other: not <= 98% on both) -> gate now: <= 98% over the two routings
summed and <= 101% on each. The judge prints its own floor: 11.4 / 37.3 / 52.5 / 70.8 /
101.3 / 108.9 us (bs 1/8/16/32/128/256); we are at 25-45% of it.

v7 N 64 (t8, W0, 2 CTAs/SM, 93-96 registers, no spills or C7512): correct and repeatable;
bs 256 360.7 us vs t2 368.0 / t4 334.0; worse at bs 32/128 (small experts pay a 64-wide tile).

The one measured fact that stands: v6 at bs 8 runs 0.031 SM-clk/weight math-only (no memory),
0.049 with memory, against a 0.034 static instruction count and a 0.032 HBM budget at 4.49 TB/s.

TMA token tiles for v7 (128B swizzle, K-major; config t2 = stages 304): judge 408b8505 1.714
(bs1 1.658, bs8 1.682, bs16 1.667, bs32 1.363, bs128 1.983, bs256 2.016).

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

### Large batch: wgmma width per chunk (v8), and why small batch overlaps badly (2026-10-23 morning)

- Per-tile cost fit on GPU routings (run 7cdd1063): GEMM1 4.20 / 5.40 / 6.76 us per expert
  chunk at N 16 / 32 / 64 (N 16 runs the tensor core at ~55%, N 32 / 64 at ~90%). v8 = 64-row
  chunks, wgmma width chosen per chunk by its live rows (16 / 32 / 64), one k-loop copy per
  width: bs 256 -9% to -15% vs t2 on four routings (hot experts get 123-256 tokens).
  Two launches by expert size (dual) did not beat it: a launch over few long items pays a
  whole item (~20 us: 288 KB at 10-15 GB/s per CTA). Judge 165ac36d: **1.741** (bs 256
  2.173 with v8, bs 8 1.689 with s1d kept by the summed gate).
- Refuted: one mbarrier arrive per warp instead of 256 per stage (probe and kernels: no change).
- Bandwidth probe with paced consumers (each stage held `hold` clk before release): when the
  consumers' rate is near HBM's, delivered bandwidth collapses (4 CTAs x 4 x 9 KB: 64% of the
  consumer-limited rate; 4 or 6 stages alike), less so with bigger copies (16 KB: 74%, 4 KB:
  50%); with consumers well above HBM it stays ~97%. v6 at bs 8 has math (35 us) ~ memory
  (32 us), so ~65-70% is expected: the 48 us measured. Levers: 16 KB copies; math well below
  memory.
- 16 KB weight copies (v6 SUB 2: two 128-k chunks per stage, one bulk copy each): 2-6% (s1dw
  bs 32 -10%); 8 KB copies issued in back-to-back pairs: no gain (probe 692595f8), so it is the
  request size. PDL (GEMM2 launched as a programmatic dependent, weights warmed in L2 before
  griddepcontrol.wait, GEMM1 producers trigger when the queue is empty): ~1.5 us at bs 1 only.
- Route compaction: the per-expert programs looped over 256-id chunks with a dependent load per
  pass (~0.6 us each): 10 us at bs 256, 8 at bs 128. One load of all ids + tl.histogram + one
  cumsum: 2.2 / 1.4 us (v7 route + gather: 13.5 -> 6.3 us).
- Judge 233c679f: **1.834** (bs 1 1.883, bs 8 1.809, bs 16 1.722, bs 32 1.434, bs 128 2.039,
  bs 256 2.218); plans s1k4 / s1dw / s1dw / s1dw / s2w / v8.
- Refuted: GEMM2-only split K (kparts2) for bs >= 8: GEMM2 stage unchanged (26.2 -> 25.3-26.4
  us at bs 8) while post-norm grows 2-4 us; only bs 1 gains (47.0 vs 51.7). GEMM2 is not
  stream-starved; it sits in the same compute ~ memory regime.
- 32 KB copies: s2q (16 tokens, four-chunk stages, 2 CTAs/SM) bs 8 77.9 vs s2w 90.6, bs 128
  217-219 vs 230 us; MT 1 s1dq ~ s1dw.
- Exhaustive layout search (repack freedom): no bit layout makes two of the four unpack outputs
  pure left shifts (IMAD, FMA pipe) with the fourth a multiply-gather, nor three of them
  shifts/rotations; the present (direct, rot 10, rot 13, multiply) is the floor for moving ALU
  work onto the FMA pipe.
- Refuted at bs 1: warming L2 with GEMM2's weights from GEMM1 (GEMM1 +9 us, GEMM2 ~-1 us: GEMM2
  at bs 1 is round-trip bound, not bandwidth bound); split configs whose parts fit the stage
  ring (no gain). Kept: GEMM1 split parts reduced by the last-arriving CTA (fused, no reduction
  kernel) + GEMM2 as a PDL dependent: bs 1 -5..-9% on one routing, neutral on another, +2-3% at
  bs 4-8 (candidate config s1dw2fp). Post-norm kernels made branch-free (slot loads overlap).
- Decode microbenchmark variants (run 6456df78): rotations on the FMA pipe (IMAD lo + IMAD.HI,
  merged by one LOP3) are slower with the MMA in the loop (0.0273-0.0287 vs 0.0236-0.0241
  SM-clk/weight); scale build via IMAD instead of LEA: -5% decode-only, nothing with the MMA.
  The real loop (math only) runs 0.031: ~11% more instructions (LDS / ldmatrix / barriers) plus
  stalls. At 4.49 TB/s the HBM budget is 0.0327 SM-clk/weight: the microbenchmark alone is at
  72% of it, so small batches stay in the compute ~ memory regime.
- Hybrid at bs 256 (run a9364a88): experts > 16 routes on v8, the rest on s2q as v8's PDL
  co-runner (no start wait; waits for v8 before exiting, so the next launch sees both): 242 /
  252 / 240.5 us vs v8 249 / 270 / 251.5 (-3..-7%); loses at bs 64-128 (config s2qv16).
- decode_layer: the CUDA expert path bundled with its own gate (configs s1k4 s1dw s2w s2q v8
  s2qv16), tested inside a prefill-board host (same layer weights) via exec of the submission.
- In-kernel SM clock (clock64 / globaltimer per item, run dfe780fd): 1.70-1.79 GHz under our MoE
  GEMMs at bs 8 / 32 / 256, full and math-only alike (1.76 typical; 1.70 in bs 256 GEMM2), not
  the 1.98 spec (matches Guild post #12, power cap). Static instruction counts converted at
  1.98 GHz were ~11% optimistic; µs-based conclusions unchanged.

## Bottleneck table, judge stage profile (prefill-host check, Oct 23 evening)

Each stage as its own CUDA graph, cold L2, adopted configs; % of 4.06 TB/s:

| bs | experts (max tok) | GEMM1 | GEMM2 | route | post | total |
|---|---|---|---|---|---|---|
| 1 | 2 (1) | 19.1 us 1.98 TB/s | 11.2 us 1.69 TB/s | 1.0 | 6.0 | 41.3 |
| 8 | 10 (3) | 59.7 us 78% | 40.6 us 57% | 0.5 | 3.1 | 99.4 |
| 16 | 6 (16) | 45.7 us 61% | 24.4 us 57% | 0.7 | 3.1 | 77.0 |
| 32 | 14 (24) | 80.5 us 81% | 51.3 us 64% | 0.8 | 4.0 | 131.2 |
| 128 | 27 (56) | 144.2 us 87% | 81.2 us 77% | 1.4 | 4.9 | 234.6 |
| 256 | 27 (46) | 154.6 us 81% | 94.6 us 66% | 5.2 | 4.9 | 265.3 |

SASS (v6 main loop, 8192 weights/warp/iteration): 544 issue, 300 ALU (600 clk), 128 HMUL2 ->
at 4.06 TB/s the ALU needs 57% and issue 52%: the unpack has ~1.8x headroom; the gap is overlap.

1. GEMM2 57-77% everywhere (half the bytes: its ramp/tail weigh double; starts cold after GEMM1).
2. Hot experts re-decoded per 8-token tile (bs16: one 16-token expert, GEMM1 61%): pick the tile
   width per expert.
3. Fixed kernels 4-7 us (bs1 split-K post-norm 6 us; bs256 route+gather 5.2 us).
4. Then overlap (bytes in flight, decode next chunk under current MMAs).
