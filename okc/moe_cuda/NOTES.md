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

## dualsab refuted; the small-batch GEMMs sit on the per-SM decode rate (run d89995c7, Oct 23)

Stage profile self-hosted in an attention check (synthetic weights, judge routing seeds, cold L2):

| bs | config | total | route / GEMM1 / GEMM2 / post (us) |
|---|---|---|---|
| 1 | s1k4 (adopted) | 42.7 | 0.8 / 19.0 / 11.6 / 5.9 |
| 1 | s1dw2fp | **39.0** | 0.4 / 17.6 / 12.6 / 4.5 |
| 8 | s1dw (adopted) | 98.5 | 0.6 / 58.8 / 40.7 / 3.1 |
| 8 | s1dws8 | 104.3 | 0.6 / 61.7 / 39.2 / 3.1 |
| 16 | s1dw (adopted) | 76.8 | 0.6 / 45.4 / 24.4 / 3.1 |
| 16 | s1dws8 | 77.2 | 1.1 / 46.5 / 25.4 / 3.1 |
| 32 | s2w (adopted) | 130.8 | 0.9 / 80.5 / 50.6 / 4.2 |
| 32 | s1dws8 | 133.8 | 1.4 / 80.4 / 47.7 / 4.5 |
| 128 | s2q (adopted) | 234.3 | 1.1 / 144.4 / 83.7 / 4.6 |
| 128 | s1dws8 | 241.4 | 1.3 / 147.0 / 84.6 / 4.7 |

- Refuted: hot-expert re-decode as the bs 16 cost. Decoding the 16-token expert once (16-token
  tiles, flag 32 so the co-runner starts at once) leaves GEMM1 at 46.5 vs 45.4 us. s1dws8 dropped
  from the gate (the dualsab option stays in moe_cuda_stages).
- Model that fits: GEMM time ~ weights x 0.031 SM-clk (the real v6 loop) / 132 SMs / 1.76 GHz +
  ~6 us: bs 8 GEMM1 378 M weights -> 50 + 6 vs 58.8 measured. The HBM budget is 0.0327: the decode
  loop runs at ~95% of the bandwidth roofline, so any stall drops below it; rebalancing bytes
  (stream-K, split K, dualsab) cannot help. Lever: the loop's 0.031 -> the 0.024 the
  microbenchmark reaches with the MMA (loads / ldmatrix / barriers / stalls, ~23%).
- s1dw2fp wins bs 1 by 8.7% here (the gate already carries it).
- aaln's tip (compile-time ring stage indices) is already how v6's consumer runs (unrolled st).

## Correction: the bs 8-32 GEMMs lose to wave quantization, not the decode rate (run 74e36e12)

s1dw ablations (profiling build, cold L2, synthetic weights, judge-like routing), bs 8 GEMM1 (7 experts, 448 items, 264 slots):

| variant | span | TB/s | slot use | item med |
|---|---|---|---|---|
| full | 45.1 us | 2.93 | 64% | 18.2 us |
| no math (memory only) | 41.2 | 3.21 | 62% | 15.9 |
| math only | 33.5 | 3.94 | 73% | 11.3 |
| decode only | 22.3 | 5.93 | 69% | 7.7 |
| mma only | 18.9 | 6.97 | 74% | 8.4 |
| loop only | 15.9 | 8.32 | 69% | 7.2 |

- Memory alone takes 41 of the 45 us: 448 equal items on 264 slots run in two waves (starts p90
  24.6 us), the second ~70% full. Per CTA an item streams at ~18.5 GB/s (264 x = 4.9 TB/s > HBM),
  so with equal bytes per CTA GEMM1 would sit at the HBM ceiling: ~132 MB / 4.3 TB/s = 31 us + fill.
  bs 32 the same (slot use 78%, starts p50 26 us). The "0.031 clk/weight decode rate" model above
  is wrong for these sizes: math hides under memory (45.1 vs 41.2).
- Next: stream K (sk, equal 72 KB units per CTA) on the current v6 (two-chunk stages) - the old
  refutation was on an earlier kernel generation. Probe moe_sk_attn (run after 74e36e12).
- Stream K on today's v6 (run ff6e0fea, stage times, cold L2): still a loss. bs 8 GEMM1 60.1 ->
  63.2 (sk2) / 63.5 (sk3), post 3.3 -> 5.1-5.8; bs 16 45.7 -> 49.9; bs 32 81.2 -> 91.4; bs 128 s2q
  144.4 -> 181.9. Balanced bytes per CTA did not buy the slot-use gain the ablations promised.
  Note the stage time (60.1) vs the in-kernel item span (45.1): ~15 us sits outside the items
  (launch, first fill, exit) - probe moe_skprof_attn times the launches with events beside the
  span, for s1dw and s1dwsk2, full and memory-only.
- Profiled sk (run after ff6e0fea): sk's records are not comparable (item layout), and the eager
  event times carry Python launch gaps (useless); but memory-only sk 51.8 vs s1dw 50.4 us eager:
  balancing does not help even without math, and producers wait on the ring 69-95% of the time.
  Suspect: the item boundary. The producer resolves claim -> active -> exp_start -> row_route (four
  dependent global round trips, several us under load) before issuing the new item's first copy,
  with only 4 stages (~4 us) buffered; sk adds boundaries. Fix (flag 64, option mpf, configs *pf):
  fetch one step of the next item's chain per stage over the current item's last PF_LEAD = 5
  stages, claim kept late so balancing is unchanged. Compiles (sm_90a), registers unchanged.
  A/B probe moe_pf_attn.
- Refuted (run after moe_pf_attn): the item-boundary metadata chain. mpf (next item's metadata
  fetched over the current item's last stages) is neutral everywhere: bs 1 42.7 / 42.9, bs 8
  GEMM1 61.1 / 61.1, bs 16 47.2 / 47.9, bs 32 81.9 / 82.1, bs 128 145.2 / 144.2. Correct, no hang;
  the *pf gate candidates are removed again (option kept).
- Open: GEMM1 bs 8 stage 61 us vs a 45 us item span. Profiling build now stamps per-CTA entry
  and consumer exit (PCTA slots before the item records); probe moe_cta_pf prints the CTA
  timeline (entry -> first producer issue -> first data -> last item end -> exits).
- CTA timeline (run 6774bc77, bs 8 s1dw GEMM1, times from the first CTA entry): all 264 CTAs enter
  within 0.3 us; producer first issue 2.0-2.6 us (prologue + metadata, cold); first data at the
  consumers 7.4 us p50 (first fill ~5 us cold); last item end 48.1; last exit 48.4. The stage time
  (graph replay minus a tiny-kernel graph) is 60.3: ~12 us lies outside the CTAs' lifetime. GEMM2
  the same (27.4 life vs 37.6 stage). Two GEMMs -> ~20 us of the ~100 us bs 8 MoE, if real.
  Probe moe_launch_attn: [globaltimer stamp] -> GEMM -> [stamp] in one graph, cold L2, plus the
  same GEMM with no items and a stamp-only graph.

## PDL chain (flag 128 / chain=True): judge GPU, run df7fe776 (decode_layer-hosted, triton 3.8.0, torch 2.14.1+cu130)

route -> GEMM1 -> (split-K reduction) -> GEMM2 -> post-norm as programmatic dependent launches,
each signalling at its start; GEMM1 waits for the route at its start; GEMM2 issues its first
item's weight + scale stages before its wait (only activation rows wait); post loads route data
and norm weights before its wait. Full-forward graph timing, cold L2, standard seeds, two rounds
(second in reversed order):

| bs | incumbent | chained | change |
|---|---|---|---|
| 1 | s1k4 42.4 / 42.6 | s1k4c 39.4 / 41.2 | -5% |
| 1 | s1dw2fp 38.5 / 37.7 | s1dw2fpc 38.5 / 38.0 | 0 (GEMM2 was already a PDL dependent) |
| 8 | s1dw 99.5 / 98.4 | s1dwc 93.2 / 93.2 | -5.8% |
| 16 | s1dw 77.8 / 78.1 | s1dwc 72.3 / 72.8 | -6.9% |
| 32 | s2w 132.9 / 132.2 | s2wc 126.4 / 126.7 | -4.6% |
| 128 | s2q 239.2 / 237.7 | s2qc 237.1 / 236.5 | -0.7% |

All correct (rel diff vs v6 s2 identical to the unchained configs, bitwise repeatable). ~5-6 us
saved per call at bs 1-32: about a third of the ~20 us measured outside the GEMMs' CTA lifetimes.
Not yet overlapped: GEMM1's first fill (its weights depend on the route).

## Oct 24: launch gap resolved; PDL chain adopted (gate check 56dc4994: 1.889)

- In-graph launch probe (run 59fa2c2e, bs 8 s1dw, stamp kernels around one GEMM in a graph after
  the cold-L2 flush): stamp -> first CTA entry 0.8 us, CTA life 59.1 (GEMM1) / 37.6 (GEMM2), last
  exit -> stamp 0.8; empty GEMM 2.8 us total; no-kernel graph 0.8. There is no launch-side gap:
  the earlier "~12 us outside the CTAs" came from eager profiling, where the GEMM ran after a
  Python launch delay. Most likely cause of the eager vs graph difference (48 vs 59 us CTA life):
  the 256 MB zero-fill flush leaves L2 full of dirty lines; in graph replay the GEMM starts right
  away and its reads compete with ~50 MB of write-back (~11 us at 4.5 TB/s). Effective roofline =
  (weights + ~50 MB) / BW: bs 8 ~74 us. (The bs 128 part of the probe failed: no profiling
  build for s2q's stage count.)
- Builder A's PDL chain (chain=True): gate check 56dc4994 1.889 vs 1.825 (174486, same day,
  without chain); plans s1dw2fpc / s1dwc / s1dwc / s1dwc / s2q / s2qv16 for bs 1 / 8 / 16 / 32 /
  128 / 256 at 43.4 / 78.5 / 105.7 / 151.7 / 244.2 / 268.8 us.

## Oct 24: judge speed-of-light and the pure-streaming floor

- Judge records carry speed_of_light_ns per case (check 56dc4994): decode 1 / 16 / 32 = 11.4 / 52.5 /
  70.8 us, verify 1 / 16 / 32 = 37.3 / 101.3 / 108.9 us; ours 39.6 / 102.9 / 140.1 / 75.3 / 219.9 /
  273.7 -> 29 / 51 / 51 / 50 / 46 / 40% of SOL. SOL = routed weight bytes at 4.8 TB/s (bs 1 matches
  two experts exactly). Leaders' 2.18-2.34 ~ 55-60% of SOL. Scored run c5681b25: 1.879 (chained).
- Pure streaming under judge conditions (run 99a005b0: graph replay after the 256 MB zero-fill
  flush, overhead subtracted; per-CTA contiguous, balanced): 38 MB 13.0-14.0 us TMA / 10.7 LDG; 66 MB
  21.2-21.8 / 17.9; 132 MB 37.7-38.8 / 33.4; 264 MB 66.8-68.0 / 66.6; 523 MB 122.9-123.9 / 129.8
  (4.24 TB/s). Ring shape (2x4x16K, 2x2x32K, 4x4x8K, 1x8x16K, 2x6x16K) makes no difference. Our
  GEMMs at the same bytes: GEMM1 bs 8 (132 MB) ~59 us vs 38; GEMM2 (66 MB) ~37 vs 21.6: the
  pipeline leaves ~35-40% on the table. Small sizes: LDG starts faster than TMA (10.7 vs 14.0 at 38 MB).
- Next: graph-timed ablations of s1dw (full / no math / no act rows / weights only / math only /
  decode only / loop only), probe moe_gabl_pf.
- Graph-timed ablations (run 2db7fe47, s1dw, judge conditions; this seed routes bs 8 to 10 experts,
  so GEMM1 189 MB - the 132 MB comparison above used the wrong expert count):
  bs 8 GEMM1 full 60.1 / no math 54.2 / no act rows 58.4 / weights only 53.8 / math only 41.6 /
  decode only 28.0 / loop only 20.8 (floor at 189 MB ~50); GEMM2 37.4 / 32.8 / 37.0 / 32.4 / 24.8 /
  17.0 / 15.8 (floor at 94 MB ~28). bs 32 GEMM1 (246 MB) 99.2 / 79.9 / 93.2 / 75.1 / 81.3 / 53.2 /
  37.7 (floor ~63); GEMM2 49.8 / 41.1 / 46.6 / 39.6 / 39.8 / 27.4 / 23.4 (floor ~35).
  -> small batch: memory path within 7-14% of the streaming floor, math adds ~6 us. bs 32: math
  (81) > memory (75), poor overlap (99), and loop-only plumbing is 38 of math's 81 us. Per case vs
  a fused-streaming floor: bs 1 48%, bs 8 69%, bs 16 66%, bs 32 62%, bs 128 54%, bs 256 46%;
  geomean ceiling ~3.3x. Next: 32 rows per consumer warp (B-fragment reuse: today all 8 warps
  ldmatrix the same activations every k step).
- rb 2 (two 128-row blocks per item, 32 rows per consumer warp; run 71f58265, chained, graph stage
  times): correct everywhere (rel diff 0, repeat equal). bs 8 s1dwc 91.4 vs s1r2c 111.8 / s1r2wc
  104.8; bs 16 72.8 vs 91.4 / 82.7 (GEMM1 47.1 -> 43.9 but GEMM2 25.2 -> 36.6: 16 row-block groups,
  ~112 items on 264 slots); **bs 32 135.8 vs 136.0 / 124.5 (s1r2wc -8.3%: GEMM1 86.1 -> 78.2)**;
  bs 128 s2qc 238.6 vs s1r2c 302.4 / s2r2c 267.6 (MT 2 rb 2 spills). Gate candidates s1r2c,
  s1r2wc; gate check after.
- Gate check e8a1ed39 (rb 2 candidates): not adopted anywhere. bs 32 s1r2wc 163.6 vs 151.6 on
  routing 1 (+8%), 127.2 vs 133.2 on routing 2 (-4.5%): the single sim seed (8950030) favoured it.
  Score 1.879 (= chained build). Lesson: probes must time >= 2 routings per size, as the gate does.

## Oct 24: big swings

- H1 refuted (run after 71f58265, two routings per size): L2 prefetch distance (l2d, the producer
  prefetching chunk kc + D as it issues kc) is worse everywhere and monotonically with D: bs 8
  GEMM1 56.7 -> 61.9 / 65.7 / 72.6 (D 2 / 4 / 8), bs 32 83.8 -> 89.1 / 91.2 / 98.0, bs 16 (routing
  2) 100.1 -> 107.6 / 115.4 / 120.4; bs 128 D 8 equal only because s2q items have 8 stages (never
  fires). The math / memory overlap gap is not a shortage of bytes in flight.
- H2 (tiny: one persistent grid for both GEMMs at bs <= 8) correct but no gain. bs 1: resident W2
  39.2-40.5 vs s1dw2fpc 38.3-39.2; W2 via L2 + 8 / 16-stage ring 39.0-39.2 / 41.0-41.8; 2 CTAs/SM
  42.6; 3-expert routing worse (55 vs 49.5). Phase stamps (graph, cold L2): GEMM1 phase 24-27 us
  (W13 + the concurrent W2 prefetch: ~2.2 TB/s), expert wait ~1 us, GEMM2 phase 5.4 us (vs 12.5 as
  its own kernel). GEMM2's saving is eaten by GEMM1 sharing HBM with W2: both designs move the
  56.6 MB at ~2.2 TB/s; at these sizes the ramp dominates. Realistic bs 1 floor ~25-30 us. Closed.
- Large-batch graph ablations (s2q, run after the tiny closing): bs 128 GEMM1 (510 MB) full 150.4 /
  no math 129.8 / weights only 126.4 / math only 123.1 / decode only 80.7 / loop only 65.3 (floor
  ~120); bs 256 GEMM1 209.2 / 162.0 / 148.3 / 177.8 / 117.3 / 96.9; GEMM2 bs 256 109.2 / 81.2 /
  74.6 / 91.3 / 59.5 / 51.9. Memory path near the floor; math is the limit at bs 256 and the bare
  loop (loads, ldmatrix, barriers) grows with tokens: per-warp activation re-reads ~4 B of smem per
  weight byte -> ~166 B/clk/SM needed at bs 256 vs 128 available.
- H4 refuted in full runs: rb 2 with activation fragments loaded once per chunk and shared by both
  row blocks (MT 2: 95 registers, no spills; one accumulator chain per block, rel diff 5e-4,
  repeatable): bs 128 245.2 / 239.2 vs s2qc 241.9 / 225.7; bs 256 280.5 / 301.5 vs s2qv16 270.6 /
  268.4; bs 32 mixed. Cutting activation reads moves the limit to act-row copies (+14 us at bs
  256: 16 x 1 KB copies per stage) and math / memory overlap (+47 us over memory-only).
- v7 / v8 (wgmma: activations read once per warpgroup, TMA token tiles) vs incumbents at large batch
  (run 56892927, two routings): bs 128 s2qc 241.2 / 227.2 vs t2 271.3 / 232.4, t4 282.1 / 250.0, v8
  269.3 / 242.8; bs 256 s2qv16 270.2 / 269.2 vs t2 287.8 / 296.5, t4 313.4 / 314.9, v8 280.8 / 276.0.
  Removing the activation re-reads / small copies does not move large batch: the limit is the decode
  (~88-117 us of ALU at bs 256) not overlapping the ~120 us memory stream. The wgmma redesign is not
  pursued.

## Oct 24: prior art and the overlap re-check

- Prior art read: LiquidGEMM (W4A8, Hopper: two compute warpgroups ping-pong, each dequantizes into
  registers and issues wgmma; a load warpgroup does TMA), Machete (= our v7: swap-AB, RS wgmma,
  prepacked weights; wins only at >= 128 tokens), Marlin FP4 decode (~18 instructions per 8 values:
  we use ~12), TRT-LLM weightOnlyBatchedGemv (CUDA-core HFMA2, weights LDG.128 straight to registers:
  no shared memory, no barriers; decode once, reuse for every token; warp-shuffle reduction). The
  sglang baseline on Hopper W4A16 NVFP4 is Marlin.
- Overlap re-check by instruction type (run after 56892927, one wgmma warpgroup + one ALU warpgroup per
  CTA, 2 CTAs/SM, clocks per iteration, both / wgmma only / ALU only): decode mix vs RS N64 502 / 286 /
  307 (32% of the smaller hidden); LOP3 587 / 286 / 405 (36%); IMAD 543 / 286 / 378 (42%); HMUL2 560 /
  286 / 213 (worse than serial); FFMA 504 / 286 / 214 (0%). The old claim holds: ALU and tensor work
  largely serialize on the judge's H200. Ping-pong (LiquidGEMM-style) is off the table; compute ~
  plumbing + decode + MMA, and the plumbing (smem ring, LDS, ldmatrix, barriers) is the largest part
  at large batch -> next: TRT-GEMV-style direct-to-register weight streaming, micro-benchmark first.
- GEMV upper bound (TRT-style: LDG.128 straight to registers, our decode + scale, T token FMAs with
  activations assumed free; judge timing): 132 MB bf16x2 HFMA2 T1/2/4/8 2.17 / 2.14 / 1.56 / 2.00
  TB/s, fp32 FFMA 1.81 / 1.68 / 1.80 / 1.28; 523 MB at most 2.02. Flat in T: the load + decode path,
  not the FMAs. Slower than our TMA ring (~3+ TB/s): refuted.
- FP8 compute is out: the MoE tolerance is 1.5% relative L2 (rows 7.5%); FP8 rounding of
  activations or of e2m1 x scale products adds ~2% per stage (we are at ~0.4%), and NVFP4's 16-k
  block scales cannot be applied inside FP8 wgmma's k32.
- Every published Hopper W4A16 design is now measured against ours under judge conditions; ours is
  the fastest. The leaders' edge is not a published design.
- bs 1 deeper split K refuted (three routings): GEMM1 17.7 / 19.0 (k2) vs 23.1 / 23.6 (k8); GEMM2 -1 us
  but post +3 us. bs 1 graph ablation: GEMM1 memory-only 16-17 us (= its full 18.4: memory path),
  GEMM2 math-only 11.5 > memory 8.7 (latency-bound, ~0.5 us per chunk per CTA).
- Judge weights (run after the split-K probe, prepare-time stats): FP4 codes exactly uniform (4.000
  bits entropy: synthetic random, incompressible); E4M3 block scales ~31 distinct values per tensor
  (entropy 3.71 bits), per-row range mean 18 / max 29 codes; global scales all 3.85e-5. A lossless
  scale repack (base byte per 8 scales + 4-bit deltas, exact fallback) saves ~37% of scale bytes =
  ~4% of HBM bytes. Real but small.
- Atomic post (apost: GEMM2 partials by red.add into fp32 y, the chunk's last item RMS-norms its rows
  into the tokens, GEMM1 zeroes the accumulators; three routings): correct (rel diff <= 1.5e-4, mostly
  repeatable). First epilogue (one warp per row, serial cold loads) made GEMM2 88 us; CTA-wide float4
  version: bs 1 39.8 / 39.0 / 49.8 vs 39.8 / 39.9 / 49.2 (equal), bs 8 +6-16%, bs 16 +12-36%. The
  norm on GEMM2's critical path costs more than the separate post kernel (parallel over tokens,
  launch overlapped by the chain). Refuted.
- stage_release now fences the async proxy (fence.proxy.async.shared::cta) before releasing a ring
  slot (Guild #42): a correctness fix, cost ~0 on the judge.
- MT 4 (four 8-row token tiles per CTA, 1 CTA/SM): slower at every size (bs 128 s4qc 262.9 vs s2qc
  238.6 us). Refuted.
- Leaderboard (2026-10-25): #1 2.523, #2-#10 2.387 -> 2.335, 14 teams in 2.31-2.39; us 1.879 (124th).
  The tight cluster suggests a shared approach rather than one team's trick.
- Prior art read: chord/humming (H200 decode swap-AB mma.sync, warp-N 64, 3 CTAs/SM, cp.async ring
  4-6, semi-static token tiles, block-M from the routing; ~3.2-3.6 TB/s, level with ours). MonoMoE
  (arXiv 2609.04244): weight-major persistent megakernel, 8 compute + 4 prefetch warps, TMA several
  items ahead across expert boundaries, readiness flags between up and down, atomicAdd down partials,
  epilogues on the prefetch warps, routing/top-k fused; 1.17-1.54x over vLLM grouped GEMM at B 1-8.
- LDG-direct small-batch path refuted (prefill-host check e760ef1d, ldg_probe: LDG.128 straight into
  mma.sync A fragments, D-deep register ring per warp, real decode + MMA, judge timing): bs 1 GEMM1
  (37.7 MB) best 18.0 us vs plain LDG 11.2 and our TMA ring ~17.6; GEMM2 (18.9 MB) 12.2 vs 6.0 / ~12;
  bs 8 GEMM1 (189 MB) 66.8 vs 48.8 / ~59; GEMM2 (94 MB) 38.3 vs 25.2 / ~37. No better than the ring:
  the streaming engine is not what holds small batch back; decode + MMA under the stream is.
- DRAM locality refuted (check 2a183b6a): per-warp contiguous regions stream as fast as grid-stride
  (bs 8 G1 189 MB: 49.0-49.9 vs 48.8 us; 38 MB 11.0 vs 10.9); an interleaved [k block][tile] weight
  layout leaves the LDG+MMA kernel unchanged (65.6 vs 66.9). The loss is math under the stream
  (~0.5 x the decode time adds on top of memory), not the access pattern.
- Fused GEMM1 + GEMM2 (MODE 3, MonoMoE-style one-launch queue; attention-host check 233a7491, two
  routings, us): bitwise equal to v6. bs 1 s1dw2fpc 49.1 / 38.8 vs f1dwc 50.6 / 41.8; bs 8 s1dwc
  69.7 / 89.7 vs f1dwc 72.8 / 85.6; bs 16 72.4 / 85.8 vs 76.5 / 80.1; bs 32 141.8 / 137.1 vs f1dwc
  141.3 / 136.7, f2wc 134.4 / 134.5; bs 128 s2qc 254.6 / 258.3 vs f2qc 248.9 / 258.9. Neutral within
  noise: the GEMM boundary (second ramp / wave tail) is not where small batch loses. Kept as an
  option (fused=True), not adopted.
- Gate check 53d0fedc (fused configs f1dwc / f2wc / f2qc as candidates): adopted only at bs 128
  (f2qc 242.1 vs s2q 250.3 / 245.7 vs 249.0, ~2%); score in noise (decode 1 / 16 / 32, verify 1 / 16 /
  32: 1.824 / 1.756 / 1.462 / 1.836 / 2.145 / 2.259). Not submitted.
- Decode layout search (scratchpad lay*.py): no lossless bit layout moves the unpack's funnel shifts
  to IMAD (left-shift + mask for one / two outputs, multiply trick for the rest, <= 3-term
  multipliers): the current 8-op unpack (5 LOP3 + 2 SHF + 1 IMAD) stays. Main loop per 8 weights
  (SASS, s1dw): 6 LOP3, 2 SHF, 1 LEA, 1 IMAD, 4 HMUL2, 1 HMMA, 0.5 LDS, 0.5 LDSM: ALU-pipe bound
  (18.4 clk per warp-octet vs 16.7 issue).
- v8 / s2q ablations at bs 256 / 128 (attention-host check 0e41e7bf, synthetic weights, 25 / 21 experts,
  profiling build, cold L2), GEMM1 / GEMM2 span us:
  v8 bs 256: full 180.5 / 98.6, no math 134.7 / 76.3 (3.5 / 3.1 TB/s), math only 145.9 / 77.1,
  loop only (unpack kept, no scale/MMA/memory) 70.9 / 36.4; timed: data wait 18%, tensor wait 7%,
  decode 35%, fence + issue 19%; producers blocked on free slots 21%. SM clock 1.95 GHz memory-only,
  1.79 math-only, 1.69-1.73 full (power cap: math costs clock). v8 bs 128: 146.2 / 82.7 full,
  113.7 / 60.9 no math, 116.0 / 63.5 math only. s2q bs 256: 182.5 / 99.8 full, 140.3 / 71.2 no math,
  158.7 / 76.3 math only, decode only 108.3 / 55.0, loop only 93.2 / 49.9.
  -> large batch loses ~25-35 us per GEMM1 to math / memory overlap (full vs max), plus ~13% clock
  when math runs. v8's ring is shallow in weight bytes (activation tiles 16 KB of each 25 KB stage).
- Final scored run 8f564da9 (chained build + proxy fence + fused option in the gate): 1.869 (decode 1 / 16 /
  32, verify 1 / 16 / 32: 1.825 / 1.768 / 1.461 / 1.851 / 2.136 / 2.285), within noise of c5681b25 (1.879),
  which stays our board row. Work on MoE stopped here.
- Public search (Oct 25): no top-10 MoE approach is public. lee101/openkernels (participant notes: persistent
  tile queue, 11-op dequant, mma.sync, TMA) reports 1.418 (bs 1 48 us, bs 256 455 us); lilaclilac09 is a
  starter kit; PostTrainLLM #206 has MoE below SGLang at 32 tokens; vLLM Humming (Oct 21) claims ~2x Marlin
  for MXFP4 W4A16 MoE at medium-large batch on H200 (we are 2.1-2.3x there), small batch via cp.async +
  mma.sync, symmetric formats decoded as zero-exponent floats (our denormal trick). Guild: stronger trade
  offer to lurrobert/scout (we send the attention design first), pending.
- FP8 path refuted statically (moe_cuda/fp8_probe.py, SASS via tools/sass.py, no check spent): mma.sync
  m16n8k32 e4m3 is emulated on sm_90a (ptxas emits F2FP conversions + fp16 HMMA + FADDs: ~4 F2FP and 16
  FADD per MMA); the e4m3 decode-to-FP8 loop (two-word unpack ~5 ops / 8 weights, scale on the fp32
  accumulator) is conv-pipe bound at ~544 clk vs 278 (ALU) for the bf16 loop per 16 octets. Full-rate FP8
  on Hopper is wgmma only; there the per-16-k scale on the accumulator costs N/2 FFMA per thread per
  block, eating the decode saving. Also: no single shift / rotation / multiply maps FP4 nibbles to e4m3
  byte positions (lay8.py), so the FP8 unpack needs two rotations per 16 weights.
- Step tests (Oct 26; attention-host check 5b80e812, prefill-host check 578cd037):
  - v8 with an 8-stage ring at 1 CTA/SM (v8d): much worse (bs 256 GEMM1 214-216 vs v8 164 us; bs 128
    216-217 vs 162). Ring depth is not v8's limiter; occupancy is. Refuted.
  - SUB4 stages at MT1 (s1q, 2 stages x 4 chunks) vs s1dw (4 x 2): loop-only plumbing -7% (21.9 vs 23.5
    us bs 8 GEMM1) but full unchanged (59.7 vs 60.8; bs 32 90.3 vs 88.9); forward times equal. Refuted.
  - bs 1 CUDA-core GEMV (ldg_gemv: lane = row, unpack8 + x2^126 HMUL2 + HFMA2 per 16-value block into
    bf16x2, fp32 FMA by the block scale; no HMMA): GEMM1 (37.7 MB) 13.0 us vs mma 18.0 vs plain stream
    11.1; GEMM2 (18.9 MB) 7.5 vs 11.7 vs 6.1. Best K split 16, 2 CTAs x 512. bf16 sums over a 16-value
    block cost ~0.35-0.46% rel L2 per GEMM (CPU emulation). -> built: moe_gemv_kernel + repack_gemv /
    gemv_layout + moe_gemv_stages (configs g16 / g8), gate check pending.
- bs 1 GEMV in the pipeline (stage probes 5af555ca / 8be7d763 / ecccc31d, three routings): after issuing
  the first weights before the activation staging (GEMM2: before the PDL wait), GEMM2 reducing its own
  slices (rows post-norm 3.0 us instead of the parts post-norm 8-13 us) and prefetching the next unit's
  first block: g8 / g16 31.8 / 31.2, 32.3 / 31.1, 42.8 / 43.5 us vs s1dw2fpc 38.2, 39.1, 51.0 (-15-19%).
  Accuracy 7e-3 rel vs v6 (bf16 sums over 16-value blocks), judge test bs 1 7.1e-3.
- RACE (gate check 295cf18e): with one acq_rel atomic by lane 0 after __syncwarp (no per-lane fence),
  the judge's recheck failed on repetition 3 of decode 1 (rel error 0.755): the last-arriving slice read a
  stale part. Fixed with fence.acq_rel.gpu in every lane before the count and in the last arriver before
  the reads. New soak test (sim_src.run_soak): 2000 graph replays alternating two inputs, bitwise vs
  eager; a racy variant (flags 256) is soaked too to prove the detector.
- PDL stale-metadata bug (soak checks a5ebb192 / 8624b979): the chained GEMV returned wrong outputs on
  ~50% of replays when two inputs alternate (always the 3-expert input after a 2-expert one); unchained 0,
  the chained tensor-core configs 0. A rebuild that only changed the code shape around the metadata loads
  (branch on a test flag) passed 400/400, so the non-coherent (ld.global.nc / L1) loads of the route's
  output were returning the previous call's values under PDL. Fix: every read of routing metadata
  (n_active, active, exp_start, row_route) in the GEMV and v6 (sab) kernels goes through L2 (__ldcg).
  The sab chain configs never failed a judge recheck, but they read the same metadata the same way.
- Metadata via L2 cost (check fa4bc679): GEMV with the metadata staged once per CTA in shared memory: g16 31.8 /
  32.2 us, g8 32.7 / 33.0 vs s1dw2fpc 39.3 / 39.7 (-19%), soak 0 / 500 each. sab with ld.cg metadata: s1dwc
  bs 32 139.1 / 139.9 us (prior builds 141-142: no cost); mpf neutral (138.4 / 139.7).
- Scored run ecf830cd (GEMV g16 at bs 1): 1.878 (decode 1 2.043, decode 16 1.722, decode 32 1.436, verify 1
  1.826, verify 16 2.116, verify 32 2.247); decode 1 up from ~1.85-1.94, the rest within run-to-run noise.
  Board row stays 1.879. Leaderboard Oct 26: top-10 cutoff 2.406 (risen from 2.335); we are 218th.
- GEMV with T tokens per expert (check 3f67e102, bs 8 sizes, us): GEMM1 189 MB plain 49.2, mma loop 65.5, GEMV
  T1 53.6 / T2 60.1 / T3 73.8 / T4 86.8; GEMM2 94 MB plain 25.1, mma 39-40, GEMV 30.6 / 34.0 / 41.3 / 46.3. The
  GEMV wins only at 1 token (~9% over the floor) and ties the production loop at 2 (issue-bound: +4 HFMA2 + 1 LDS
  per octet per token). Next probe: drop the constant HMUL2 (x pre-scaled by 2^100) if HFMA2.BF16 keeps
  subnormal inputs.
- No-rescale GEMV (check 51ab608b): HFMA2.BF16 keeps subnormal inputs (all 16 codes exact with x * 2^100), so the
  constant HMUL2 can go (fold 2^-100 into the block scale): bs 1 GEMM1 13.1 -> 12.6 us; bs 8 GEMM1 T1 54.4 -> 53.2,
  T2 59.8 -> 57.1, T3 74.3 -> 68.9; GEMM2 T2 33.5 -> 32.2. Hybrid (1-2 token experts on the GEMV) worth ~+2% overall.
