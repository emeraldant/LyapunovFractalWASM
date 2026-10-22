"""Exact work accounting for the Open Kernel Challenge problems, with independent validation.

For every scored shape this counts, from the problem definitions:
  - HBM bytes: every tensor the computation must read or write at least once, by name, with
    a fully fused kernel (each input read once, each output written once), and per dependent
    phase where phases must exchange results through memory;
  - tensor-core FLOPs (BF16 / FP8), CUDA-core INT ops for NVFP4 dequant (from Marlin's
    instruction sequence), and SFU ops (exp2 / rsqrt / sigmoid);
then the time each resource needs at H200 peak rates derived from per-SM throughputs, and
the floors: HBM-only, all-resources (perfect overlap), and phase-serial (dependent phases
can't overlap).

Data-dependent quantities are expectations: request lengths (agentic spread) and, for the
MoE, the busiest GPU's routes and distinct experts, simulated with the judge's own generator
and cross-checked against the values the judge's floors imply (validation V5).

Validation (python3 tools/exact.py --validate):
  V1 the judge's own speed_of_light_ns reproduces every floor in TASK.md;
  V2 our byte and FLOP inventories, restricted to the judge's assumptions, equal the judge's;
  V3 tensor FLOPs recounted by torch's FlopCounterMode on meta tensors at the exact shapes;
  V4 byte inventories recounted from meta tensors of the exact shapes and dtypes;
  V5 routing statistics: simulation vs the values backed out of the judge's floors;
  V6 hardware peaks rebuilt from per-SM rates reproduce the spec sheet.
"""
import json
import math
import os
import re
import sys

OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ---------------------------------------------------------------------------------------
# H200 SXM from first principles (CUDA C Programming Guide, arithmetic instruction
# throughput for compute capability 9.0, per SM per clock; NVIDIA H200 datasheet).
SMS = 132
CLK = 1.830e9                 # the clock NVIDIA's dense-tensor TFLOPS figures imply (V6)
TC_BF16 = 4 * 512 * 2         # 4 tensor cores x 512 dense BF16 FMA/clk = 4096 FLOP/clk/SM
TC_FP8 = 2 * TC_BF16          # 8192 FLOP/clk/SM
FP32_LANES = 128              # FP32 add/mul/FMA results per clock per SM
INT_LANES = 64                # 32-bit integer add, shift, AND/OR/XOR (LOP3) per clock per SM
SFU_LANES = 16                # MUFU (ex2, rsqrt, ...) per clock per SM
HBM_STACKS, HBM_BITS, HBM_GBPS = 6, 1024, 6.25  # HBM3e: 6 stacks x 1024-bit x 6.25 Gb/s/pin
HBM = HBM_STACKS * HBM_BITS * HBM_GBPS * 1e9 / 8  # 4.8 TB/s
L2_BYTES = 50 * 2**20

RATE = {
    "hbm": HBM,                               # B/s
    "tc_bf16": SMS * CLK * TC_BF16,           # FLOP/s  (989.4 T)
    "tc_fp8": SMS * CLK * TC_FP8,             # FLOP/s  (1978.9 T)
    "fp32": SMS * CLK * FP32_LANES,           # lane ops/s (30.9 T)
    "int": SMS * CLK * INT_LANES,             # lane ops/s (15.5 T)
    "sfu": SMS * CLK * SFU_LANES,             # ops/s (3.87 T)
}
# The judge's two peaks.
JUDGE_HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}

# NVFP4 -> BF16 dequant in Marlin (marlin/dequant.h, dequant<nv_bfloat162, kFE2M1f, true>):
# per call, 4 values: Out = (q & 0x80008000) | ((q & 0x70007000) >> 6) is SHF + 2 LOP3, twice,
# plus q <<= 4 (7 INT ops), and the caller shifts q >> 8 once per two calls (0.5).
DEQUANT_INT_PER_VALUE = 7.5 / 4  # 1.875
# Block scale: one BF16x2 multiply per 2 values (on the FP16x2 pipe, 256 results/clk/SM: not binding).

# ---------------------------------------------------------------------------------------
# Model constants (from the problems' reference.py)
H, D_EXP, N_EXP, TOPK, LOCAL = 8192, 4096, 256, 8, 32
HQ, HKV, HD = 16, 2, 64
QKV_N = (2 * HQ + 2 * HKV) * HD     # 2304
O_K = HQ * HD                       # 1024
WINDOW = 2048
UNPOOL_SLICE, SLICED_MAX = 1024, 8
EXPERT_W13 = 2 * D_EXP * D_EXP // 2 + 2 * D_EXP * D_EXP // 16   # packed E2M1 + E4M3 scales: 18,874,368
EXPERT_W2 = D_EXP * D_EXP // 2 + D_EXP * D_EXP // 16            # 9,437,184
EXPERT_BYTES = EXPERT_W13 + EXPERT_W2 + 2 * 4                    # + two FP32 global scales
EXPERT_VALUES = 3 * D_EXP * D_EXP                                # 50,331,648 weights
AGENTIC = 0.075


def load_reference(problem):
    """The judge's reference.py for a problem, as its own module (with its checks.py)."""
    import importlib.util
    d = os.path.join(OKC, problem)
    sys.path.insert(0, d)
    for name in ("checks", "nvfp4"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(f"reference_{problem}", os.path.join(d, "reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def load_spec(problem):
    return json.load(open(os.path.join(OKC, problem, "spec.json")))


def task_floors(problem):
    """The floor column of TASK.md, in table order, µs."""
    text = open(os.path.join(OKC, problem, "TASK.md")).read()
    rows = re.findall(r"^\| ([^|]+) \| ([^|]+) \| (\d+) \| ([\d,.]+) µs \|$", text, re.M)
    return [float(f.replace(",", "")) for *_, f in rows]


def mean_len(ctx, spread):
    """Expected cached length: fixed, or floor(ctx * (1 + 0.075 (2u - 1))) with u ~ U[0, 1)."""
    return ctx if spread == "fixed" else ctx - 0.5


# ---------------------------------------------------------------------------------------
# Work records


class Work:
    def __init__(self):
        self.bytes = {}   # tensor name -> bytes (read + write)
        self.ops = {"tc_bf16": 0.0, "tc_fp8": 0.0, "fp32": 0.0, "int": 0.0, "sfu": 0.0}

    def io(self, name, nbytes):
        self.bytes[name] = self.bytes.get(name, 0) + nbytes
        return self

    def op(self, unit, n):
        self.ops[unit] += n
        return self

    @property
    def hbm(self):
        return sum(self.bytes.values())

    def times(self):
        t = {"hbm": self.hbm / RATE["hbm"]}
        for unit, n in self.ops.items():
            t[unit] = n / RATE[unit]
        return t

    def floor(self):
        return max(self.times().values())

    def __add__(self, other):
        w = Work()
        for src in (self, other):
            for k, v in src.bytes.items():
                w.io(k, v)
            for k, v in src.ops.items():
                w.op(k, v)
        return w


def gemm(name, m, n, k, w_bytes, x_bytes=2, out_bytes=2, unit="tc_bf16", reads_x=True, writes_out=True):
    w = Work().io(f"{name}.W", n * k * w_bytes)
    if reads_x:
        w.io(f"{name}.x", m * k * x_bytes)
    if writes_out:
        w.io(f"{name}.out", m * n * out_bytes)
    return w.op(unit, 2.0 * m * n * k)


def rmsnorm_ops(rows, n):
    # x*x accumulate (FMA), scale (mul), weight (mul): 3 FP32 ops per element; one rsqrt per row.
    return Work().op("fp32", 3.0 * rows * n).op("sfu", rows)


# ---------------------------------------------------------------------------------------
# Problems


def proj_shape(m, layer):
    n, k = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
            "router": (256, 8192)}[layer]
    if layer == "unpooler" and m <= SLICED_MAX:
        n = UNPOOL_SLICE
    if layer == "router":
        # FP32 weights, FP32 output. Up to 8 tokens SGLang runs FP32 FMAs; above, BF16 hi + lo
        # on tensor cores (2 GEMMs). Minimal: same 4 B/weight either way.
        w = Work().io("router.W", n * k * 4).io("router.x", m * k * 2).io("router.out", m * n * 4)
        if m <= 8:
            return w.op("fp32", 1.0 * m * n * k)  # FMA instructions (2 FLOPs each)
        return w.op("tc_bf16", 2 * 2.0 * m * n * k)
    return gemm(layer, m, n, k, 2)


def attention_work(batch, ctx, length, layer, spread, page=1):
    """One step of attention for the batch: QK-norm/RoPE, cache write, attention, gated norm."""
    seq = mean_len(ctx, spread)
    tokens = batch * length
    sliding = layer == "sliding"
    cached = min(seq, WINDOW - 1) if sliding else seq        # cached keys the block needs
    keys_per_query = [min(seq + j + 1, WINDOW) if sliding else seq + j + 1 for j in range(length)]
    w = Work()
    w.io("qkv", tokens * QKV_N * 2)
    w.io("k_cache.read", batch * cached * HKV * HD)           # FP8
    w.io("v_cache.read", batch * cached * HKV * HD)
    w.io("page_table.read", batch * math.ceil(cached / page) * 4)
    w.io("k_cache.write", tokens * HKV * HD)
    w.io("v_cache.write", tokens * HKV * HD)
    w.io("seq_lens", batch * 4).io("cache_loc", tokens * 8).io("kv_scales", 8)
    w.io("out", tokens * O_K * 2)
    pairs = batch * sum(keys_per_query)                       # (query token, key) pairs
    w.op("tc_bf16", 4.0 * pairs * HQ * HD)                    # QK and PV, per query head
    w.op("sfu", pairs * HQ)                                   # one exp2 per score
    w.op("fp32", 3.0 * pairs * HQ)                            # scale-subtract FMA, row sum, P conversion
    w += rmsnorm_ops(tokens * (HQ + HKV), HD)                 # QK-norm
    w += rmsnorm_ops(tokens * HQ, HD).op("sfu", tokens * HQ * HD)  # output norm, sigmoid(gate)
    if sliding:
        w.op("sfu", 2 * tokens * (HQ + HKV) * HD // 2)        # RoPE sin and cos
    return w, pairs


def moe_work(tokens, routes, distinct, tiles_per_expert=1):
    w = Work()
    w.io("experts.W", distinct * EXPERT_BYTES)
    w.io("x", tokens * D_EXP * 2).io("topk", tokens * TOPK * 8).io("post_norm_w", D_EXP * 2)
    w.io("out", tokens * D_EXP * 2)
    w.op("tc_bf16", 2.0 * routes * EXPERT_VALUES)
    w.op("int", distinct * EXPERT_VALUES * DEQUANT_INT_PER_VALUE * tiles_per_expert)
    w.op("sfu", routes * D_EXP)                               # silu's exp
    w.op("fp32", routes * D_EXP * 4)                          # silu * up
    w += rmsnorm_ops(routes, D_EXP)                           # post-expert norm
    w.op("fp32", 2.0 * routes * D_EXP)                        # weighted sum
    return w


def moe_sglang_extra(tokens, routes):
    """Intermediate traffic in SGLang's pipeline (Marlin GEMM1 -> silu_and_mul -> GEMM2 ->
    post-norm), excluding Marlin's FP32 split-K reduce buffer."""
    return {"cache1 (h) write+read": routes * 2 * D_EXP * 2 * 2,
            "cache2 (silu) write+read": routes * D_EXP * 2 * 2,
            "cache3 (y) write+read": routes * D_EXP * 2 * 2}


def layer_phases(batch, ctx, length, layer, spread, routes, distinct, page=1):
    """The decoder layer as dependent phases (each exchanges results through HBM)."""
    t = batch * length
    unpool_n = UNPOOL_SLICE if t <= SLICED_MAX else H
    att, pairs = attention_work(batch, ctx, length, layer, spread, page)
    att.bytes.pop("qkv")
    att.io("qkv (from phase 1)", t * QKV_N * 2)
    tiles = 1 if routes / max(distinct, 1) <= 128 else math.ceil(routes / distinct / 128)
    phases = [
        ("qkv_gate", gemm("qkv_gate", t, QKV_N, H, 2, writes_out=True)),
        ("attention", att),
        ("o + boundary", gemm("o", t, H, O_K, 2, writes_out=False)
         .io("stream.read", t * H * 4).io("stream.write", t * H * 4).io("ffn_input.write", t * H * 2)
         .io("norm weights", 2 * H * 2) + rmsnorm_ops(2 * t, H)),
        ("router + top-8", proj_shape(t, "router") if t > 0 else Work()),
        ("pooler + pre-expert norm", gemm("pooler", t, D_EXP, H, 2, reads_x=True) + rmsnorm_ops(t, D_EXP)),
        ("experts", moe_work(t, routes, distinct, tiles)),
        ("unpooler + boundary", gemm("unpooler", t, unpool_n, D_EXP, 2, writes_out=False)
         .io("stream.read", t * H * 4).io("stream.write", t * H * 4).io("next_attn_input.write", t * H * 2)
         .io("norm weights", 3 * H * 2) + rmsnorm_ops(2 * t, H)),
    ]
    return phases, pairs


# ---------------------------------------------------------------------------------------
# Routing statistics


def simulate_routing(tokens_list, seeds=200):
    """Busiest-GPU local routes and distinct experts under the MoE problem's generator."""
    R = load_reference("muse_moe_w4a16")
    out = {}
    for bs in tokens_list:
        rs, ds = [], []
        n = seeds if bs <= 256 else max(8, seeds // 20)
        for seed in range(n):
            d = R.generate_input(bs, "ep8", 10_000 + seed, device="cpu")
            loc = d["topk_ids"][d["topk_ids"] < LOCAL]
            rs.append(loc.numel())
            ds.append(loc.unique().numel())
        mr, md = sum(rs) / n, sum(ds) / n
        sr = (sum((r - mr) ** 2 for r in rs) / n) ** 0.5
        sd = (sum((d - md) ** 2 for d in ds) / n) ** 0.5
        out[bs] = (mr, md, sr, sd)
    return out


def backsolve_distinct(problem):
    """router_distinct per scored shape, solved from the judge's floor (memory-bound shapes):
    floor = bytes / 4.8 TB/s with touched = max(min, 0.75 * distinct)."""
    R = load_reference(problem)
    spec = load_spec(problem)
    floors = task_floors(problem)
    out = []
    for case, floor in zip(spec["benchmarks"], floors):
        kw = {k: v for k, v in case.items() if k not in ("load", "weight", "seed")}

        def f(distinct, load=0.0):
            return R.speed_of_light_ns(hw=JUDGE_HW, router_distinct=distinct, router_load=load, **kw) / 1e3

        lo, hi = 0.0, 64.0
        if abs(f(hi) - f(lo)) < 1e-9:   # not memory-bound in experts: solve router_load instead
            lo, hi = 0.0, 1e6
            for _ in range(100):
                mid = (lo + hi) / 2
                (lo, hi) = (mid, hi) if f(0.0, mid) < floor else (lo, mid)
            out.append(("load", (lo + hi) / 2, floor))
            continue
        for _ in range(100):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if f(mid) < floor else (lo, mid)
        out.append(("distinct", (lo + hi) / 2, floor))
    return out


# ---------------------------------------------------------------------------------------
# Tables


def fmt_us(s):
    return f"{s * 1e6:,.1f}"


def resource_cols(w):
    t = w.times()
    bind = max(t, key=t.get)
    return t, bind


STEP = []  # (problem, load, weight, exact floor s, judge floor µs)


def step_summary():
    """Per decode step (prefill load), the judge's way: sum weight x time per load, geometric
    mean over loads."""
    import collections
    out = ["| problem | SGLang µs | leader (Oct 22) | judge floor µs | exact floor µs | SGLang / exact floor |",
           "|---|---|---|---|---|---|"]
    sglang = {"muse_bf16_proj": (4830.8, 1.203), "muse_attention_decode": (3766.3, 1.263),
              "muse_moe_w4a16": (13448.5, 1.309), "muse_decode_layer": (22620.3, 1.242),
              "muse_prefill_layer": (387403.9, 1.097)}
    for problem, (sg, lead) in sglang.items():
        loads = collections.OrderedDict()
        for prob, load, weight, floor, judge in STEP:
            if prob == problem:
                e, j = loads.get(load, (0.0, 0.0))
                loads[load] = (e + weight * floor * 1e6, j + weight * judge)
        if not loads:
            continue
        geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
        ex = geo([v[0] for v in loads.values()])
        jd = geo([v[1] for v in loads.values()])
        out.append(f"| {problem} | {sg:,.0f} | {lead:.2f}x | {jd:,.0f} | {ex:,.0f} | {sg / ex:.2f}x |")
    return "\n".join(out)


def report(routing):
    lines = []
    P = lines.append
    P("## Hardware (derived)\n")
    P("| resource | per SM per clock | x 132 SMs x 1.83 GHz |")
    P("|---|---|---|")
    P(f"| HBM3e | 6 stacks x 1024 bit x 6.25 Gb/s | {RATE['hbm'] / 1e12:.2f} TB/s |")
    P(f"| BF16 tensor (dense) | {TC_BF16} FLOP | {RATE['tc_bf16'] / 1e12:.1f} TFLOP/s |")
    P(f"| FP8 tensor (dense) | {TC_FP8} FLOP | {RATE['tc_fp8'] / 1e12:.1f} TFLOP/s |")
    P(f"| FP32 FMA pipe | {FP32_LANES} lanes | {RATE['fp32'] / 1e12:.1f} T ops/s |")
    P(f"| INT32 / LOP3 / shift | {INT_LANES} lanes | {RATE['int'] / 1e12:.1f} T ops/s |")
    P(f"| SFU (ex2, rsqrt) | {SFU_LANES} lanes | {RATE['sfu'] / 1e12:.2f} T ops/s |")
    P(f"\nNVFP4 dequant (Marlin's instruction sequence): {DEQUANT_INT_PER_VALUE} INT ops per weight ->"
      f" {RATE['int'] / DEQUANT_INT_PER_VALUE / 1e12:.2f} T weights/s, against HBM delivering"
      f" {RATE['hbm'] / (EXPERT_BYTES / EXPERT_VALUES) / 1e12:.2f} T weights/s at 0.5625 B each.\n")

    # bf16_proj
    spec = load_spec("muse_bf16_proj")
    P("## muse_bf16_proj\n")
    P("Minimal traffic: W + x + out (x and out are reused from L2 across N tiles). Times at peak, µs.\n")
    P("| m | layer | wt | W bytes | x+out bytes | tensor GFLOP | HBM µs | tensor µs | fp32 µs | floor µs | bound | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    floors = task_floors("muse_bf16_proj")
    for case, jf in zip(spec["benchmarks"], floors):
        w = proj_shape(case["m"], case["layer"])
        t, bind = resource_cols(w)
        wb = sum(v for k, v in w.bytes.items() if k.endswith(".W"))
        P(f"| {case['m']} | {case['layer']} | {case['weight']} | {wb:,} | {w.hbm - wb:,} |"
          f" {(w.ops['tc_bf16']) / 1e9:.3f} | {fmt_us(t['hbm'])} | {fmt_us(t['tc_bf16'])} | {fmt_us(t['fp32'])} |"
          f" {fmt_us(w.floor())} | {bind} | {jf} |")
        STEP.append(("muse_bf16_proj", case["load"], case["weight"], w.floor(), jf))

    # attention
    spec = load_spec("muse_attention_decode")
    P("\n## muse_attention_decode\n")
    P("Cached keys are read once per step (K and V rows: 2 heads x 64 B each); new entries are"
      " written and attended from registers. Lengths: 80,000 fixed (B 1), else E[floor(ctx(1 ± 7.5%))].\n")
    P("| shape | wt | KV+page-table bytes | other bytes | (query,key) pairs | tensor GFLOP | exp2 (M) |"
      " HBM µs | tensor µs | SFU µs | floor µs | bound | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case, jf in zip(spec["benchmarks"], task_floors("muse_attention_decode")):
        w, pairs = attention_work(case["batch"], case["ctx"], case.get("q_len", 1), case["layer"],
                                  case.get("spread", "fixed"), case.get("page", 1))
        t, bind = resource_cols(w)
        kv = sum(v for k, v in w.bytes.items() if k.endswith(".read"))
        P(f"| B {case['batch']} L {case.get('q_len', 1)} {case['layer']} | {case['weight']} | {kv:,.0f} |"
          f" {w.hbm - kv:,.0f} | {pairs:,.0f} | {w.ops['tc_bf16'] / 1e9:.2f} | {w.ops['sfu'] / 1e6:.2f} |"
          f" {fmt_us(t['hbm'])} | {fmt_us(t['tc_bf16'])} | {fmt_us(t['sfu'])} | {fmt_us(w.floor())} | {bind} | {jf} |")
        STEP.append(("muse_attention_decode", case["load"], case["weight"], w.floor(), jf))

    # MoE
    spec = load_spec("muse_moe_w4a16")
    P("\n## muse_moe_w4a16\n")
    P(f"Per touched expert: {EXPERT_BYTES:,} B (w13 {EXPERT_W13:,} + w2 {EXPERT_W2:,} + 8 B global"
      f" scales) for {EXPERT_VALUES:,} weights. Routes and distinct experts: mean ± sd over seeds of the"
      " judge's generator (busiest GPU).\n")
    P("| bs | wt | routes | distinct | expert bytes | other bytes | tensor GFLOP | dequant INT (G) |"
      " HBM µs | tensor µs | INT µs | floor µs | bound | SGLang intermediates (MB) | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case, jf in zip(spec["benchmarks"], task_floors("muse_moe_w4a16")):
        bs = case["bs"]
        r, d, sr, sd = routing[bs]
        w = moe_work(bs, r, d)
        t, bind = resource_cols(w)
        extra = sum(moe_sglang_extra(bs, r).values())
        P(f"| {bs} | {case['weight']} | {r:.1f} ± {sr:.1f} | {d:.2f} ± {sd:.2f} | {w.bytes['experts.W']:,.0f} |"
          f" {w.hbm - w.bytes['experts.W']:,.0f} | {w.ops['tc_bf16'] / 1e9:.2f} | {w.ops['int'] / 1e9:.1f} |"
          f" {fmt_us(t['hbm'])} | {fmt_us(t['tc_bf16'])} | {fmt_us(t['int'])} | {fmt_us(w.floor())} | {bind} |"
          f" {extra / 1e6:.1f} | {jf} |")
        STEP.append(("muse_moe_w4a16", case["load"], case["weight"], w.floor(), jf))

    # decode layer
    spec = load_spec("muse_decode_layer")
    solved = backsolve_distinct("muse_decode_layer")
    P("\n## muse_decode_layer\n")
    P("Distinct local experts per shape are the judge's own (solved from its floors, V5); routes"
      " scale with them as in the MoE problem's routing. Phase-serial = sum over the 7 dependent"
      " phases of each phase's bound.\n")
    P("| shape | wt | distinct | HBM bytes | tensor GFLOP | HBM µs | all-resources floor µs | phase-serial floor µs |"
      " largest phases (µs) | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    layer_rows = []
    for case, jf, (_, distinct, _) in zip(spec["benchmarks"], task_floors("muse_decode_layer"), solved):
        t_ = case["batch"] * case.get("q_len", 1)
        routes = max(t_ * 1.0, t_ * TOPK * LOCAL / N_EXP * _route_skew(t_, routing))
        phases, _ = layer_phases(case["batch"], case["ctx"], case.get("q_len", 1), case["layer"],
                                 case.get("spread", "fixed"), routes, distinct, case.get("page", 1))
        total = Work()
        for _, p in phases:
            total += p
        serial = sum(p.floor() for _, p in phases)
        top = sorted(((p.floor(), name) for name, p in phases), reverse=True)[:3]
        layer_rows.append((case, total, serial))
        STEP.append(("muse_decode_layer", case["load"], case["weight"], serial, jf))
        P(f"| B {case['batch']} L {case.get('q_len', 1)} {case['layer']} | {case['weight']} | {distinct:.2f} |"
          f" {total.hbm:,.0f} | {(total.ops['tc_bf16'] + total.ops['tc_fp8']) / 1e9:.1f} |"
          f" {fmt_us(total.times()['hbm'])} | {fmt_us(total.floor())} | {fmt_us(serial)} |"
          f" {', '.join(f'{n} {s * 1e6:.1f}' for s, n in top)} | {jf} |")

    # prefill
    spec = load_spec("muse_prefill_layer")
    solved = backsolve_distinct("muse_prefill_layer")
    P("\n## muse_prefill_layer\n")
    P("Compute-bound: routes are the judge's (router_load, solved from its floors), distinct experts from simulation. Experts dequantize once per 128-token tile."
      " Attention shown at BF16; FP8 halves its tensor time but not its exp2s.\n")
    P("| shape | wt | routes | tensor TFLOP (dense/experts/attention) | exp2 (G) | tensor µs | SFU µs |"
      " INT µs | HBM µs | phase-serial floor µs | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for case, jf, (kind, val, _) in zip(spec["benchmarks"], task_floors("muse_prefill_layer"), solved):
        t_ = case["batch"] * case["q_len"]
        # router_load (the mean) where the floor reveals it; else the simulated busiest-GPU skew.
        routes = val if kind == "load" else t_ * _route_skew(t_, routing)
        distinct = min(32.0, routing.get(16384 if t_ > 3012 else 3012 if t_ > 753 else 753)[1])
        phases, pairs = layer_phases(case["batch"], case["ctx"], case["q_len"], case["layer"],
                                     case.get("spread", "fixed"), routes, distinct, case.get("page", 1))
        total = Work()
        for _, p in phases:
            total += p
        serial = sum(p.floor() for _, p in phases)
        tt = total.times()
        dense = sum(p.ops["tc_bf16"] for n, p in phases if n not in ("attention", "experts"))
        ex = dict(phases)["experts"].ops["tc_bf16"]
        at = dict(phases)["attention"].ops["tc_bf16"]
        P(f"| B {case['batch']} ctx {case['ctx']} q {case['q_len']} {case['layer']} | {case['weight']} |"
          f" {routes:,.0f} | {dense / 1e12:.2f} / {ex / 1e12:.2f} / {at / 1e12:.2f} | {total.ops['sfu'] / 1e9:.2f} |"
          f" {fmt_us(tt['tc_bf16'])} | {fmt_us(tt['sfu'])} | {fmt_us(tt['int'])} | {fmt_us(tt['hbm'])} |"
          f" {fmt_us(serial)} | {jf} |")
        STEP.append(("muse_prefill_layer", case["load"], case["weight"], serial, jf))
    lines.insert(0, "## Per step\n\nExact floors at H200 peak rates (every resource, dependent phases"
                    " serialized for the layers), against SGLang's measured step time.\n\n"
                    + step_summary() + "\n")
    return "\n".join(lines)


_ROUTE_RATIO = {}


def _route_skew(tokens, routing):
    """Busiest-GPU routes over the mean (tokens), from the MoE simulation at the nearest size."""
    keys = sorted(routing)
    k = min(keys, key=lambda b: abs(b - tokens))
    return routing[k][0] / k


# ---------------------------------------------------------------------------------------
# Validation


def validate(routing):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    out = []
    P = out.append
    ok_all = True

    # V6 hardware
    P("### V6 hardware peaks from per-SM rates")
    for name, got, spec in (("BF16 dense", RATE["tc_bf16"], 989.4e12), ("FP8 dense", RATE["tc_fp8"], 1978.9e12),
                            ("HBM", RATE["hbm"], 4.8e12)):
        ok = abs(got / spec - 1) < 1e-3
        ok_all &= ok
        P(f"- {name}: {got / 1e12:.1f} T vs datasheet {spec / 1e12:.1f} T: {'ok' if ok else 'MISMATCH'}")

    # V1 the judge's floors
    P("\n### V1 the judge's speed_of_light_ns reproduces TASK.md")
    for problem in ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer",
                    "muse_prefill_layer"):
        spec = load_spec(problem)
        floors = task_floors(problem)
        solved = backsolve_distinct(problem) if problem in ("muse_moe_w4a16", "muse_decode_layer",
                                                            "muse_prefill_layer") else None
        R = load_reference(problem)
        worst = 0.0
        for i, (case, jf) in enumerate(zip(spec["benchmarks"], floors)):
            kw = {k: v for k, v in case.items() if k not in ("load", "weight", "seed")}
            if solved:
                kind, val, _ = solved[i]
                kw["router_distinct" if kind == "distinct" else "router_load"] = val
            got = R.speed_of_light_ns(hw=JUDGE_HW, **kw) / 1e3
            worst = max(worst, abs(got - jf))
        ok = worst <= 0.0501  # the table prints 0.1 µs
        ok_all &= ok
        P(f"- {problem}: {len(floors)} shapes, worst |difference| {worst:.3f} µs (table precision 0.1 µs)"
          f"{' (router stats solved from the floors themselves)' if solved else ''}: {'ok' if ok else 'MISMATCH'}")

    # V2 our inventories under the judge's assumptions equal the judge's bytes/FLOPs
    P("\n### V2 our inventories under the judge's simplifications reproduce its floors")
    spec = load_spec("muse_bf16_proj")
    worst = 0.0
    for c, jf in zip(spec["benchmarks"], task_floors("muse_bf16_proj")):
        w = proj_shape(c["m"], c["layer"])
        flops = 2.0 * c["m"] * (w.bytes[f"{c['layer']}.W"] // (4 if c["layer"] == "router" else 2))
        jj = max(w.hbm / JUDGE_HW["hbm_bytes_per_s"], flops / JUDGE_HW["bf16_dense_flops_per_s"]) * 1e6
        worst = max(worst, abs(jj - jf))
    P(f"- bf16_proj: max(our bytes / 4.8 TB/s, our FLOPs / 989 T) vs the table: worst |difference|"
      f" {worst:.3f} µs")
    ok_all &= worst <= 0.0501
    spec = load_spec("muse_attention_decode")
    worst = 0.0
    for c, jf in zip(spec["benchmarks"], task_floors("muse_attention_decode")):
        # judge: shortest_share 0.925 on agentic spreads; K+V of ctx (or window) keys + qkv in and out x2.
        share = 1 - AGENTIC if c.get("spread", "fixed") == "agentic" else 1.0
        att = min(c["ctx"], WINDOW) if c["layer"] == "sliding" else c["ctx"]
        tok = c["batch"] * att * share
        L = c.get("q_len", 1)
        b = tok * HKV * HD * 2 + c["batch"] * L * QKV_N * 2 * 2
        f = 4.0 * tok * L * HQ * HD
        jj = max(b / 4.8e12, f / 989e12) * 1e6
        worst = max(worst, abs(jj - jf))
    P(f"- attention: our KV + activation bytes and FLOPs, at the judge's 0.925 length share, vs the table:"
      f" worst |difference| {worst:.3f} µs")
    ok_all &= worst <= 0.0501

    # V3 FLOPs via FlopCounterMode on meta tensors
    P("\n### V3 tensor FLOPs recounted by torch FlopCounterMode (meta tensors, exact shapes)")
    dev = "meta"

    def count(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops()

    bad = 0
    spec = load_spec("muse_bf16_proj")
    for c in spec["benchmarks"]:
        n, k = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
                "router": (256, 8192)}[c["layer"]]
        if c["layer"] == "unpooler" and c["m"] <= 8:
            n = 1024
        x = torch.empty(c["m"], k, device=dev, dtype=torch.bfloat16)
        wt = torch.empty(n, k, device=dev, dtype=torch.bfloat16)
        got = count(lambda: x @ wt.T) * (2 if c["layer"] == "router" and c["m"] > 8 else 1)
        mine = proj_shape(c["m"], c["layer"]).ops
        bad += got != mine["tc_bf16"] + 2 * mine["fp32"]
    P(f"- bf16_proj: {len(spec['benchmarks'])} shapes, {bad} mismatches")
    ok_all &= bad == 0

    bad = 0
    spec = load_spec("muse_attention_decode")
    for c in spec["benchmarks"]:
        L, B = c.get("q_len", 1), c["batch"]
        seq = 80000  # an integer length: the formula is linear in it
        sliding = c["layer"] == "sliding"

        def run():
            for j in range(L):
                nk = min(seq + j + 1, WINDOW) if sliding else seq + j + 1
                q = torch.empty(B, HQ, 1, HD, device=dev, dtype=torch.bfloat16)
                kk = torch.empty(B, HQ, nk, HD, device=dev, dtype=torch.bfloat16)
                s = q @ kk.transpose(-1, -2)
                _ = s @ kk
        got = count(run)
        w, _ = attention_work(B, seq, L, c["layer"], "fixed")
        bad += got != w.ops["tc_bf16"]
    P(f"- attention: {len(spec['benchmarks'])} shapes (per-query causal key counts), {bad} mismatches")
    ok_all &= bad == 0

    bad = 0
    for routes in (1, 7, 398):
        def run():
            x = torch.empty(routes, D_EXP, device=dev, dtype=torch.bfloat16)
            w13 = torch.empty(2 * D_EXP, D_EXP, device=dev, dtype=torch.bfloat16)
            w2 = torch.empty(D_EXP, D_EXP, device=dev, dtype=torch.bfloat16)
            h = x @ w13.T
            _ = h[:, :D_EXP] @ w2.T
        bad += count(run) != moe_work(routes, routes, 1).ops["tc_bf16"]
    P(f"- experts (GEMM1 + GEMM2 per route): 3 route counts, {bad} mismatches")
    ok_all &= bad == 0

    # V4 bytes from meta tensors
    P("\n### V4 byte inventories recounted from meta tensors (exact shapes and dtypes)")

    def nbytes(*shape, dtype):
        return torch.empty(*shape, device=dev, dtype=dtype).untyped_storage().nbytes()

    fp4 = torch.uint8
    e = (nbytes(2 * D_EXP, D_EXP // 2, dtype=fp4) + nbytes(2 * D_EXP, D_EXP // 16, dtype=torch.float8_e4m3fn)
         + nbytes(D_EXP, D_EXP // 2, dtype=fp4) + nbytes(D_EXP, D_EXP // 16, dtype=torch.float8_e4m3fn)
         + 2 * nbytes(1, dtype=torch.float32))
    P(f"- one expert: {e:,} B from tensors vs {EXPERT_BYTES:,} B closed form: {'ok' if e == EXPERT_BYTES else 'MISMATCH'}")
    ok_all &= e == EXPERT_BYTES
    kv = nbytes(1, 1, HKV, HD, dtype=torch.float8_e4m3fn)
    P(f"- one cached key row: {kv} B per cache (K and V separate, 2 heads x 64): {'ok' if kv == 128 else 'MISMATCH'}")
    dense = sum(nbytes(n, k, dtype=torch.bfloat16) for n, k in ((2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096)))
    mine = sum(proj_shape(32, l).bytes[f"{l}.W"] for l in ("qkv_gate", "o", "pooler", "unpooler"))
    P(f"- dense BF16 weights (m > 8): {dense:,} B vs {mine:,} B: {'ok' if dense == mine else 'MISMATCH'}")
    ok_all &= dense == mine

    # V5 routing
    P("\n### V5 busiest-GPU distinct experts: simulation vs the judge's own (solved from its floors)")
    solved = backsolve_distinct("muse_moe_w4a16")
    spec = load_spec("muse_moe_w4a16")
    P("| bs | judge distinct | simulated distinct (mean ± sd) | difference |")
    P("|---|---|---|---|")
    for c, (kind, val, _) in zip(spec["benchmarks"], solved):
        r, d, sr, sd = routing[c["bs"]]
        P(f"| {c['bs']} | {val:.2f} | {d:.2f} ± {sd:.2f} | {val - d:+.2f} |")
    P(f"\n**all checks {'passed' if ok_all else 'FAILED'}**")
    return "\n".join(out), ok_all


if __name__ == "__main__":
    routing = simulate_routing([1, 8, 16, 32, 128, 256, 753, 3012, 16384])
    text = report(routing)
    if "--validate" in sys.argv:
        v, ok = validate(routing)
        text += "\n\n## Validation\n\n" + v
    print(text)
