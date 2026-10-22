"""First-principles rooflines for the Open Kernel Challenge problems on one H200 SXM.

For every scored shape: the minimum bytes and FLOPs, the judge's floor (peak HBM or peak
BF16, whichever is longer), and a practical floor: realistic bandwidth / tensor throughput /
dequant ALU, whichever binds, plus a fixed cost per dependent kernel stage. Every constant
below is an assumption to calibrate against measured runs (okc check prints per-shape times).

Run: python3 okc/tools/roofline.py  (prints the tables in ROOFLINES.md)
"""
import math

# H200 SXM. Peaks from the spec sheet (the judge uses the first two).
BW_PEAK = 4.8e12          # B/s HBM3e
BF16_PEAK = 989e12        # dense BF16 tensor FLOP/s
FP8_PEAK = 1979e12        # dense FP8 tensor FLOP/s
# Assumed achievable rates (to calibrate).
BW_STREAM = 4.3e12        # long contiguous reads, cold L2 (~90% of peak)
BW_GATHER = 3.6e12        # one random 128 B row per key per cache (page size 1, ~75%)
BF16_EFF = 700e12         # wgmma BF16 at realistic clocks/occupancy (~70%)
FP8_EFF = 1300e12
ALU = 132 * 64 * 1.83e9   # INT32/LOP3/shift lane-ops/s (15.5 T): dequant is integer work
DEQUANT_OPS = 1.875       # INT ops per NVFP4 weight in Marlin's sequence (tools/exact.py)
STAGE_US = 1.0            # per dependent kernel stage beyond the first: launch gap + ramp + tail
                          # (the judge subtracts ~4 us of graph replay cost per shape)

H, D_EXP, N_EXP, TOPK, LOCAL = 8192, 4096, 256, 8, 32
QKV_N, O_K, HQ, HKV, HD = 2304, 1024, 16, 2, 64
EXPERT_ELEMS = 3 * D_EXP * D_EXP
NVFP4_B = 0.5 + 1 / 16    # bytes per weight element with its E4M3 block scale
NVFP4_B_PACKED = 0.5 + 3.71 / 8 / 16  # scales entropy-coded (3.7 bits measured on the judge's)
BF16_PACKED = 12 / 16     # 4-bit exponent code: 99.986% of the judge's BF16 weights, rest escaped

# Simulated busiest-GPU routing for the judge's MoE generator (tools/... see ROOFLINES.md):
# tokens -> (local routes, distinct local experts)
ROUTES = {1: (2.5, 2.5), 8: (14.1, 8.6), 16: (25.5, 11.8), 32: (50.8, 16.2), 128: (195.1, 22.0),
          256: (398.2, 24.3), 753: (1085.8, 27.4), 3012: (4707.8, 27.3), 16384: (23236.7, 28.2)}


def us(seconds):
    return seconds * 1e6


def row(name, weight, bytes_, flops, judge, practical, packed=None, bound="", note=""):
    return dict(name=name, weight=weight, mb=bytes_ / 1e6, gflop=flops / 1e9, ai=flops / max(bytes_, 1),
                naive=us(max(bytes_ / BW_PEAK, flops / BF16_PEAK)), judge=judge, practical=practical,
                packed=packed if packed is not None else practical, bound=bound, note=note)


# ---------------------------------------------------------------------------------------
def bf16_proj():
    layers = {"qkv_gate": (2304, 8192, 2), "o": (8192, 1024, 2), "pooler": (4096, 8192, 2),
              "unpooler": (8192, 4096, 2), "router": (256, 8192, 4)}
    judge = {}
    rows = []
    for m in (1, 16, 32, 8, 128, 256):
        for name, (n, k, wb) in layers.items():
            if name == "unpooler" and m <= 8:
                n = 1024
            w = n * k * wb
            b = w + m * k * 2 + m * n * (4 if name == "router" else 2)
            f = 2 * m * n * k
            t_mem, t_cmp = b / BW_STREAM, f / BF16_EFF
            bound = "mem" if t_mem > t_cmp else "tensor"
            pr = us(max(t_mem, t_cmp))
            packed_b = b - w * (1 - (28 / 32 if wb == 4 else BF16_PACKED))
            pk = us(max(packed_b / BW_STREAM, t_cmp))
            rows.append(row(f"m {m:<3d} {name}", 62, b, f, None, pr, pk, bound))
    return rows


def attention():
    rows = []
    for batch, length in ((1, 1), (16, 1), (32, 1), (1, 8), (16, 8), (32, 8)):
        for layer in ("sliding", "full"):
            ctx = 80000
            keys = (min(ctx, 2047) + length) if layer == "sliding" else ctx + length
            kv = batch * keys * HKV * HD * 2               # K and V, FP8
            pt = batch * keys * 4                          # page table, page size 1
            act = batch * length * ((2 * HQ + 2 * HKV) * HD * 2 + HQ * HD * 2)
            b = kv + pt + act
            f = 4 * batch * keys * length * HQ * HD
            t_mem = (kv + pt) / BW_GATHER + act / BW_STREAM
            t_cmp = f / BF16_EFF
            # A CTA's critical path is >= 3 dependent DRAM round trips (qkv, page table, KV)
            # plus the combine stage: ~3 us however little data there is.
            lat = 3.0e-6
            pr = us(max(t_mem, t_cmp, lat)) + STAGE_US
            bound = "latency" if lat > max(t_mem, t_cmp) else ("gather" if t_mem > t_cmp else "tensor")
            rows.append(row(f"B {batch:<2d} L {length} {layer}", 46 if layer == "sliding" else 16, b, f,
                            None, pr, bound=bound))
    return rows


def moe_cost(tokens, packed=False, stages=4):
    routes, distinct = ROUTES[tokens]
    bpe = NVFP4_B_PACKED if packed else NVFP4_B
    w = distinct * EXPERT_ELEMS * bpe
    act = tokens * D_EXP * 2 * 2 + routes * (2 * D_EXP + D_EXP + D_EXP) * 2 * 2  # x, out, intermediates
    b = w + act
    f = 2 * routes * EXPERT_ELEMS
    t_mem, t_cmp = b / BW_STREAM, f / BF16_EFF
    t_alu = distinct * EXPERT_ELEMS * DEQUANT_OPS / ALU
    t = max(t_mem, t_cmp, t_alu)
    bound = ["mem", "tensor", "dequant"][[t_mem, t_cmp, t_alu].index(t)]
    return b, f, us(t) + (stages - 1) * STAGE_US, bound


def moe():
    rows = []
    for tokens in (1, 16, 32, 8, 128, 256):
        b, f, pr, bound = moe_cost(tokens)
        _, _, pk, _ = moe_cost(tokens, packed=True)
        r = ROUTES[tokens]
        rows.append(row(f"bs {tokens:<3d}", 60, b, f, None, pr, pk, bound, f"{r[0]:.0f} routes, {r[1]:.1f} experts"))
    return rows


def layer_cost(batch, length, layer, ctx=80000, packed=False, stages=10):
    tokens = batch * length
    unpool_n = 1024 if tokens <= 8 else H
    dense = (QKV_N * H + H * O_K + D_EXP * H + unpool_n * D_EXP) * 2
    dense_b = dense * (BF16_PACKED if packed else 1) + N_EXP * H * 4 * (28 / 32 if packed else 1)
    keys = (min(ctx, 2047) + length) if layer == "sliding" else ctx + length
    kv = batch * keys * HKV * HD * 2 + batch * keys * 4
    act = tokens * H * (4 + 2) * 2
    mb, mf, _, _ = moe_cost(tokens, packed) if tokens in ROUTES else (0, 0, 0, 0)
    b = dense_b + kv + act + mb
    f = 2 * tokens * (dense / 2 + N_EXP * H) + mf + 4 * batch * keys * length * HQ * HD
    t = (dense_b + act) / BW_STREAM + kv / BW_GATHER + max(mb / BW_STREAM, mf / BF16_EFF)
    t = max(t, f / BF16_EFF)
    return b, f, us(t) + (stages - 1) * STAGE_US


def decode_layer():
    rows = []
    for batch, length in ((1, 1), (16, 1), (32, 1), (1, 8), (16, 8), (32, 8)):
        for layer in ("sliding", "full"):
            b, f, pr = layer_cost(batch, length, layer)
            _, _, pk = layer_cost(batch, length, layer, packed=True, stages=7)
            rows.append(row(f"B {batch:<2d} L {length} {layer}", 46 if layer == "sliding" else 16, b, f, None,
                            pr, pk, "mem"))
    return rows


def prefill():
    rows = []
    shapes = [(1, 80000, 753, "sliding", 46), (1, 80000, 753, "full", 16), (4, 80000, 753, "sliding", 46),
              (4, 80000, 753, "full", 16), (1, 0, 16384, "full", 16), (1, 32768, 16384, "sliding", 138),
              (1, 32768, 16384, "full", 16), (1, 65536, 16384, "full", 16)]
    for batch, ctx, length, layer, weight in shapes:
        tokens = batch * length
        dense = (QKV_N * H + H * O_K + D_EXP * H + H * D_EXP)
        f_dense = 2 * tokens * (dense + N_EXP * H)
        r = ROUTES[min(ROUTES, key=lambda t: abs(t - tokens))]
        routes = r[0] * tokens / min(ROUTES, key=lambda t: abs(t - tokens))
        f_moe = 2 * routes * EXPERT_ELEMS
        if layer == "sliding":
            # each query attends min(window, its position + 1) keys
            pairs = sum(min(2048, ctx + i + 1) for i in range(0, length, 16)) * 16
        else:
            pairs = length * ctx + length * (length + 1) / 2
        f_att = 4 * batch * pairs * HQ * HD
        kv = batch * ((min(ctx, 2047) if layer == "sliding" else ctx) + length) * HKV * HD * 2
        b = dense * 2 + 28 * EXPERT_ELEMS * NVFP4_B + kv + tokens * H * 12
        f = f_dense + f_moe + f_att
        t = f_dense / BF16_EFF + max(f_moe / BF16_EFF, 28 * EXPERT_ELEMS * NVFP4_B / BW_STREAM) + f_att / FP8_EFF
        t = max(t, b / BW_STREAM)
        rows.append(row(f"B {batch} ctx {ctx} q {length} {layer}", weight, b, f, None, us(t) + 9 * STAGE_US,
                        bound=f"att {f_att / f:.0%} moe {f_moe / f:.0%} dense {f_dense / f:.0%}"))
    return rows


def print_table(title, rows, loads=None):
    print(f"\n### {title}\n")
    print("| shape | wt | MB | GFLOP | FLOP/B | judge floor µs | practical µs | packed µs | bound | note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['weight']} | {r['mb']:.1f} | {r['gflop']:.2f} | {r['ai']:.0f} | {r['naive']:.1f} |"
              f" {r['practical']:.1f} | {r['packed']:.1f} | {r['bound']} | {r['note']} |")


SGLANG = {"muse_bf16_proj": (4830.8, 1.203), "muse_attention_decode": (3766.3, 1.263),
          "muse_moe_w4a16": (13448.5, 1.309), "muse_decode_layer": (22620.3, 1.242),
          "muse_prefill_layer": (387403.9, 1.097)}  # SGLang µs per step (geomean over loads), leader on Oct 22
LOAD_ROWS = {"muse_bf16_proj": [5] * 6, "muse_attention_decode": [2] * 6, "muse_moe_w4a16": [1] * 6,
             "muse_decode_layer": [2] * 6, "muse_prefill_layer": [2, 2, 4]}


def per_step(rows, sizes, col):
    """The judge's step time: each load's sum of weight x time, geometric mean over loads."""
    loads, at = [], 0
    for n in sizes:
        loads.append(sum(r["weight"] * r[col] for r in rows[at:at + n]))
        at += n
    return math.exp(sum(math.log(v) for v in loads) / len(loads)), loads


if __name__ == "__main__":
    summary = []
    for title, fn in (("muse_bf16_proj", bf16_proj), ("muse_attention_decode", attention), ("muse_moe_w4a16", moe),
                      ("muse_decode_layer", decode_layer), ("muse_prefill_layer", prefill)):
        rows = fn()
        print_table(title, rows)
        sg, lead = SGLANG[title]
        steps = {col: per_step(rows, LOAD_ROWS[title], col)[0] for col in ("naive", "practical", "packed")}
        summary.append((title, sg, lead, steps))
    print("\n### Headroom per step (geometric mean over loads)\n")
    print("| problem | SGLang µs | leader | judge floor µs (max) | practical µs (max) | packed µs (max) |")
    print("|---|---|---|---|---|---|")
    for title, sg, lead, st in summary:
        print(f"| {title} | {sg:.0f} | {lead:.2f}x | {st['naive']:.0f} ({sg / st['naive']:.2f}x) |"
              f" {st['practical']:.0f} ({sg / st['practical']:.2f}x) | {st['packed']:.0f} ({sg / st['packed']:.2f}x) |")
