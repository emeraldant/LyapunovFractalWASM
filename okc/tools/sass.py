"""Exact instruction accounting for our CUDA kernels on sm_90a, without a GPU.

CUDA source -> PTX (NVRTC, pip nvidia-cuda-nvrtc-cu12) -> cubin (ptxas, bundled with Triton)
-> SASS (nvdisasm). Then:
  - ptxas resource usage: registers, spills, static shared memory;
  - the main loop (the body of the innermost backward branch with the most HMMAs), its
    instructions classified by the pipe that executes them;
  - a per-SM-sub-partition issue model with Hopper's per-clock pipe rates.

  python3 okc/tools/sass.py moe        # the MoE GEMM instantiations
"""
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

NV = "/usr/local/lib/python3.13/dist-packages/nvidia"
TRITON_BIN = glob.glob("/usr/local/lib/python3*/dist-packages/triton/backends/nvidia/bin")[0]
PTXAS = os.path.join(TRITON_BIN, "ptxas")
NVDISASM = os.path.join(TRITON_BIN, "nvdisasm")
CUOBJDUMP = os.path.join(TRITON_BIN, "cuobjdump")

# Pipe of each SASS opcode family on sm_90 (per SM sub-partition, warp instructions per clock):
# issue 1/clk; FMA-heavy (FFMA/FMUL/FADD, HFMA2/HMUL2) 1/clk; INT/ALU (LOP3, SHF, IADD3, IMAD,
# PRMT, ISETP, SEL, LEA, ...) 1/2 clk (16 lanes); MUFU 1/8 clk; HMMA m16n8k16 BF16: 1/8 clk
# (256 dense FMA per clock per sub-partition); memory instructions go to the LSU/MIO queue.
PIPES = {
    # FP32/FP16x2 arithmetic: the FMA pipe (fmalite + fmaheavy), 32 lanes per sub-partition.
    "fma": ("FFMA", "FMUL", "FADD", "HFMA2", "HMUL2", "HADD2", "FCHK"),
    # Integer multiply-add family (incl. IMAD.SHL / IMAD.MOV / IMAD.X): fmaheavy only, 16 lanes.
    "imad": ("IMAD", "IDP", "IMUL"),
    # Integer/logic ALU: 16 lanes per sub-partition.
    "alu": ("LOP3", "SHF", "IADD3", "PRMT", "ISETP", "SEL", "LEA", "IABS", "IMNMX", "LOP", "SHL", "SHR", "BMSK",
            "FLO", "POPC", "BREV", "IADD", "VIADD", "VIMNMX", "I2IP", "MOV", "P2R", "R2P", "PLOP3", "FMNMX", "FSETP",
            "FSEL", "S2R", "CS2R", "VOTE"),
    # Uniform datapath (per warp, scalar): its own pipe, not lane throughput.
    "uniform": ("UMOV", "UIADD3", "ULOP3", "USHF", "UIMAD", "ULEA", "UISETP", "USEL", "UPRMT", "ULDC", "S2UR",
                "UPLOP3", "VOTEU", "R2UR"),
    "tensor": ("HMMA", "HGMMA", "IMMA", "QGMMA"),
    "sfu": ("MUFU",),
    "conv": ("F2F", "I2F", "F2I", "F2FP", "I2FP", "F2IP", "FRND"),
    "mem": ("LDG", "STG", "LDS", "STS", "LDSM", "LDGSTS", "LDGDEPBAR", "LD", "ST", "ATOM", "RED", "ATOMS", "SHFL",
            "LDC", "ULDC", "MEMBAR", "STSM", "UBLKCP", "UTMALDG", "SYNCS"),
    "ctrl": ("BRA", "BAR", "EXIT", "DEPBAR", "WARPSYNC", "BSSY", "BSYNC", "NOP", "YIELD", "RET", "CALL", "ERRBAR",
             "CCTL", "ACQBULK", "WARPGROUP"),
}
# Warp instructions per clock per sub-partition for the throughput-limited pipes.
RATE = {"fma": 1.0, "imad": 0.5, "alu": 0.5, "tensor": 1 / 8, "sfu": 1 / 8, "conv": 1 / 8, "mem": 0.5,
        "uniform": 1.0}


def pipe_of(op):
    base = op.split(".")[0]
    for pipe, ops in PIPES.items():
        if base in ops:
            return pipe
    return "other"


# NVRTC has no C library headers.
NVRTC_PRELUDE = ("typedef unsigned int uint32_t; typedef int int32_t; typedef unsigned char uint8_t;\n"
                 "typedef unsigned short uint16_t; typedef unsigned long long uint64_t; typedef long long int64_t;\n"
                 "typedef unsigned long size_t;\n")


def nvrtc_ptx(src, name_exprs, arch="compute_90a", extra=()):
    lib = ctypes.CDLL(os.path.join(NV, "cuda_nvrtc/lib/libnvrtc.so.12"))
    prog = ctypes.c_void_p()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None) == 0
    for n in name_exprs:
        assert lib.nvrtcAddNameExpression(prog, n.encode()) == 0
    opts = [f"--gpu-architecture={arch}", "-std=c++17", f"-I{NV}/cuda_runtime/include", "--use_fast_math",
            "-default-device", *extra]
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise RuntimeError(log.value.decode()[-4000:])
    lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetPTX(prog, ptx)
    return ptx.value.decode()


def compile_sass(ptx, arch="sm_90a"):
    with tempfile.TemporaryDirectory() as d:
        pf, cf = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
        open(pf, "w").write(ptx)
        res = subprocess.run([PTXAS, f"-arch={arch}", "-v", "-O3", pf, "-o", cf], capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr[-3000:])
        sass = subprocess.run([CUOBJDUMP, "-sass", cf], capture_output=True, text=True).stdout
        return res.stderr, sass


def split_functions(sass):
    funcs, cur, name = {}, [], None
    for line in sass.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            if name:
                funcs[name] = cur
            name, cur = m.group(1), []
        elif name:
            cur.append(line)
    if name:
        funcs[name] = cur
    return funcs


def parse(lines):
    """[(address, opcode)] of the instructions in a function's SASS."""
    out = []
    for line in lines:
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)(.*);", line)
        if m:
            out.append((int(m.group(1), 16), m.group(3), m.group(4)))
    return out


def loops(instrs):
    """Backward branches: (start address, end address)."""
    res = []
    for addr, op, args in instrs:
        if op.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", args)
            if m and int(m.group(1), 16) <= addr:
                res.append((int(m.group(1), 16), addr))
    return res


def main_loop(instrs):
    best = None
    for lo, hi in loops(instrs):
        body = [i for i in instrs if lo <= i[0] <= hi]
        hmma = sum(op.startswith("HMMA") for _, op, _ in body)
        if hmma and (best is None or (hmma, -(hi - lo)) > best[0]):
            best = ((hmma, -(hi - lo)), body)
    return best[1] if best else []


def report(body, label, work_note=""):
    c = Counter(pipe_of(op) for _, op, _ in body)
    ops = Counter(op.split(".")[0] for _, op, _ in body)
    issue = sum(c.values())
    cycles = {p: n / RATE[p] for p, n in c.items() if p in RATE}
    # IMAD and FP32/FP16x2 ops share the FMA pipe's issue: one warp instruction per clock in all.
    cycles["fma"] = c.get("fma", 0) + c.get("imad", 0)
    bound = max(cycles.items(), key=lambda kv: kv[1]) if cycles else ("-", 0)
    print(f"  main loop: {issue} warp instructions; per pipe: " + ", ".join(f"{p} {c[p]}" for p in sorted(c)))
    print(f"  top opcodes: " + ", ".join(f"{o} {n}" for o, n in ops.most_common(14)))
    print(f"  per sub-partition clocks for one warp's iteration: issue {issue}, "
          + ", ".join(f"{p} {v:.0f}" for p, v in sorted(cycles.items())) + f" -> bound by {bound[0]} {work_note}")
    return c, issue, cycles


def moe_ws(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_cuda"))
    import kernel as K
    insts = [(1, 1, 2, 4), (2, 1, 1, 4), (1, 2, 2, 4), (2, 2, 1, 4)]
    names = [f"moe_gemm_ws_kernel<{m}, {t}, {g}, {s}>" for m, t, g, s in insts]
    src = NVRTC_PRELUDE + K.MOE_DEVICE_SRC.replace("#include <stdint.h>", "") + "\n" + "\n".join(
        f"template __global__ void moe_gemm_ws_kernel<{m}, {t}, {g}, {s}>("
        "const __nv_bfloat16*, const int32_t*, const uint8_t*, const uint8_t*, "
        "const float*, const int32_t*, const int32_t*, const int32_t*, "
        "__nv_bfloat16*, int);" for m, t, g, s in insts)
    log, sass = compile_sass(nvrtc_ptx(src, names))
    for line in log.splitlines():
        if "Used" in line or "spill" in line:
            print(line.strip())
    for fname, lines in split_functions(sass).items():
        if "ws_kernel" not in fname:
            continue
        instrs = parse(lines)
        body = main_loop(instrs)
        m = re.search(r"ILi(\d)ELi(\d)ELi(\d)ELi(\d)E", fname)
        ngw = int(m.group(3))
        print(f"\n== WS MODE {m.group(1)} MT {m.group(2)} NGW {ngw} STAGES {m.group(4)}: {len(instrs)} instructions")
        report(body, "", f"(per {int(m.group(4))} chunks x {ngw * 1024} weights per warp)")


def moe(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_cuda"))
    import kernel as K
    insts = [(1, 1, 2, 4), (1, 4, 2, 4), (2, 1, 1, 4), (2, 4, 1, 4)]
    names = [f"moe_gemm_kernel<{m}, {t}, {g}, {s}>" for m, t, g, s in insts]
    src = NVRTC_PRELUDE + K.MOE_DEVICE_SRC.replace("#include <stdint.h>", "") + "\n" + "\n".join(f"template __global__ void moe_gemm_kernel<{m}, {t}, {g}, {s}>("
                                              "const __nv_bfloat16*, const int32_t*, const uint32_t*, const uint8_t*, "
                                              "const float*, const int32_t*, const int32_t*, const int32_t*, "
                                              "__nv_bfloat16*, int);" for m, t, g, s in insts)
    ptx = nvrtc_ptx(src, names)
    log, sass = compile_sass(ptx)
    funcs = split_functions(sass)
    usage = re.findall(r"Compiling entry function '(\S+)'.*?\n.*?Used (\d+) registers.*?(\d+) bytes smem|"
                       r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log, re.S)
    print(log.strip()[-2500:])
    for fname, lines in funcs.items():
        instrs = parse(lines)
        body = main_loop(instrs)
        m = re.search(r"ILi(\d)ELi(\d)ELi(\d)ELi(\d)E", fname)
        label = f"MODE {m.group(1)} MT {m.group(2)} NGW {m.group(3)} STAGES {m.group(4)}" if m else fname
        print(f"\n== {label}: {len(instrs)} instructions total")
        # One iteration = one 128-k chunk for this warp: NGW groups x 8 rows x 128 k = NGW x 1024 weights.
        ngw = int(m.group(3)) if m else 1
        report(body, label, f"(per {ngw * 1024} weights per warp)")


if __name__ == "__main__":
    {"moe": moe, "moe_ws": moe_ws}[sys.argv[1]](sys.argv[2:])
