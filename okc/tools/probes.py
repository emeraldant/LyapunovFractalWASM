"""Machine probes run inside prepare() on the judge's H200 (untimed), printed to the log.

Measured the way the judge times shapes: CUDA graph replays, each after a cold L2.
  - stream: contiguous reads of S bytes by G programs -> achieved bandwidth and fixed cost
  - gather: N random 128 B rows out of a large buffer (the paged KV cache's pattern)
  - launch: a graph of K tiny kernels -> cost per extra kernel

Bundled into a submission with tools/bundle.py; call run_probes(budget_s) from prepare().
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _probe_stream_kernel(src_ptr, out_ptr, n_words, per_prog, BLOCK: tl.constexpr):
    """Each program sums its contiguous slice of int32 words (so the loads aren't dead)."""
    pid = tl.program_id(0)
    start = pid.to(tl.int64) * per_prog
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.int32)
    for i in range(0, per_prog, BLOCK):
        idx = start + i + offs
        acc ^= tl.load(src_ptr + idx, mask=idx < n_words, other=0, eviction_policy="evict_first")
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _probe_gather_kernel(src_ptr, rows_ptr, out_ptr, n_rows, ROWS: tl.constexpr):
    """Each program reads ROWS random 128 B rows (32 int32 words each)."""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    rmask = r < n_rows
    row = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int64)
    w = tl.arange(0, 32)
    v = tl.load(src_ptr + row[:, None] * 32 + w[None, :], mask=rmask[:, None], other=0)
    tl.store(out_ptr + pid, tl.sum(tl.sum(v, axis=1), axis=0))


@triton.jit
def _probe_tiny_kernel(out_ptr):
    tl.store(out_ptr + tl.program_id(0), 1)


class _ProbeBench:
    def __init__(self, device):
        self.flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device)

    def time(self, fn, reps=21):
        with torch.cuda.stream(self.stream):
            fn()
            fn()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        with torch.cuda.stream(self.stream):
            for i in range(reps):
                self.flush.zero_()
                starts[i].record()
                graph.replay()
                ends[i].record()
        ends[-1].synchronize()
        times = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))  # µs
        del graph
        return times[len(times) // 2]


def run_probes(budget_s=60.0):
    t0 = time.time()
    device = torch.device("cuda")
    bench = _ProbeBench(device)
    out = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    print(f"probe gpu {torch.cuda.get_device_name()} sms {torch.cuda.get_device_properties(0).multi_processor_count}")

    # Launch cost: graphs of K tiny kernels.
    base = None
    for k in (1, 2, 4, 8, 16):
        def fn(k=k):
            for _ in range(k):
                _probe_tiny_kernel[(1,)](out)
        t = bench.time(fn)
        base = base if base is not None else t
        print(f"probe launch k={k:<2d} {t:7.2f} us  (+{(t - base) / max(k - 1, 1):.2f} us per extra kernel)")

    # Streaming reads.
    src = torch.randint(0, 1 << 30, (800 * 1024 * 1024 // 4,), dtype=torch.int32, device=device)
    for size_mb in (1, 4, 8, 17, 38, 67, 75, 150, 340, 690):
        n_words = size_mb * 1024 * 1024 // 4
        best = None
        for progs in (132, 264, 528, 1056, 2112):
            for block, warps in ((1024, 4), (2048, 8), (4096, 8)):
                if time.time() - t0 > budget_s:
                    break
                per_prog = triton.cdiv(triton.cdiv(n_words, progs), block) * block
                grid = triton.cdiv(n_words, per_prog)

                def fn(n_words=n_words, per_prog=per_prog, grid=grid, block=block, warps=warps):
                    _probe_stream_kernel[(grid,)](src, out, n_words, per_prog, BLOCK=block, num_warps=warps)
                try:
                    t = bench.time(fn)
                except Exception as e:
                    print(f"probe stream error {type(e).__name__}: {str(e)[:100]}")
                    continue
                if best is None or t < best[0]:
                    best = (t, grid, block, warps)
        if best:
            t, grid, block, warps = best
            print(f"probe stream {size_mb:4d} MB: {t:8.2f} us raw -> {n_words * 4 / ((t - base) * 1e-6) / 1e12:5.2f} TB/s"
                  f" net of one launch (best grid {grid}, block {block}, warps {warps})")

    # Random 128 B rows (both caches of a page-size-1 KV step read this way).
    n_src_rows = src.numel() // 32
    gen = torch.Generator(device=device).manual_seed(0)
    for n_rows in (2048, 32768, 160000, 1280000, 2560000, 5120000):
        rows = torch.randint(0, n_src_rows, (n_rows,), generator=gen, device=device, dtype=torch.int32)
        best = None
        for per in (16, 32, 64):
            if time.time() - t0 > budget_s:
                break

            def fn(rows=rows, per=per):
                _probe_gather_kernel[(triton.cdiv(rows.numel(), per),)](src, rows, out, rows.numel(), ROWS=per,
                                                                        num_warps=4)
            try:
                t = bench.time(fn)
            except Exception as e:
                print(f"probe gather error {type(e).__name__}: {str(e)[:100]}")
                continue
            if best is None or t < best[0]:
                best = (t, per)
        if best:
            t, per = best
            mb = n_rows * 128 / 1e6
            print(f"probe gather {n_rows:8d} rows ({mb:7.1f} MB): {t:8.2f} us raw ->"
                  f" {n_rows * 128 / ((t - base) * 1e-6) / 1e12:5.2f} TB/s net (rows/program {per})")
    del src, bench
    torch.cuda.empty_cache()
    print(f"probes: {time.time() - t0:.0f} s")
