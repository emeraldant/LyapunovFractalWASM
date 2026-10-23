"""CPU check of moe_cuda/kernel.py's Triton parts (route compaction, row-indexed post-norm)."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import interp_patch  # noqa: F401
import torch
sys.path.insert(0, os.path.join(here, "..", "moe_cuda"))
import kernel as K

fails = 0
gen = torch.Generator().manual_seed(0)
import triton
for bs, ls, rows_per, split in ((1, 0, 16, 0), (5, 32, 16, 0), (33, 224, 32, 0), (257, 96, 16, 0), (257, 96, 64, 0),
                                (256, 96, 16, 16), (128, 0, 16, 16), (3, 7, 8, 0)):
    ids = torch.randint(0, 256, (bs, 8), generator=gen, dtype=torch.int32)
    if bs >= 128:   # a few hot experts, as the routing's heavy tail makes them
        ids[: bs // 2, 0] = (ls + 3) % 256
        ids[: bs // 4, 1] = (ls + 9) % 256
    p = bs * 8
    row_route = torch.full((p,), -7, dtype=torch.int32); pos = torch.full((p,), -7, dtype=torch.int32)
    exp_start = torch.zeros(33, dtype=torch.int32); active = torch.full((2 * 32 + 2 * p + 1,), -7, dtype=torch.int32)
    n_active = torch.zeros(2, dtype=torch.int32)
    K._route_compact_kernel[(32,)](ids, torch.tensor([ls], dtype=torch.int32), row_route, pos, exp_start, active,
                                   n_active, p, rows_per, NE=256, E=32, CHUNK=max(256, triton.next_power_of_2(p)),
                                   SPLIT=split, ROWS2=64)
    loc = ((ids.flatten().long() - ls) % 256)
    counts = torch.bincount(loc[loc < 32], minlength=32)
    ok = torch.equal(exp_start[:32].long(), torch.cumsum(counts, 0) - counts) and int(exp_start[32]) == int(counts.sum())
    small = [e for e in range(32) if not (split and counts[e] > split)]
    act = torch.tensor([e | (j << 8) for e in small for j in range(-(-int(counts[e]) // rows_per))], dtype=torch.long)
    ok &= int(n_active[0]) == len(act) and torch.equal(active[:len(act)].long(), act)
    if split:
        act2 = torch.tensor([e | (j << 8) for e in range(32) if counts[e] > split for j in range(-(-int(counts[e]) // 64))],
                            dtype=torch.long)
        ok &= int(n_active[1]) == len(act2) and torch.equal(active[32 + p:32 + p + len(act2)].long(), act2)
    for e in range(32):
        rows = row_route[exp_start[e]:exp_start[e + 1]].long()
        ok &= torch.equal(rows, (loc == e).nonzero().flatten())                  # route order within an expert
        ok &= torch.equal(pos[rows].long(), torch.arange(int(exp_start[e]), int(exp_start[e + 1])))
    ok &= bool((pos[loc >= 32] == -1).all())
    fails += not ok
    print(f"{'OK ' if ok else 'BAD'} route bs{bs} local_start {ls}: rows/chunk {rows_per}, split {split}: {n_active.tolist()} chunks,"
          f" {int(exp_start[32])} rows")
# Post-norm through the row map.
bs = 5
y = torch.randn(40, 4096).to(torch.bfloat16); s = torch.rand(bs, 8); w = (1 + 0.1 * torch.randn(4096)).to(torch.bfloat16)
pos = torch.randint(-1, 40, (bs, 8), dtype=torch.int32)
out = torch.empty(bs, 4096, dtype=torch.bfloat16)
K._post_norm_rows_kernel[(bs,)](y, w, s, pos, out, 1e-8, TOPK=8, H=4096)
f = y.float()[pos.clamp(min=0).long()] * w.float()
f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-8)
ref = (f * (s * (pos >= 0)).unsqueeze(-1)).sum(1)
err = float((out.float() - ref).norm() / ref.norm()); fails += err > 5e-3
print(f"{'OK ' if err < 5e-3 else 'BAD'} post-norm rows: err {err:.2e}")
print("failures:", fails)
