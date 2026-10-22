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
for bs, ls in ((1, 0), (5, 32), (33, 224), (257, 96)):
    ids = torch.randint(0, 256, (bs, 8), generator=gen, dtype=torch.int32)
    p = bs * 8
    row_route = torch.full((p,), -7, dtype=torch.int32); pos = torch.full((p,), -7, dtype=torch.int32)
    exp_start = torch.zeros(33, dtype=torch.int32); active = torch.full((32,), -7, dtype=torch.int32)
    n_active = torch.zeros(1, dtype=torch.int32)
    K._route_compact_kernel[(32,)](ids, torch.tensor([ls], dtype=torch.int32), row_route, pos, exp_start, active,
                                   n_active, p, NE=256, E=32, CHUNK=256)
    loc = ((ids.flatten().long() - ls) % 256)
    counts = torch.bincount(loc[loc < 32], minlength=32)
    ok = torch.equal(exp_start[:32].long(), torch.cumsum(counts, 0) - counts) and int(exp_start[32]) == int(counts.sum())
    act = (counts > 0).nonzero().flatten()
    ok &= int(n_active) == len(act) and torch.equal(active[:len(act)].long(), act)
    for e in range(32):
        rows = row_route[exp_start[e]:exp_start[e + 1]].long()
        ok &= torch.equal(rows, (loc == e).nonzero().flatten())                  # route order within an expert
        ok &= torch.equal(pos[rows].long(), torch.arange(int(exp_start[e]), int(exp_start[e + 1])))
    ok &= bool((pos[loc >= 32] == -1).all())
    fails += not ok
    print(f"{'OK ' if ok else 'BAD'} route bs{bs} local_start {ls}: {int(n_active)} active, {int(exp_start[32])} rows")
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
