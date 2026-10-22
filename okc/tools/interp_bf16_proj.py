"""CPU check of muse_bf16_proj/submission.py's kernels with Triton's interpreter."""
import os, sys, random, time
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "muse_bf16_proj"))
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_patch  # noqa: F401
import submission as S

torch.manual_seed(0)
SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}
W = {k: (torch.randn(s) / s[1] ** 0.5).to(torch.float32 if k == "router" else torch.bfloat16) for k, s in SHAPES.items()}
state = {k: v.contiguous() for k, v in W.items()}
hi = W["router"].to(torch.bfloat16); lo = (W["router"] - hi.float()).to(torch.bfloat16)
state.update(router_hi=hi, router_lo=lo, router_hi_t=hi.t().contiguous(), router_lo_t=lo.t().contiguous())
total = 0; keys = set(); bad = 0
for bucket in S.BUCKETS:
    for layer in S.LAYERS:
        n_full, k = SHAPES[layer]
        n = 1024 if layer == "unpooler" and bucket <= 8 else n_full
        cands = S._candidates(layer, bucket, n, k)
        total += len(cands); keys.update(cands)
        print(f"{layer:9s} m{bucket:<4d}: {len(cands)} candidates")
print("total", total, "unique plans", len(keys))
if "--count" in sys.argv: sys.exit()
# Run a random sample of plans with an odd m inside the bucket.
random.seed(1)
gen = torch.Generator().manual_seed(0)
sample = []
for bucket in S.BUCKETS:
    for layer in S.LAYERS:
        n_full, k = SHAPES[layer]
        n = 1024 if layer == "unpooler" and bucket <= 8 else n_full
        for plan in random.sample(S._candidates(layer, bucket, n, k), 1):
            sample.append((layer, bucket, n, k, plan))
for layer, bucket, n, k, plan in sample:
    m = max(1, bucket - 3) if bucket > 1 else 1
    x = S._make_x(m, k, "cpu", gen)
    t = time.time()
    out = S._run_plan(plan, x, state, layer, n)
    ok = S._check(out, x, state[layer][:n], layer == "router")
    ref = x.double() @ state[layer][:n].double().T
    err = float((out.double() - ref).norm() / ref.norm())
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {layer:9s} m{m:<4d} {plan} err {err:.2e} dtype {out.dtype} {time.time()-t:.1f}s", flush=True)
print("failures:", bad)
