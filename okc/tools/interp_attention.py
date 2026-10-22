"""CPU check of muse_attention_decode/submission.py with Triton's interpreter, against the
judge's reference.check_implementation, on the test cases at reduced context lengths."""
import os, sys, time
os.environ["TRITON_INTERPRET"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "muse_attention_decode"))
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_patch  # noqa: F401
interp_patch.stub_sglang()
import reference as R
import submission as S
R._fa3_scheduler_metadata = lambda *a, **k: None  # FA3 only; not on CPU

CASES = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=2, ctx=2600, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=2, ctx=3000, heads="tp8", layer="full"),
    dict(batch=2, ctx=700, heads="dp", layer="full", spread="mixed"),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=2, ctx=3000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=2000, heads="tp8", layer="full", kv_scale=0.015625),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
    dict(batch=1, ctx=3000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=2, ctx=2500, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
]
only = [int(a) for a in sys.argv[1:]]
fails = 0
for i, c in enumerate(CASES):
    if only and i not in only:
        continue
    t = time.time()
    data = R.generate_input(seed=100 + i, device="cpu", **c)
    if os.environ.get("VARIANT") == "v3":
        cfg = data["config"]
        out = S._attn3(data["qkv"].reshape(cfg["batch"] * cfg["q_len"], -1), data, cfg,
                       block_n=32, min_chunk=128).view(cfg["batch"], cfg["q_len"], -1)
    else:
        out = S.kernel_fn(data)
    raw = R.collect(data, out)
    ok, msg = R.check_implementation(data, raw)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {i} {c}: {msg} ({time.time()-t:.1f}s)", flush=True)
print("failures:", fails)
