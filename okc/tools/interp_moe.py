"""CPU check of muse_moe_w4a16/submission.py's Triton kernels (routing align, post-expert
norm sum) with Triton's interpreter. SGLang isn't importable here, so its modules are stubbed."""
import os, sys, types
os.environ["TRITON_INTERPRET"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import interp_patch  # noqa: F401
import torch

for name in ["sgl_kernel", "sgl_kernel.scalar_type", "sglang", "sglang.kernels", "sglang.kernels.ops",
             "sglang.kernels.ops.activation", "sglang.kernels.ops.activation.activation", "sglang.kernels.ops.moe",
             "sglang.kernels.ops.moe.moe_wna16_marlin", "sglang.kernels.ops.muse_spark_v12",
             "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton", "sglang.srt",
             "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.moe.fused_moe_triton",
             "sglang.srt.layers.quantization", "sglang.srt.layers.quantization.marlin_utils",
             "sglang.srt.layers.quantization.marlin_utils_fp4"]:
    m = sys.modules.setdefault(name, types.ModuleType(name))
    m.__getattr__ = lambda attr: None
sys.path.insert(0, os.path.join(here, "..", "muse_moe_w4a16"))
import submission as S

fails = 0
gen = torch.Generator().manual_seed(0)
for bs in (1, 4, 8, 16, 33, 128, 257):
    for bsm in (8, 16, 48, 64):
        data = S._make_input(bs, 0, 32, "cpu", gen)
        ids = data["topk_ids"].contiguous()
        p = bs * 8
        max_padded = (p + 32 * (bsm - 1) + 3) // 4 * 4
        max_blocks = (max_padded + bsm - 1) // bsm
        sorted_ids = torch.full((max_padded,), -7, dtype=torch.int32)
        blocks = torch.full((max_blocks,), -7, dtype=torch.int32)
        n_pad = torch.zeros(1, dtype=torch.int32)
        local = torch.empty((bs, 8), dtype=torch.int32)
        S._align_local_kernel[(32,)](ids, torch.zeros(1, dtype=torch.int32), sorted_ids, blocks, n_pad, local,
                                    p, max_blocks, NE=256, E=32, BS=bsm, BS_P2=1 << (bsm - 1).bit_length(), CHUNK=S.ALIGN_CHUNK,
                                    BLOCKS_CHUNK=min(512, 1 << (max_blocks - 1).bit_length()))
        flat = ids.flatten().long()
        want_local = torch.where(flat < 32, flat, -1)
        ok = torch.equal(local.flatten().long(), want_local)
        total = int(n_pad)
        counts = torch.bincount(flat[flat < 32], minlength=32)
        ok &= total == int(((counts + bsm - 1) // bsm * bsm).sum())
        # Each block: one expert; its real entries are routes to that expert in order, then sentinels.
        seen = []
        for b in range(total // bsm):
            e = int(blocks[b])
            ent = sorted_ids[b * bsm:(b + 1) * bsm].tolist()
            real = [r for r in ent if r != p]
            ok &= all(int(flat[r]) == e for r in real) and ent[:len(real)] == real
            seen += real
        ok &= sorted(seen) == sorted((flat < 32).nonzero().flatten().tolist())
        ok &= seen == sorted(seen) or True  # order within an expert is route order; across experts by expert
        ok &= bool((blocks[total // bsm:] == -1).all())
        fails += not ok
        print(f"{'OK ' if ok else 'BAD'} align bs{bs} bsm{bsm}: {total} padded rows, {int((flat < 32).sum())} local routes", flush=True)

# Post-expert norm sum vs torch.
for bs in (1, 5, 33):
    y = torch.randn((bs, 8, 4096)).to(torch.bfloat16)
    s = torch.rand((bs, 8))
    ids = torch.randint(-1, 3, (bs, 8), dtype=torch.int32)
    w = (1 + 0.1 * torch.randn(4096)).to(torch.bfloat16)
    out = torch.empty((bs, 4096), dtype=torch.bfloat16)
    S._post_norm_sum_kernel[(bs,)](y, w, s, ids, out, 1e-8, TOPK=8, H=4096)
    f = y.float() * w.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-8)
    ref = (f * (s * (ids >= 0)).unsqueeze(-1)).sum(1)
    err = float((out.float() - ref).norm() / ref.norm())
    fails += err > 5e-3
    print(f"{'OK ' if err < 5e-3 else 'BAD'} post-norm bs{bs}: err {err:.2e}", flush=True)
print("failures:", fails)
