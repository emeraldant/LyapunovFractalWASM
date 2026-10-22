"""CPU smoke test of muse_decode_layer/submission.py's own parts (fused attention and its
in-prepare check, the projection runner) under Triton's interpreter, SGLang stubbed out."""
import os, sys, types
os.environ["TRITON_INTERPRET"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import interp_patch  # noqa: F401
import torch

for name in ["sgl_kernel", "sgl_kernel.flash_attn", "sgl_kernel.scalar_type", "sglang", "sglang.kernels",
             "sglang.kernels.ops", "sglang.kernels.ops.activation", "sglang.kernels.ops.activation.activation",
             "sglang.kernels.ops.kvcache", "sglang.kernels.ops.kvcache.kvcache", "sglang.kernels.ops.moe",
             "sglang.kernels.ops.moe.moe_wna16_marlin", "sglang.kernels.ops.muse_spark_v12",
             "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing",
             "sglang.kernels.ops.muse_spark_v12.fused_norm_gate", "sglang.kernels.ops.muse_spark_v12.fused_norm_residual",
             "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
             "sglang.kernels.ops.muse_spark_v12.fused_qk_norm", "sglang.srt", "sglang.srt.layers",
             "sglang.srt.layers.moe", "sglang.srt.layers.moe.fused_moe_triton", "sglang.srt.layers.muse_spark_layers",
             "sglang.srt.layers.quantization", "sglang.srt.layers.quantization.marlin_utils",
             "sglang.srt.layers.quantization.marlin_utils_fp4", "sglang.srt.layers.quantization.unquant"]:
    m = sys.modules.setdefault(name, types.ModuleType(name))
    m.__getattr__ = lambda attr: None
sys.path.insert(0, os.path.join(here, "..", "muse_decode_layer"))
import submission as S

gen = torch.Generator().manual_seed(0)
print("attention check:", S._check_attention("cpu", gen), flush=True)
w = (torch.randn(8192, 4096) / 64).to(torch.bfloat16)
x = torch.randn(5, 4096).to(torch.bfloat16)
for plan in [("fma", 8, 8, 256, 2, 4, 2), ("dot", 16, 32, 256, 2, 4, 3), ("dot", 16, 64, 128, 1, 4, 4)]:
    out = torch.zeros(5, 8192, dtype=torch.bfloat16)
    S._run_proj(plan, x, w, 1024, out=out)
    ref = x.double() @ w[:1024].double().T
    print(plan, "err", S._rel(out[:, :1024], ref), "rest zero", bool((out[:, 1024:] == 0).all()), flush=True)
