"""Import before Triton kernels run under TRITON_INTERPRET=1: replaces the interpreter's
float conversion, which drops the carry when rounding overflows the mantissa (1.99 -> 1.0
in E4M3, the same at BF16), with torch's round-to-nearest-even casts as the GPU does."""
import numpy as np
import torch
import triton.language as tl
from triton.runtime import interpreter as _interp

_TORCH = {
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp8e4nv": torch.float8_e4m3fn, "fp8e5": torch.float8_e5m2,
}
_orig = _interp._convert_float


def _convert_float(input, input_dtype, output_dtype, rounding_mode):
    src, dst = _TORCH.get(input_dtype.name), _TORCH.get(output_dtype.name)
    if src is None or dst is None:
        return _orig(input, input_dtype, output_dtype, rounding_mode)
    bits_in = getattr(np, f"uint{input_dtype.primitive_bitwidth}")
    bits_out = getattr(torch, f"uint{output_dtype.primitive_bitwidth}")
    raw = np.ascontiguousarray(np.frombuffer(input.tobytes(), dtype=bits_in)).astype(np.int64)
    t = torch.from_numpy(raw).to(getattr(torch, f"int{input_dtype.primitive_bitwidth}") if input_dtype.primitive_bitwidth > 8 else torch.uint8)
    val = t.view(src).to(torch.float32)
    if dst in (torch.float8_e4m3fn,):
        val = val.clamp(-448.0, 448.0)  # satfinite, as cvt.rn.satfinite does
    out = val.to(dst).view(torch.uint8 if output_dtype.primitive_bitwidth == 8 else getattr(torch, f"int{output_dtype.primitive_bitwidth}"))
    out_np = out.numpy().view(getattr(np, f"uint{output_dtype.primitive_bitwidth}"))
    return out_np.reshape(input.shape)


_interp._convert_float = _convert_float


def _to_f32(handle):
    name = handle.dtype.name
    if name in ("bf16", "fp8e4nv", "fp8e5", "fp16"):
        return _convert_float(handle.data, handle.dtype, tl.float32, None).view(np.float32)
    return handle.data


def _create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
    # The interpreter multiplies BF16 operands' raw bits; widen them (exactly) instead.
    out = np.matmul(_to_f32(a).astype(np.float64), _to_f32(b).astype(np.float64)).astype(d.data.dtype)
    return _interp.TensorHandle(out + d.data, d.dtype.scalar)


_interp.InterpreterBuilder.create_dot = _create_dot


def stub_sglang():
    """Make `from sglang... import x` / `from sgl_kernel... import x` succeed on CPU: every
    attribute is None. Only for interpreter tests of our own Triton code paths."""
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("sglang", "sgl_kernel"):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda attr: None
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())
