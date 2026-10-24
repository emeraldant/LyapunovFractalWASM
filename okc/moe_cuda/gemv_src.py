"""Sim template: TRT-GEMV-style direct-to-register streaming upper bound (moe_cuda/gemv_probe.py)."""
import random

import torch

# @bundle sim/harness.py SimBench
# @bundle moe_cuda/gemv_probe.py _GEMV_SRC _GEMV_CPP run_gemv_probe
