# Muse Spark Open Kernel Challenge

Work for https://openkernelchallenge.dev/ (kernel track: Oct 22 – Nov 4, 2026; serving track
Nov 5 – Nov 19). Each `muse_*` directory holds what `okc init <problem>` writes (TASK.md,
AGENTS.md, template.py, baseline.py, reference.py, checks.py, nvfp4.py) plus our
`submission.py`.

No GPU is available where these were written, so:

- `tools/interp_*.py` run the Triton kernels on CPU with `TRITON_INTERPRET=1` against the
  judge's reference checks (at reduced sizes). `tools/interp_patch.py` fixes two interpreter
  bugs first: float down-casts that drop the rounding carry (1.99 -> 1.0 in E4M3), and BF16
  `tl.dot` multiplying raw bit patterns.
- Where a problem has `prepare()` (all but attention), the submission times its candidate
  kernels against SGLang's own path on the judge's GPU, with CUDA graph replays and a cold
  L2, and keeps SGLang's path for any shape where no candidate is both correct and faster.

Submitting needs a person to run `okc login` once (rules + terms), then:

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
okc login
okc check okc/muse_attention_decode/submission.py   # off the board
okc submit okc/muse_attention_decode/submission.py
```

| problem | status |
|---|---|
| muse_bf16_proj | Triton skinny GEMM / GEMV, split-K, tuned in prepare() with SGLang fallback |
| muse_attention_decode | fused 2-launch split-KV attention (QK-norm, RoPE, FP8 cache write, gated norm); passes the reference in the interpreter |
| muse_moe_w4a16 | starter only |
| muse_decode_layer | starter only |
| muse_prefill_layer | starter only |
