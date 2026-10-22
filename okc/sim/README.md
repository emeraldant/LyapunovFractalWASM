# Sim environment

Final positions come from reruns on held-out data and on shapes near the scored ones, so a
change is only kept if it holds up here, not just on the scored seeds.

## How it runs

The judge's GPUs are the only GPUs we have. A sim is bundled into a real submission
(`tools/make_host.py`) whose untimed `prepare()` runs it first, then does its own work; a
check run (off the board) returns the host's normal result plus the sim's printed report
(`tools/wait.py <job>`; the last 16,000 characters of output come back).

- `harness.py`: the judge's timing protocol. CUDA graph replays, a cold L2 before every
  replay, candidate and SGLang alternating in pairs with the order swapped, the median of
  paired ratios with a bootstrap 95% interval, net of a measured empty-graph replay.
- `attention_sim_src.py`: inputs from the judge's own `generate_input` (copied source, not an
  import), the judge's correctness bound, and three case groups:
  - **tests**: the 16 judge tests (1M-token context, page 64, kv_scale 1/64, the 128-head
    layout): correctness only;
  - **scored**: the 12 timed shapes;
  - **nearby**: 14 shapes around them (other batches, contexts, q_len 2/4, normal / coherent /
    edges distributions).
- CPU: `tools/interp_*.py` run the Triton kernels in Triton's interpreter against the same
  checks at reduced sizes, for correctness before spending a check run.

## Rules we hold ourselves to

From the judge's rules and published experience with kernel benchmarks (KernelBench and its
reward-hacking post-mortems, KernelBot/KernelGuard, SOL-ExecBench; research notes in
`RESEARCH.md`):

1. Never tune on the scored seeds. Every sim case draws a seed the scored runs don't use;
   prepare()-time tuning uses its own random inputs.
2. Choose configurations by shape bucket and require them to hold across the nearby group:
   a change is kept only if the nearby group's minimum speedup stays >= 1.0 and nothing is
   wrong in any group.
3. Correctness is the judge's own bound, on every case including adversarial distributions
   (outliers, skewed routing, keys at window edges, small kv_scale), plus row-wise checks,
   NaN/Inf rejection.
4. Physics floor: a time below the roofline floor (`tools/exact.py`) is a bug in the
   measurement or the kernel until explained.
5. No reward hacking, by construction: no state derived from inputs across calls, no work
   outside the timed call except weight repacking, no reduced precision below the rules, no
   streams beyond what SGLang itself uses (the router's side stream), no touching timing.
6. Submit to the board only when the check run beats the current leader outside the
   judge's noise interval and the sim's nearby group agrees.
