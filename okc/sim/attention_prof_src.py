# Phase profile of the v4 attention kernel: bundled after attention_sim_src.py. Per case, one
# cold-L2 call with the kernel's clock64 stamps on; prints, per event, the median and max over
# CTAs of its time since the CTA's start, plus the kernel's span and the CTAs' start skew.

ATT_EVENTS = {0: "start", 1: "lengths", 2: "segments", 3: "prologue", 40: "seg end", 41: "partial+atomic",
              42: "merged", 61: "end"}


def run_attn_prof(cases, variants, seed=7_300_000):
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for ci, case in enumerate(cases):
        data = generate_input(seed=seed + ci, device="cuda", **case)
        for name, fn in variants.items():
            fn(_clone_inputs(data))
            prof = torch.zeros(4096 * 64, dtype=torch.int64, device="cuda")
            trial = _clone_inputs(data)
            torch.cuda.synchronize()
            flush.zero_()
            fn(trial, prof=prof)
            torch.cuda.synchronize()
            pr = prof.view(-1, 64).cpu()
            live = pr[(pr[:, 63] != 0) & (pr[:, 61] != 0)]
            if len(live) == 0:
                print(f"prof {name} {_attn_name(case)}: no stamps")
                continue
            g0 = int(live[:, 63].min())
            t0g, t1g = (live[:, 63] - g0).double(), (live[:, 62] - g0).double()  # int64 first: ns since 1970
            cyc = (live[:, 61] - live[:, 0]).double()
            clk = float(cyc.sum() / (t1g - t0g).clamp(min=1).sum())  # cycles per ns
            span = float(t1g.max() - t0g.min()) / 1e3
            skew = float(t0g.max() - t0g.min()) / 1e3
            line = [f"prof {name} {_attn_name(case)}: {len(live)} CTAs, span {span:.1f}us, start skew {skew:.1f}us, clk {clk:.2f}GHz"]
            for k in list(range(0, 4)) + list(range(4, 36)) + [40, 41, 42] + list(range(44, 60)) + [61]:
                col = live[:, k]
                ok = col != 0
                if k >= 4 and k < 36 and int(ok.sum()) < len(live) // 2:
                    continue
                d = (col[ok] - live[ok, 0]).double() / clk / 1e3
                if len(d) == 0:
                    continue
                if 44 <= k < 60:
                    nm = f"T{(k - 44) // 4}:" + ("qk", "smax", "pv-issued", "pv-done")[(k - 44) % 4]
                else:
                    nm = ATT_EVENTS.get(k, (f"t{(k - 4) // 2} ready" if k % 2 == 0 else f"t{(k - 4) // 2} done"))
                line.append(f"{nm} {float(d.median()):.2f}/{float(d.max()):.2f}")
            print(" | ".join(line), flush=True)
        del data
        torch.cuda.empty_cache()
