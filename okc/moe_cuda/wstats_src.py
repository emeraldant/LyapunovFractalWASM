"""Sim template: statistics of the judge's actual expert weights (codes and scales), to see whether a
lossless repack could move fewer bytes than NVFP4's 4.5 bits per weight."""
import math

import torch


def run_weight_stats(weights, experts=(0, 1, 2, 31)):
    def ent(counts):
        p = counts.double() / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log2()).sum())
    for name in ("w13", "w2"):
        q = weights[name]
        sf = weights[name + "_sf"]
        gs = weights[name + "_gs"]
        print(f"wstats {name}: shape {tuple(q.shape)} sf {tuple(sf.shape)} gs[:4] {gs[:4].tolist()}", flush=True)
        for e in experts:
            codes = torch.cat([(q[e] & 15).flatten(), (q[e] >> 4).flatten()]).long()
            cc = torch.bincount(codes, minlength=16)
            s = sf[e].view(torch.uint8).long()
            sc = torch.bincount(s.flatten(), minlength=256)
            # per row: range of scale bytes and entropy of scale deltas along k
            d = (s[:, 1:] - s[:, :-1]).flatten()
            dc = torch.bincount(d + 255, minlength=511)
            rng = (s.max(1).values - s.min(1).values).float()
            # conditional entropy of a code given its 16-block scale (a proxy for joint modelling)
            print(f"wstats {name} e{e}: code entropy {ent(cc):.3f} bits (top {[round(float(x), 3) for x in (cc.double() / cc.sum()).tolist()]});"
                  f" scale entropy {ent(sc):.3f} bits, distinct {int((sc > 0).sum())}; scale delta entropy {ent(dc):.3f} bits;"
                  f" per-row scale range mean {float(rng.mean()):.1f} max {float(rng.max()):.0f}", flush=True)
