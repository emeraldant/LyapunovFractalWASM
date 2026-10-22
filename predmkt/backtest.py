#!/usr/bin/env python3
"""Backtest: rest quotes on the illiquid venue (Polymarket), hedge each fill by crossing Kalshi.

Strategy (per pair, both sides):
  bid on PM YES at  b = floor_tick(Kalshi_bid - Kalshi_taker_fee(per contract) - m)
  ask on PM YES at  a = ceil_tick (Kalshi_ask + Kalshi_taker_fee(per contract) + m)
  When a PM taker sells into our bid, we sell the same size of YES on Kalshi at its bid (taker), and vice versa.
  m = required margin after the hedge fee. Fill price is OUR price (maker). Polymarket maker/taker fees are 0 on these markets.

Fill model (conservative): a PM taker print strictly through our price fills us (print price < b for bids, > a for asks).
A print exactly at our price fills us only with --touch (optimistic, ignores queue).
Fill size = min(--size, print size).

Two information models for where Kalshi's bid/ask is:
  slow: only 1-minute candles. The quote for minute [T, T+60) is set at T from the candle that ended
        at T - 60*lag_min, i.e. 0 to 60 s plus lag_min stale. The hedge executes at the first Kalshi
        quote after t (next minute boundary). All intra-minute moves hit us as adverse selection.
  fast: perfect, zero-latency tracking of Kalshi (estimated from Kalshi prints inside the minute, else the nearer
        of the candle's open/close). Quote and hedge use the same Kalshi price, so every fill earns >= m by
        construction; this is an UPPER BOUND on what a very fast trader sees, and mainly counts opportunities.
Usage: backtest.py [--m 0.01] [--size 25] [--lag 1] [--touch] [--json out.json]
"""
import argparse
import bisect
import collections
import json
import math
import statistics as st

from common import load


def kfee(p, c):
    """Kalshi taker fee for one order of c contracts at yes-price p (dollars)."""
    return math.ceil(0.07 * c * p * (1 - p) * 100 - 1e-9) / 100


class KMarket:
    def __init__(self, candles, trades):
        c = [x for x in candles if x[0] is not None]
        self.end = [x[0] for x in c]
        self.bo = [x[1] for x in c]; self.bc = [x[2] for x in c]
        self.ao = [x[3] for x in c]; self.ac = [x[4] for x in c]
        self.tt = [x[0] for x in trades]
        self.tp = [x[1] for x in trades]
        self.ts = [x[3] for x in trades]
        self.idx = {e: i for i, e in enumerate(self.end)}

    def candle_ending(self, e):
        return self.idx.get(e)

    def est(self, t):
        """(bid, ask) estimate at time t from prints in the current minute, else nearer of candle open/close."""
        b0 = int(t // 60) * 60
        i = self.idx.get(b0 + 60)
        if i is None:
            return None, None
        bid = self.bo[i] if t - b0 < 30 else self.bc[i]
        ask = self.ao[i] if t - b0 < 30 else self.ac[i]
        j = bisect.bisect_left(self.tt, t) - 1
        lastno = lastyes = None
        while j >= 0 and self.tt[j] >= b0:
            if self.ts[j] == "no" and lastno is None:
                lastno = self.tp[j]
            if self.ts[j] == "yes" and lastyes is None:
                lastyes = self.tp[j]
            if lastno is not None and lastyes is not None:
                break
            j -= 1
        return (lastno if lastno is not None else bid), (lastyes if lastyes is not None else ask)


def _prints_before(km, x, b0):
    j = bisect.bisect_left(km.tt, x) - 1
    lastno = lastyes = None
    while j >= 0 and km.tt[j] >= b0:
        if km.ts[j] == "no" and lastno is None:
            lastno = km.tp[j]
        if km.ts[j] == "yes" and lastyes is None:
            lastyes = km.tp[j]
        if lastno is not None and lastyes is not None:
            break
        j -= 1
    return lastno, lastyes


def causal(km, x):
    """Kalshi (bid, ask) knowable at time x: last taker print in this minute, else the minute's opening quote."""
    b0 = int(x // 60) * 60
    i = km.idx.get(b0 + 60)
    if i is None:
        return None, None
    no, yes = _prints_before(km, x, b0)
    return (no if no is not None else km.bo[i]), (yes if yes is not None else km.ao[i])


def truth(km, x):
    """Best estimate of the true Kalshi (bid, ask) at x for pricing a hedge (uses the minute's close: hindsight)."""
    b0 = int(x // 60) * 60
    i = km.idx.get(b0 + 60)
    if i is None:
        return None, None
    f = (x - b0) / 60
    no, yes = _prints_before(km, x, b0)
    lerp = lambda o, c: None if o is None or c is None else o + (c - o) * f
    return (no if no is not None else lerp(km.bo[i], km.bc[i])), (yes if yes is not None else lerp(km.ao[i], km.ac[i]))


def run_pair_lat(pr, km, ev, a, L):
    """Quote from information L seconds old; hedge executes L seconds after the fill (round trip L+L)."""
    tick = float(pr["p"].get("tick") or 0.01)
    fills = []
    free_at = {"sell_yes": 0.0, "buy_yes": 0.0}   # one resting order per side: after a fill, re-post only once hedged
    for t, kind, p, size in ev:
        S = min(a.size, size)
        if S < 1 or t < free_at[kind]:
            continue
        kb, ka = causal(km, t - L)
        if kb is None or ka is None or ka - kb > a.max_spread:
            continue
        hb, ha = truth(km, t + L)
        if hb is None or ha is None:
            continue
        if kind == "sell_yes":
            fee = kfee(kb, S) / S
            b = math.floor((kb - fee - a.m) / tick + 1e-9) * tick
            if ((p < b - 1e-9) or (a.touch and abs(p - b) < 1e-9)) and 0.02 <= b <= 0.98:
                fills.append(("lat", "bid", t, S, b, kb, hb, kfee(hb, S) / S, kb - hb))
                free_at[kind] = t + 2 * L + 1
        else:
            fee = kfee(ka, S) / S
            ask = math.ceil((ka + fee + a.m) / tick - 1e-9) * tick
            if ((p > ask + 1e-9) or (a.touch and abs(p - ask) < 1e-9)) and 0.02 <= ask <= 0.98:
                fills.append(("lat", "ask", t, S, ask, ka, ha, kfee(ha, S) / S, ha - ka))
                free_at[kind] = t + 2 * L + 1
    return fills


def pm_events(trades, token, no_token):
    """Normalise PM taker prints to YES-equivalent (t, kind, price, size); kind is the taker's action on YES."""
    out, seen = [], set()
    for t, asset, price, size, side, wallet, tx in trades:
        if asset == token:
            kind, p = ("sell_yes" if side == "SELL" else "buy_yes"), price
        elif asset == no_token:
            kind, p = ("sell_yes" if side == "BUY" else "buy_yes"), 1 - price
        else:
            continue
        key = (tx, kind, round(p, 3), round(size, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append((t, kind, p, size))
    out.sort()
    return out


def run_pair(pr, km, ev, a):
    tick = float(pr["p"].get("tick") or 0.01)
    fills = []
    nopp = 0
    for t, kind, p, size in ev:
        S = min(a.size, size)
        if S < 1:
            continue
        est_b, est_a = km.est(t)
        # ---- fast model: quote tracks Kalshi instantly
        if est_b is not None and est_a is not None and est_a - est_b <= a.max_spread:
            if kind == "sell_yes":
                fee = kfee(est_b, S) / S
                b = math.floor((est_b - fee - a.m) / tick + 1e-9) * tick
                ok = (p < b - 1e-9) or (a.touch and abs(p - b) < 1e-9)
                if ok and 0.02 <= b <= 0.98:
                    fills.append(("fast", "bid", t, S, b, est_b, est_b, fee, 0))
            else:
                fee = kfee(est_a, S) / S
                ask = math.ceil((est_a + fee + a.m) / tick - 1e-9) * tick
                ok = (p > ask + 1e-9) or (a.touch and abs(p - ask) < 1e-9)
                if ok and 0.02 <= ask <= 0.98:
                    fills.append(("fast", "ask", t, S, ask, est_a, est_a, fee, 0))
        # ---- slow model: candle-based, with lag
        Tm = int(t // 60) * 60
        i = km.candle_ending(Tm - 60 * a.lag)
        h = km.candle_ending(int(math.ceil((t + 1e-9) / 60)) * 60 + 60)   # candle starting at the next boundary
        if i is None or h is None:
            continue
        kb, ka = km.bc[i], km.ac[i]
        if kb is None or ka is None or ka - kb > a.max_spread:
            continue
        if kind == "sell_yes":
            fee = kfee(kb, S) / S
            b = math.floor((kb - fee - a.m) / tick + 1e-9) * tick
            ok = (p < b - 1e-9) or (a.touch and abs(p - b) < 1e-9)
            hb = km.bo[h]
            if ok and 0.02 <= b <= 0.98 and hb is not None:
                fills.append(("slow", "bid", t, S, b, kb, hb, kfee(hb, S) / S, kb - hb))
        else:
            fee = kfee(ka, S) / S
            ask = math.ceil((ka + fee + a.m) / tick - 1e-9) * tick
            ok = (p > ask + 1e-9) or (a.touch and abs(p - ask) < 1e-9)
            ha = km.ao[h]
            if ok and 0.02 <= ask <= 0.98 and ha is not None:
                fills.append(("slow", "ask", t, S, ask, ka, ha, kfee(ha, S) / S, ha - ka))
    return fills


def pnl(f):
    model, side, t, S, px, info, hedge, fee, drift = f
    gross = (hedge - px) if side == "bid" else (px - hedge)     # sell Kalshi at hedge bid / buy Kalshi at hedge ask
    return S * (gross - fee)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.01)
    ap.add_argument("--size", type=int, default=25)
    ap.add_argument("--lag", type=int, default=1, help="extra minutes of stale Kalshi info in the slow model")
    ap.add_argument("--max-spread", type=float, default=0.03, help="only quote when Kalshi's spread is at most this")
    ap.add_argument("--touch", action="store_true")
    ap.add_argument("--lat", default="0,1,2,5,10,20,30,60,120", help="comma list of latencies (s) for the latency curve")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    P = load("pairs.json")["pairs"]
    res = collections.defaultdict(list)
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    for pr in P:
        k, p = pr["k"], pr["p"]
        cand = load(f"k_candles/{k['ticker']}.json") or []
        kt = load(f"k_trades/{k['ticker']}.json") or []
        pt = load(f"p_trades/{p['condition']}.json") or []
        if len(cand) < 100 or not pt:
            continue
        km = KMarket(cand, kt)
        ev = pm_events(pt, p["token"], p["no_token"])
        for f in run_pair(pr, km, ev, a):
            res[f[0]].append(f)
            per[f[0]][pr["race"] + " " + (k["label"] or "")[:16]] += pnl(f)
    out = {}
    lats = [float(x) for x in a.lat.split(",") if x]
    curve = []
    for L in lats:
        F, pairs_hit = [], set()
        for pr in P:
            k, p = pr["k"], pr["p"]
            cand = load(f"k_candles/{k['ticker']}.json") or []
            pt = load(f"p_trades/{p['condition']}.json") or []
            if len(cand) < 100 or not pt:
                continue
            kt = load(f"k_trades/{k['ticker']}.json") or []
            fl = run_pair_lat(pr, KMarket(cand, kt), pm_events(pt, p["token"], p["no_token"]), a, L)
            F += fl
            if fl:
                pairs_hit.add(pr["race"] + k["label"])
        if F:
            c = sum(f[3] for f in F); tot = sum(pnl(f) for f in F)
            curve.append({"latency_s": L, "fills": len(F), "net_usd": round(tot, 2), "net_c_per_contract": round(100 * tot / c, 2),
                          "locked_at_quote_c": round(100 * sum(f[3] * ((f[5] - f[4]) if f[1] == "bid" else (f[4] - f[5])) for f in F) / c, 2),
                          "adverse_drift_c": round(100 * sum(f[3] * f[8] for f in F) / c, 2),
                          "fees_c": round(100 * sum(f[3] * f[7] for f in F) / c, 2),
                          "loss_share": round(sum(pnl(f) < 0 for f in F) / len(F), 2), "pairs": len(pairs_hit)})
        else:
            curve.append({"latency_s": L, "fills": 0})
    out["latency_curve"] = curve
    for model in ("fast", "slow"):
        F = res[model]
        if not F:
            out[model] = {"fills": 0}
            continue
        c = sum(f[3] for f in F)
        total = sum(pnl(f) for f in F)
        drifts = [f[8] for f in F]
        out[model] = {
            "fills": len(F), "contracts": c, "net_pnl_usd": round(total, 2),
            "net_per_contract_cents": round(100 * total / c, 2),
            "fees_per_contract_cents": round(100 * sum(f[3] * f[7] for f in F) / c, 2),
            "locked_edge_at_quote_cents": round(100 * sum(f[3] * ((f[5] - f[4]) if f[1] == "bid" else (f[4] - f[5])) for f in F) / c, 2),
            "mean_adverse_drift_cents": round(100 * sum(f[3] * f[8] for f in F) / c, 2),
            "share_fills_with_loss": round(sum(pnl(f) < 0 for f in F) / len(F), 3),
            "share_fills_drift_ge_1c": round(sum(abs(x) >= 0.01 for x in drifts) / len(F), 3),
            "bid_fills": sum(f[1] == "bid" for f in F), "ask_fills": sum(f[1] == "ask" for f in F),
            "pairs_with_fills": len(per[model]),
            "top5_share_of_pnl": round(sum(sorted(per[model].values(), reverse=True)[:5]) / total, 2) if total > 0 else None,
            "top": sorted(((k, round(v, 2)) for k, v in per[model].items()), key=lambda kv: -kv[1])[:6],
        }
    print(json.dumps({"params": {"m": a.m, "size": a.size, "lag_min": a.lag, "max_spread": a.max_spread, "touch": a.touch}, **out}, indent=1))
    if a.json:
        json.dump({"params": vars(a), "fills": {m: res[m] for m in res}}, open(a.json, "w"))


if __name__ == "__main__":
    main()
