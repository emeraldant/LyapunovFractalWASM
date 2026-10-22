#!/usr/bin/env python3
"""Download history for every pair in data/pairs.json (resumable; one file per market).

  data/k_trades/<ticker>.json   Kalshi trades   [t_exch, yes_price, count, taker_side('yes'|'no')]
  data/k_candles/<ticker>.json  Kalshi 1-min    [t_end, bid_open, bid_close, ask_open, ask_close, bid_low, bid_high, ask_low, ask_high]
  data/p_trades/<cond>.json     Polymarket      [t, asset, price, size, side, wallet, tx]  (taker side)
  data/p_hist/<token>.json      Polymarket 1-min price history [t, p]
Usage: fetch.py [--start 2026-10-01] [--workers 4]
"""
import argparse
import calendar
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from common import CLOB, DATA, DATA_API, KALSHI, get, load, save

F = lambda x: float(x) if x not in (None, "") else None
lock = threading.Lock()


def have(name):
    return os.path.exists(os.path.join(DATA, name))


def k_trades(ticker, start, end):
    name = f"k_trades/{ticker}.json"
    if have(name):
        return
    out, cur = [], None
    while True:
        p = {"ticker": ticker, "min_ts": start, "max_ts": end, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = get(f"{KALSHI}/markets/trades", p)
        for x in d.get("trades", []):
            t = calendar.timegm(time.strptime(x["created_time"][:19], "%Y-%m-%dT%H:%M:%S")) + float("0." + (x["created_time"][20:-1] or "0"))
            out.append([t, F(x["yes_price_dollars"]), F(x["count_fp"]), x["taker_side"]])
        cur = d.get("cursor")
        if not cur or not d.get("trades"):
            break
    out.sort()
    save(name, out)


def k_candles(series, ticker, start, end):
    name = f"k_candles/{ticker}.json"
    if have(name):
        return
    out, s = [], start
    while s < end:
        e = min(end, s + 3 * 86400)
        d = get(f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": s, "end_ts": e, "period_interval": 1})
        for c in d.get("candlesticks", []):
            b, a = c.get("yes_bid") or {}, c.get("yes_ask") or {}
            out.append([c["end_period_ts"], F(b.get("open_dollars")), F(b.get("close_dollars")), F(a.get("open_dollars")),
                        F(a.get("close_dollars")), F(b.get("low_dollars")), F(b.get("high_dollars")),
                        F(a.get("low_dollars")), F(a.get("high_dollars"))])
        s = e
    out.sort()
    save(name, out)


def p_trades(cond, start, end):
    name = f"p_trades/{cond}.json"
    if have(name):
        return
    out, seen, e = [], set(), end
    while True:
        d = get(f"{DATA_API}/trades", {"market": cond, "limit": 5000, "end": e, "start": start})
        new = 0
        for x in d:
            key = (x.get("transactionHash"), x.get("asset"), x.get("size"), x.get("price"), x.get("proxyWallet"), x.get("side"))
            if key in seen:
                continue
            seen.add(key); new += 1
            out.append([x["timestamp"], x["asset"], x["price"], x["size"], x["side"], x.get("proxyWallet"), x.get("transactionHash")])
        if len(d) < 5000 or not new:
            break
        e = min(x["timestamp"] for x in d)     # page backwards; same-second trades deduped above
    out.sort()
    save(name, out)


def p_hist(token, start, end):
    name = f"p_hist/{token}.json"
    if have(name):
        return
    out, s = [], start
    while s < end:
        e = min(end, s + 7 * 86400)
        d = get(f"{CLOB}/prices-history", {"market": token, "startTs": s, "endTs": e, "fidelity": 1})
        out += [[h["t"], h["p"]] for h in d.get("history", [])]
        s = e
    out = sorted({t: p for t, p in out}.items())
    save(name, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    start = calendar.timegm(time.strptime(a.start, "%Y-%m-%d"))
    end = int(time.time())
    pairs = load("pairs.json")["pairs"]
    jobs = []
    for pr in pairs:
        k, p = pr["k"], pr["p"]
        jobs += [(k_trades, (k["ticker"], start, end)), (k_candles, (k["series"], k["ticker"], start, end)),
                 (p_trades, (p["condition"], start, end)), (p_hist, (p["token"], start, end))]
    jobs = list({(f.__name__, args): (f, args) for f, args in jobs}.values())
    done = [0]
    t0 = time.time()

    def run(job):
        f, args = job
        try:
            f(*args)
        except Exception as e:
            with lock:
                print("ERR", f.__name__, args[0], str(e)[:150], flush=True)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"{done[0]}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print("done", len(jobs), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
