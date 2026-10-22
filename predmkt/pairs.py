#!/usr/bin/env python3
"""Match the same outcome on Kalshi and Polymarket for 2026 Senate, House, Governor and control markets.

A pair is one Kalshi market and one Polymarket YES token that resolve on the same event
(same race, same candidate or party). Output: data/pairs.json with current quotes, 24h volume
and each venue's fee terms, so the backtest knows which side is liquid.
Usage: pairs.py
"""
import json
import re
import time

from common import GAMMA, KALSHI, STATES, get, save

SERIES = {"S": r"^(KX)?SENATE([A-Z]{2})$", "H": r"^(KX)?HOUSE([A-Z]{2})(\d+)$",
          "G": r"^(KX)?GOV(PARTY)?([A-Z]{2})$", "C": r"^CONTROL([SH])$"}
PARTY = {"r": "R", "republican": "R", "gop": "R", "d": "D", "democratic": "D", "democrat": "D", "i": "I", "independent": "I"}
F = lambda x: float(x) if x not in (None, "") else None


def race_from_series(t):
    if m := re.match(SERIES["S"], t):
        return "S-" + m.group(2)
    if m := re.match(SERIES["H"], t):
        return f"H-{m.group(2)}-{int(m.group(3)):02d}"
    if m := re.match(SERIES["G"], t):
        return "G-" + m.group(3)
    if m := re.match(SERIES["C"], t):
        return "C-" + m.group(1)
    return None


def race_from_pm(title):
    if m := re.match(r"^(.+) (Senate|Governor) Election Winner$", title):
        st = STATES.get(m.group(1))
        return st and ("S-" if m.group(2) == "Senate" else "G-") + st
    if m := re.match(r"^([A-Z]{2})-(\d+) House Election Winner$", title):
        return f"H-{m.group(1)}-{int(m.group(2)):02d}"
    if m := re.match(r"^Which party will win the (Senate|House) in 2026\?$", title):
        return "C-" + m.group(1)[0]
    return None


def party_of(label):
    m = re.search(r"\(([RDI])\)\s*$", label)
    if m:
        return m.group(1)
    w = label.lower().replace(" party", "").strip()
    return PARTY.get(w)


def surname(label):
    name = re.sub(r"\([^)]*\)", "", label)
    name = re.sub(r"\b(sen|rep|gov|dr|jr|sr|ii|iii)\b\.?", "", name, flags=re.I)
    toks = [t for t in re.findall(r"[A-Za-z'-]+", name) if len(t) > 1]
    return toks[-1].lower() if toks else None


def kalshi_markets():
    series = get(f"{KALSHI}/series", {"category": "Elections", "limit": 1000})["series"]
    out = []
    for s in series:
        race = race_from_series(s["ticker"])
        if not race:
            continue
        evs = get(f"{KALSHI}/events", {"series_ticker": s["ticker"], "with_nested_markets": "true", "status": "open"})
        for e in evs.get("events", []):
            if re.search(r"20(28|29|30)", e.get("title", "") + e.get("sub_title", "")) or re.search(r"-(28|30)", e["event_ticker"]):
                continue
            for m in e.get("markets", []):
                if m.get("status") != "active":
                    continue
                label = m.get("yes_sub_title") or m.get("subtitle") or ""
                tail = m["ticker"].rsplit("-", 1)[-1]
                out.append({"race": race, "ticker": m["ticker"], "series": s["ticker"], "event": e["event_ticker"],
                            "label": label, "party": party_of(label) or (tail if tail in ("R", "D", "I") else None),
                            "surname": None if party_of(label) and "(" not in label else surname(label),
                            "bid": F(m.get("yes_bid_dollars")), "ask": F(m.get("yes_ask_dollars")),
                            "vol24": F(m.get("volume_24h_fp")), "fee_type": s.get("fee_type")})
        time.sleep(0.05)
    return out


def pm_markets():
    out, off = [], 0
    while True:
        evs = get(f"{GAMMA}/events", {"tag_slug": "midterms", "closed": "false", "limit": 100, "offset": off})
        if not evs:
            break
        for e in evs:
            race = race_from_pm(e.get("title", ""))
            if not race:
                continue
            for m in e.get("markets", []):
                if not m.get("active") or m.get("closed") or not m.get("clobTokenIds"):
                    continue
                label = m.get("groupItemTitle") or ""
                if re.match(r"^(Other|Person [A-Z]|Option [A-Z]|[A-Z])$", label):
                    continue
                yes, no = json.loads(m["clobTokenIds"])
                fs = m.get("feeSchedule") or {}
                out.append({"race": race, "token": yes, "no_token": no, "condition": m["conditionId"], "label": label,
                            "party": party_of(label), "surname": None if party_of(label) and "(" not in label else surname(label),
                            "bid": F(m.get("bestBid")), "ask": F(m.get("bestAsk")), "vol24": F(m.get("volume24hr")),
                            "fee_rate": fs.get("rate", 0) if m.get("feesEnabled") else 0, "fee_exp": fs.get("exponent", 1),
                            "rewards_daily": sum(r.get("rewardsDailyRate", 0) for r in (m.get("clobRewards") or [])),
                            "rewards_max_spread": m.get("rewardsMaxSpread"), "tick": m.get("orderPriceMinTickSize")})
        off += 100
    return out


def match(K, P):
    pairs = []
    by_race = {}
    for p in P:
        by_race.setdefault(p["race"], []).append(p)
    for k in K:
        cands = by_race.get(k["race"], [])
        hit = None
        if k["surname"]:
            hs = [p for p in cands if p["surname"] == k["surname"]]
            hit = hs[0] if len(hs) == 1 else None
        if not hit and k["party"]:
            # party market on Kalshi vs party-labelled market on PM, or the only candidate of that party
            hs = [p for p in cands if p["party"] == k["party"]]
            if len(hs) == 1 and (k["surname"] is None or hs[0]["surname"] in (None, k["surname"])):
                hit = hs[0]
        if hit:
            pairs.append({"race": k["race"], "k": k, "p": hit})
    return pairs


def main():
    K, P = kalshi_markets(), pm_markets()
    pairs = match(K, P)
    save("pairs.json", {"t": time.time(), "pairs": pairs, "n_kalshi": len(K), "n_pm": len(P)})
    print(f"kalshi markets {len(K)}, pm markets {len(P)}, matched pairs {len(pairs)}")
    races = {p['race'][0] for p in pairs}
    for r in sorted(races):
        print(r, sum(1 for p in pairs if p["race"][0] == r))


if __name__ == "__main__":
    main()
