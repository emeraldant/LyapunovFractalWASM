# Quote the illiquid venue, hedge the liquid one (Kalshi <-> Polymarket, 2026 midterms)

Stdlib-only Python. Data is downloaded from public APIs and not committed (`data/` is git-ignored).

    python3 pairs.py              # match the same outcome on both venues -> data/pairs.json (~1 min)
    python3 fetch.py --workers 6  # Kalshi trades + 1-min candles, Polymarket trades + 1-min prices (~6 min)
    python3 backtest.py --m 0.01 --size 25 --lat 0,1,2,5,10,30,60

Strategy: rest a bid/ask on Polymarket YES at Kalshi's bid/ask minus/plus Kalshi's taker fee and a margin `m`;
when a Polymarket taker trades through it, hedge by crossing Kalshi. Fill price is ours (maker); Polymarket fees are 0
on these markets. Fills need a print strictly through our price (`--touch` also counts prints at our price).
After a fill the side is idle for 2*latency+1 s (one resting order per side).

Latency model: quote uses Kalshi information `L` seconds old, hedge executes `L` seconds after the fill.
Kalshi history is 1-minute candles plus timestamped prints, so sub-minute Kalshi quotes are reconstructed from prints
(`causal()` for what a trader could know, `truth()` for hedge prices). Not modelled: Kalshi depth at the hedge,
Polymarket queue position, rewards, resolution-rule mismatches between paired markets, capital costs.
