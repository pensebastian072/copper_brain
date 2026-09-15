# Copper Brain

Dalio-style fundamental regime model for copper. Pulls free supply, demand,
inventory, macro, and price data, then outputs a directional regime
(bullish / bearish / neutral) over a 5–21 trading-day horizon as a JSON
flag-file that a TradingView webhook system can use to confirm intraday trades.

**Advisory only. Never sends orders. Paper-research stage.**

See [`PLAN.md`](PLAN.md) for the full build plan, data contracts, phases, and
overfitting controls.

## Status

Planned (2026-06-21). Build begins at Phase 0 + Phase 1 (scaffold + data layer).

## Quick map

- **Layer 2 — Copper Brain** (this repo): fundamental regime model.
- **Layer 1 — TradingView execution** (deferred): 15m/30m copper breakout with
  DXY + risk-proxy filters; reads this model's regime before firing.

## Update (once built)

```
python -m copper_brain.update            # pull newest data, recompute v1 score, republish
python -m copper_brain.update --retrain  # also retrain v2 model behind the overfit gate
```

Weekly auto-run registered via `scripts/register_weekly_update.ps1`.
