# Copper Brain

<!-- one-tap-install -->
[![Download ZIP](https://img.shields.io/badge/Download-ZIP-2ea44f?style=for-the-badge&logo=github)](https://github.com/pensebastian072/copper_brain/archive/refs/heads/main.zip)

**Run it on your computer in 3 steps:** 1) [download the ZIP](https://github.com/pensebastian072/copper_brain/archive/refs/heads/main.zip) · 2) unzip it · 3) double-click **`install.bat`** (Windows) or run **`./install.sh`** (macOS/Linux).
The dashboard opens in your browser at `http://127.0.0.1:8077` - it runs only on your machine. Next time use `start.bat` / `./start.sh`.
<!-- one-tap-install -->

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
