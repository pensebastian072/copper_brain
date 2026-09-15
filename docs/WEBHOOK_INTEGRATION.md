# Copper Brain -> hq-trading-system webhook integration (P5)

**Status: contract + reader stub only.** Layer-1 execution wiring (TradingView/
Pine intraday) is deferred until Layer-2 findings justify it. Nothing here sends
an order. The copper regime is **advisory**: it can confirm or fail-to-confirm a
copper setup, never veto a trade or route an order.

## How it plugs in (off the hot path)

Copper Brain publishes one local file on its weekly schedule:

    data/regime/copper_regime.json

The hq webhook reads that file the same way it already reads its LLM
`veto_flag.json` — a local read with a fail-safe default, never a network call on
the request path. If the file is missing, unparseable, or older than
`REGIME_STALE_DAYS`, the regime degrades to `neutral` and confirms nothing.

## The alert contract

A copper Pine alert carries the fields the gate needs:

```json
{
  "asset": "COPPER",
  "timeframe": "15m",
  "setup": "macro_confirmed_breakout",
  "direction": "long",
  "tv_signal": true,
  "required_model_regime": "bullish",
  "min_copper_score": 60
}
```

- `direction` (`long`/`short`) implies the confirming regime when
  `required_model_regime` is omitted (`long`->`bullish`, `short`->`bearish`).
- `min_copper_score` is the absolute score threshold; for a short the gate
  requires `score <= -min_copper_score`.

## The decision

`copper_brain.webhook_gate.evaluate_alert(alert)` returns:

```json
{
  "applies": true, "confirm": true, "advisory": true,
  "regime": "bullish", "score": 72, "stale": false,
  "required_regime": "bullish", "min_copper_score": 60,
  "model_version": "v1-score",
  "reason": "regime bullish confirms; score >= 60 met (score=72)"
}
```

`applies=false` means it wasn't a copper alert (caller ignores the gate).
`confirm=false` with `stale=true` is the fail-safe path.

## Two ways for hq to consume it

**A. Import (if hq adds copper_brain to its path):**

```python
from copper_brain.webhook_gate import evaluate_alert
decision = evaluate_alert(alert)          # reads the flag-file, fail-safe
if alert.get("asset") == "COPPER" and not decision["confirm"]:
    ...  # advisory: size down / skip / log, per hq policy
```

**B. Paste the self-contained reader stub** (no copper_brain import needed) —
the canonical copy lives in `copper_brain/webhook_gate.py::HQ_READER_STUB` and
points at the published file by absolute path with the same staleness guard.

## CLI smoke test

```
.venv\Scripts\python -m copper_brain.cli regime
.venv\Scripts\python -m copper_brain.cli gate --alert "{\"asset\":\"COPPER\",\"direction\":\"long\",\"min_copper_score\":60}"
```
