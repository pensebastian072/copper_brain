"""Webhook gate — the advisory decision the hq-trading-system webhook will call
to consult the copper regime OFF the hot path. (Plan P5: contract + reader stub
only; Layer-1 execution wiring is deferred.)

The webhook never blocks on a network call — it reads the locally-published
data/regime/copper_regime.json via publish.read_regime() (fail-safe to neutral
when missing/stale/corrupt) and asks evaluate_alert() whether a copper setup is
confirmed by the fundamental regime.

ADVISORY ONLY. evaluate_alert returns a recommendation; it never sends an order
and never hard-vetoes. A stale/neutral regime yields confirm=False so a copper
breakout is simply *not* fundamentally confirmed — the caller decides what to do
with that (size down, skip, or ignore in advisory mode).

Alert contract (matches the Plan example):
    {
      "asset": "COPPER", "timeframe": "15m", "setup": "macro_confirmed_breakout",
      "direction": "long", "tv_signal": true,
      "required_model_regime": "bullish", "min_copper_score": 60
    }
"""
from __future__ import annotations

from . import config
from .publish import read_regime

# direction -> the regime that confirms it, and the sign the score must take.
_DIRECTION_REGIME = {"long": "bullish", "short": "bearish"}


def evaluate_alert(alert: dict, regime: dict | None = None) -> dict:
    """Return an advisory decision for a copper alert against the current regime.

    Result dict:
      applies        - False if this isn't a copper alert (caller should ignore)
      confirm        - True only if the regime confirms the setup
      reason         - human-readable explanation
      regime, score  - the regime state consulted
      stale          - whether the regime file was stale (fail-safe to neutral)
      advisory       - always True (this gate never routes or vetoes)
    """
    if regime is None:
        regime = read_regime()

    asset = str(alert.get("asset", "")).upper()
    if asset != "COPPER":
        return {"applies": False, "confirm": True, "advisory": True,
                "reason": f"not a copper alert (asset={asset or 'unset'})"}

    cur_regime = regime.get("copper_regime", "neutral")
    score = regime.get("score", 0) or 0
    stale = bool(regime.get("stale"))

    direction = str(alert.get("direction", "")).lower()
    required = alert.get("required_model_regime") or _DIRECTION_REGIME.get(direction)
    min_score = int(alert.get("min_copper_score", 0) or 0)

    base = {"applies": True, "advisory": True, "regime": cur_regime,
            "score": score, "stale": stale,
            "required_regime": required, "min_copper_score": min_score,
            "model_version": regime.get("model_version")}

    # Fail-safe: a stale or neutral regime never confirms (Plan rule 4).
    if stale:
        return {**base, "confirm": False, "reason": "regime stale -> not confirmed"}
    if required and cur_regime != required:
        return {**base, "confirm": False,
                "reason": f"regime {cur_regime} != required {required}"}

    # Score must clear the threshold in the setup's direction.
    if direction == "short" or required == "bearish":
        score_ok = score <= -min_score
        need = f"score <= -{min_score}"
    else:  # long / bullish default
        score_ok = score >= min_score
        need = f"score >= {min_score}"

    if not score_ok:
        return {**base, "confirm": False,
                "reason": f"{need} not met (score={score})"}

    return {**base, "confirm": True,
            "reason": f"regime {cur_regime} confirms; {need} met (score={score})"}


# Self-contained stub the hq-trading-system webhook can drop in WITHOUT importing
# copper_brain (separate repo). Points at the published flag-file by absolute
# path and applies the same fail-safe. Kept here as the canonical reference.
HQ_READER_STUB = f'''
# --- copper regime reader (paste into hq webhook_receiver.py) ---
import json
from datetime import datetime, timezone
from pathlib import Path

COPPER_REGIME_FILE = Path(r"{config.REGIME_FILE}")
COPPER_STALE_DAYS = {config.REGIME_STALE_DAYS}

def read_copper_regime():
    """Off-hot-path read of Copper Brain's advisory regime. Fail-safe neutral."""
    fb = {{"copper_regime": "neutral", "score": 0, "stale": True, "veto": False}}
    try:
        raw = json.loads(COPPER_REGIME_FILE.read_text())
    except Exception:
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > COPPER_STALE_DAYS:
            raw["copper_regime"], raw["stale"] = "neutral", True
    except Exception:
        raw["copper_regime"], raw["stale"] = "neutral", True
    return raw
# --- end stub ---
'''
