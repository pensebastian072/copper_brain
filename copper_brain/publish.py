"""Publish the regime flag-file — the ONE artifact Copper Brain exists to write.

Builds data/regime/copper_regime.json per the Plan contract and writes it
atomically (tmp + replace) so a reader never sees a half-written file. Fail-safe
(Plan rule 4): if the primary copper price is stale, the regime is forced to
`neutral` and stale=True regardless of the score.

`read_regime()` is the reader the hq-trading-system webhook uses OFF the hot
path: it returns a neutral, vetoing-nothing dict if the file is missing,
unparseable, or older than the stale window — never raising, never blocking.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config
from .score import classify, main_drivers

NEUTRAL_FALLBACK = {
    "asset": "COPPER",
    "copper_regime": "neutral",
    "score": 0,
    "p_up_5d": None,
    "p_up_21d": None,
    "main_drivers": [],
    "veto": False,
    "stale": True,
    "model_version": config.MODEL_VERSION_V1,
    "source_health": {},
}


def _source_health(freshness: dict | None) -> dict:
    if not freshness:
        return {}
    srcs = freshness.get("sources", {})
    return {k: v.get("health") for k, v in srcs.items() if not k.startswith("_")}


def build_regime(scored: pd.DataFrame, missing_inputs: list[str],
                 freshness: dict | None = None,
                 model_version: str = config.MODEL_VERSION_V1) -> dict:
    """Assemble the regime dict from the latest scored row."""
    if scored is None or scored.empty:
        return dict(NEUTRAL_FALLBACK,
                    as_of=datetime.now(timezone.utc).isoformat(),
                    data_through=None, missing_inputs=["all"])

    row = scored.iloc[-1]
    data_through = scored.index[-1]
    age_days = (pd.Timestamp.now().normalize() - data_through.normalize()).days
    stale = age_days > config.REGIME_STALE_DAYS

    score = int(round(float(row["score"])))
    regime = "neutral" if stale else classify(score)

    cc = (freshness or {}).get("sources", {}).get("_crosscheck", {})
    return {
        "asset": "COPPER",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_through": data_through.date().isoformat(),
        "copper_regime": regime,
        "score": score,
        # v1 publishes no calibrated probabilities; v2 (P3) fills these.
        "p_up_5d": None,
        "p_up_21d": None,
        "main_drivers": main_drivers(row),
        "veto": False,                 # advisory only; never vetoes a trade
        "stale": bool(stale),
        "data_age_days": int(age_days),
        "missing_inputs": missing_inputs,
        "crosscheck": cc.get("verdict"),
        "crosscheck_diverging": cc.get("diverging", []),
        "model_version": model_version,
        "source_health": _source_health(freshness),
    }


def publish(regime: dict) -> str:
    """Atomically write the regime dict to data/regime/copper_regime.json."""
    tmp = config.REGIME_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(regime, indent=2))
    tmp.replace(config.REGIME_FILE)
    return str(config.REGIME_FILE)


def read_regime() -> dict:
    """Fail-safe reader for the webhook (off the hot path).

    Returns the published regime, or a neutral/vetoing-nothing fallback if the
    file is missing, unparseable, or older than the stale window. Never raises.
    """
    fb = dict(NEUTRAL_FALLBACK, as_of=None, data_through=None)
    try:
        raw = json.loads(config.REGIME_FILE.read_text())
    except Exception:  # noqa: BLE001 — missing/corrupt -> neutral
        return fb
    # Guard on the file's own freshness even if it claims not-stale.
    as_of = raw.get("as_of")
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(as_of)).total_seconds() / 86400.0
        if age > config.REGIME_STALE_DAYS:
            raw["copper_regime"] = "neutral"
            raw["stale"] = True
    except Exception:  # noqa: BLE001
        raw["copper_regime"] = "neutral"
        raw["stale"] = True
    return raw
