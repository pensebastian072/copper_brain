"""Copper Score v1 — transparent fixed-weight combination of the six component
sub-scores into a [-100, +100] score and a bullish/bearish/neutral regime.

Weights are fixed a priori in config.SCORE_WEIGHTS (Plan rule 3). Nothing here is
fitted on outcomes — this is the deliberately simple, overfit-proof baseline the
v2 ML model (P3) must beat behind the PBO/Deflated-Sharpe gate before it can
publish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .labels import forward_direction

_COMP_COLS = {c: f"c_{c}" for c in config.SCORE_WEIGHTS}

# Human-readable driver phrases by component sign.
_DRIVER_LABELS = {
    "trend":     ("trend_positive", "trend_negative"),
    "inventory": ("inventories_tight", "inventories_loose"),
    "demand":    ("demand_firm", "demand_soft"),
    "usd_rates": ("usd_rates_supportive", "usd_rates_headwind"),
    "risk":      ("risk_on", "risk_off"),
    "news":      ("news_positive", "news_negative"),
}


def score_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Add a `score` column (weighted sum of components * 100, clipped)."""
    if features is None or features.empty:
        return features
    out = features.copy()
    total = pd.Series(0.0, index=out.index)
    for comp, col in _COMP_COLS.items():
        w = config.SCORE_WEIGHTS[comp] / 100.0
        total = total + w * out[col].fillna(0.0)
    out["score"] = (total * 100.0).clip(-100, 100)
    return out


def classify(score: float) -> str:
    if score >= config.SCORE_BULLISH_THRESHOLD:
        return "bullish"
    if score <= config.SCORE_BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def main_drivers(feature_row: pd.Series, top_n: int = 3) -> list[str]:
    """Top weighted component contributions, as signed driver phrases."""
    contribs = {}
    for comp, col in _COMP_COLS.items():
        val = feature_row.get(col, 0.0)
        if pd.isna(val):
            val = 0.0
        contribs[comp] = (config.SCORE_WEIGHTS[comp] / 100.0) * val
    ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drivers = []
    for comp, contrib in ranked[:top_n]:
        if contrib == 0:
            continue
        pos, neg = _DRIVER_LABELS[comp]
        drivers.append(pos if contrib > 0 else neg)
    return drivers


def walk_forward_hitrate(scored: pd.DataFrame) -> dict:
    """Directional hit-rate of sign(score) vs forward copper direction.

    No fitting — the score's weights are fixed, so this is an honest read of how
    often the baseline's sign agreed with the realised 5d/21d move across all
    history. Neutral days (score sign 0 or forward dir 0) are excluded.
    """
    if scored is None or scored.empty or "copper_px" not in scored:
        return {}
    dirs = forward_direction(scored["copper_px"])
    sign = np.sign(scored["score"])
    out = {}
    for h in config.HORIZONS_DAYS:
        d = dirs[f"dir_{h}d"]
        mask = (sign != 0) & (d != 0) & d.notna()
        n = int(mask.sum())
        if n == 0:
            out[f"hit_rate_{h}d"] = None
            out[f"n_{h}d"] = 0
            continue
        hits = int(((sign == d) & mask).sum())
        out[f"hit_rate_{h}d"] = round(hits / n, 4)
        out[f"n_{h}d"] = n
    return out
