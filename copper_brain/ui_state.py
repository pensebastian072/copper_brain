"""Assemble the full dashboard state from the published artifacts + clean data.

Pure-ish: reads the clean parquet, regime flag-file, freshness manifest, run
records and the v2 verdict, recomputes the feature/score frames, and returns ONE
JSON-able dict the dashboard renders. No Flask here so it stays unit-testable.

Sections: regime, score components, walk-forward hit-rate, v2 gate status, source
health, cross-check, price + inventory history, plain-English findings, and the
v1/v2 projection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .features import build_wide, compute_feature_frame, _pick_copper
from .ingest import load_clean
from .model import load_latest_verdict
from .publish import read_regime
from .score import score_frame, walk_forward_hitrate

MAX_PRICE_POINTS = 500
_COMP_LABELS = {
    "trend": "Trend / momentum", "inventory": "Inventory tightness",
    "demand": "Industrial demand", "usd_rates": "USD / real rates",
    "risk": "Risk appetite", "news": "News / disruption",
}


def _downsample(s: pd.Series, n: int = MAX_PRICE_POINTS) -> pd.Series:
    if len(s) <= n:
        return s
    step = int(np.ceil(len(s) / n))
    return s.iloc[::step]


def _iso_dates(idx) -> list[str]:
    return [d.date().isoformat() for d in pd.to_datetime(idx)]


def _components(feat_row: pd.Series) -> list[dict]:
    out = []
    for comp, label in _COMP_LABELS.items():
        val = float(feat_row.get(f"c_{comp}", 0.0) or 0.0)
        w = config.SCORE_WEIGHTS[comp]
        out.append({"name": comp, "label": label, "value": round(val, 4),
                    "weight": w, "contribution": round(val * w, 2)})
    return out


def _run_history(limit: int = 60) -> list[dict]:
    files = sorted(config.RUNS_DIR.glob("run_*.json"))
    hist = []
    for fp in files[-limit:]:
        try:
            r = json.loads(fp.read_text())
            hist.append({"ran_at": r.get("ran_at"), "regime": r.get("regime"),
                         "score": r.get("score"),
                         "model_version": r.get("model_version")})
        except Exception:  # noqa: BLE001
            continue
    return hist


def _v2_block(verdict: dict | None) -> dict:
    if not verdict:
        return {"status": "not_trained", "promoted": False, "horizons": {}}
    horizons = {}
    for h, g in (verdict.get("horizons") or {}).items():
        dsr = g.get("deflated_sharpe") or {}
        horizons[h] = {
            "passes": g.get("passes"),
            "pbo": g.get("pbo"),
            "dsr_ratio": dsr.get("ratio"),
            "dsr_prob": dsr.get("prob"),
            "profit_factor": g.get("profit_factor"),
            "n_trades": g.get("n_trades"),
            "reasons": g.get("reasons"),
        }
    return {"status": verdict.get("status", "trained"),
            "promoted": bool(verdict.get("promoted")),
            "trained_at": verdict.get("trained_at"),
            "n_rows": verdict.get("n_rows"),
            "p_up": verdict.get("p_up", {}),
            "horizons": horizons}


def _findings(feat_row, wide, copper, regime, crosscheck, hitrate,
              missing) -> list[dict]:
    """Plain-English read of what the data is saying right now."""
    f = []

    def add(tone, text):
        f.append({"tone": tone, "text": text})

    # trend
    ct = float(feat_row.get("c_trend", 0) or 0)
    ret20 = (copper.iloc[-1] / copper.iloc[-21] - 1) * 100 if len(copper) > 21 else None
    if ret20 is not None:
        tone = "bull" if ct > 0.15 else "bear" if ct < -0.15 else "flat"
        add(tone, f"Trend: copper {ret20:+.1f}% over ~1 month; "
                  f"trend score {ct:+.2f}.")

    # inventory / backwardation
    bw_col = "lme_cu_backwardation__westmetall"
    if bw_col in wide and wide[bw_col].notna().any():
        bw = float(wide[bw_col].dropna().iloc[-1])
        if bw > 0:
            add("bull", f"Inventory: LME in backwardation (cash-3m = ${bw:+.0f}/t) "
                        f"-> tight physical market, bullish.")
        else:
            add("bear", f"Inventory: LME in contango (cash-3m = ${bw:+.0f}/t) "
                        f"-> ample supply, bearish.")

    # usd / rates
    cu = float(feat_row.get("c_usd_rates", 0) or 0)
    if abs(cu) > 0.1:
        add("bull" if cu > 0 else "bear",
            f"Macro: USD/real-rates {'supportive' if cu > 0 else 'a headwind'} "
            f"(score {cu:+.2f}).")

    # risk
    vix_col = "vix__fred"
    if vix_col in wide and wide[vix_col].notna().any():
        vix = float(wide[vix_col].dropna().iloc[-1])
        add("bull" if vix < 18 else "bear",
            f"Risk appetite: VIX {vix:.1f} "
            f"({'risk-on' if vix < 18 else 'risk-off'}).")

    # cross-check
    if crosscheck:
        if crosscheck.get("verdict") == "ok":
            add("flat", "Data quality: copper price agrees across sources "
                        f"(month compared {crosscheck.get('compared_month')}).")
        elif crosscheck.get("verdict") == "divergent":
            add("warn", f"Data quality: sources diverge on copper "
                        f"({', '.join(crosscheck.get('diverging', []))}) - "
                        f"treat the regime with caution.")

    # hit-rate honesty
    hr5 = hitrate.get("hit_rate_5d")
    if hr5 is not None:
        add("warn" if hr5 < 0.52 else "flat",
            f"Baseline honesty: v1 5d directional hit-rate {hr5:.1%} "
            f"(n={hitrate.get('n_5d')}) - this is a transparent floor, not an edge.")

    # missing inputs
    real_missing = [m for m in (missing or []) if m != "news"]
    if real_missing:
        add("warn", f"Reduced confidence: missing feeds {', '.join(real_missing)}.")

    if regime.get("stale"):
        add("warn", "Regime is STALE -> forced to neutral (fail-safe).")
    return f


def _projection(regime, v2, hitrate) -> dict:
    score = regime.get("score", 0) or 0
    v1_lean = ("bullish" if score >= config.SCORE_BULLISH_THRESHOLD
               else "bearish" if score <= config.SCORE_BEARISH_THRESHOLD
               else "neutral")
    proj = {
        "horizon_days": list(config.HORIZONS_DAYS),
        "v1": {
            "lean": v1_lean,
            "score": score,
            "confidence": "low",  # v1 baseline ~coin-flip by construction
            "hit_rate_5d": hitrate.get("hit_rate_5d"),
            "hit_rate_21d": hitrate.get("hit_rate_21d"),
            "note": ("Transparent fixed-weight baseline. Directional hit-rate is "
                     "near 50% over full history - treat as context, not a forecast."),
        },
        "v2": None,
    }
    if v2 and v2.get("promoted"):
        proj["v2"] = {"status": "live", "p_up": v2.get("p_up"),
                      "note": "v2 cleared the overfit gate and is publishing p_up."}
    else:
        proj["v2"] = {"status": "shadow", "p_up": None,
                      "note": ("v2 has NOT cleared the overfit gate (PBO<0.5 and "
                               "Deflated Sharpe>0). No statistical edge - kept in "
                               "shadow; v1 publishes.")}
    return proj


def build_state(clean: pd.DataFrame | None = None) -> dict:
    """Return the full dashboard state dict."""
    regime = read_regime()
    freshness = _read_freshness()
    crosscheck = (freshness.get("sources", {}) or {}).get("_crosscheck", {})
    v2_raw = load_latest_verdict()

    if clean is None:
        clean = load_clean()

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "crosscheck": crosscheck,
        "sources": _source_rows(freshness),
        "v2": _v2_block(v2_raw),
        "history": _run_history(),
    }

    if clean is None or clean.empty:
        state.update({"components": [], "hitrate": {}, "price": {}, "inventory": {},
                      "findings": [{"tone": "warn", "text": "No clean data yet - run ingest."}],
                      "projection": _projection(regime, state["v2"], {})})
        return state

    feats, missing = compute_feature_frame(clean)
    scored = score_frame(feats)
    wide = build_wide(clean)
    copper = _pick_copper(wide).dropna()
    hitrate = walk_forward_hitrate(scored)
    feat_row = scored.iloc[-1]

    # price + sma history
    sma50 = copper.rolling(50, min_periods=20).mean()
    cds = _downsample(copper)
    state["price"] = {
        "dates": _iso_dates(cds.index),
        "copper": [round(float(v), 4) for v in cds.values],
        "sma50": [None if pd.isna(v) else round(float(v), 4)
                  for v in sma50.reindex(cds.index).values],
    }

    # inventory (stock + backwardation), recent only
    inv = {}
    for col, key in (("lme_cu_stock__westmetall", "stock"),
                     ("lme_cu_backwardation__westmetall", "backwardation")):
        if col in wide and wide[col].notna().any():
            s = wide[col].dropna()
            inv.setdefault("dates", _iso_dates(s.index))
            inv[key] = [round(float(v), 2) for v in s.values]
    state["inventory"] = inv

    state["components"] = _components(feat_row)
    state["hitrate"] = hitrate
    state["findings"] = _findings(feat_row, wide, copper, regime, crosscheck,
                                  hitrate, missing)
    state["projection"] = _projection(regime, state["v2"], hitrate)
    state["missing_inputs"] = missing
    return state


# ── small readers ───────────────────────────────────────────────────

def _read_freshness() -> dict:
    try:
        return json.loads(config.FRESHNESS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _source_rows(freshness: dict) -> list[dict]:
    rows = []
    for name, r in (freshness.get("sources", {}) or {}).items():
        if name.startswith("_"):
            continue
        rows.append({"name": name, "health": r.get("health"),
                     "data_through": r.get("data_through"),
                     "age_days": r.get("age_days"), "rows": r.get("rows")})
    return rows
