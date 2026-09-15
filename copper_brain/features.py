"""Feature engineering — turn the clean observations table into a daily feature
frame with six Copper-Score component sub-scores, each in [-1, +1].

Design notes
------------
* All sources are merged onto one business-day index. Monthly series (FRED
  copper, World Bank) and business-day series (Westmetall) are forward-filled
  with a cap so a dead feed can't masquerade as fresh forever.
* Primary copper price = yfinance daily ($/lb); falls back to FRED monthly
  ($/mt, ffilled) when yfinance is absent. Units don't matter — every indicator
  is a return / ratio / spread, never a raw level fed across sources.
* Each component is squashed to [-1, +1] with tanh of a return divided by a
  FIXED scale (judgment, set a priori — Plan rule 3, not fitted on outcomes).
* Components degrade gracefully: a sub-score is the mean of whatever inputs are
  available; if none are, it is 0 (neutral) and the input is listed in
  `missing_inputs` so publish can surface reduced confidence.

The frame is computed over full history (vectorised) so the score can be graded
walk-forward against forward returns without look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Fixed normalisation scales (a priori) ───────────────────────────
# A move of this size maps to tanh(1)=~0.76 of the component's range.
SC_COPPER_RET20 = 0.06     # 20d copper return
SC_COPPER_RET60 = 0.12     # 60d copper return
SC_COPPER_SMA = 0.05       # price vs 50d SMA
SC_STOCK_CHG = 0.15        # 30d LME stock change (sign flipped: down = bullish)
SC_BACKWARD = 50.0         # cash-3m spread, $/mt (positive = tight = bullish)
SC_AUD_RET = 0.04          # 20d AUDUSD return (commodity-currency demand proxy)
SC_GROWTH_RET = 0.06       # 20d EM/China equity return (demand proxy)
SC_USD_RET = 0.03          # 20d broad-USD return (up = bearish copper)
SC_REAL10Y_CHG = 0.40      # 20d change in real 10y, pct points (up = bearish)
SC_VIX_CENTER = 18.0       # VIX neutral level
SC_VIX_SCALE = 12.0        # VIX spread from center (high = risk-off = bearish)
SC_SPY_RET = 0.05          # 20d SPY return (up = risk-on = bullish)

FFILL_LIMIT = 40           # max business days to carry a stale value forward

COMPONENTS = ["trend", "inventory", "demand", "usd_rates", "risk", "news"]


def _pick_copper(wide: pd.DataFrame) -> pd.Series:
    """Primary copper price: yfinance daily, else FRED monthly (ffilled)."""
    for col in ("copper__yfinance", "copper__fred", "copper__worldbank"):
        if col in wide and wide[col].notna().any():
            return wide[col].rename("copper_px")
    raise ValueError("no copper price series available in clean data")


def build_wide(clean: pd.DataFrame) -> pd.DataFrame:
    """Pivot long clean table to a business-day wide frame, ffilled with a cap.

    Columns are "<series>__<source>" so same-named series from different sources
    stay distinct (e.g. copper__yfinance vs copper__fred).
    """
    if clean is None or clean.empty:
        return pd.DataFrame()
    df = clean.copy()
    df["col"] = df["series"] + "__" + df["source"]
    wide = df.pivot_table(index="date", columns="col", values="value",
                          aggfunc="last")
    wide = wide.sort_index()
    idx = pd.date_range(wide.index.min(), wide.index.max(), freq="B")
    wide = wide.reindex(wide.index.union(idx)).ffill(limit=FFILL_LIMIT)
    return wide.reindex(idx)


def _tanh(x) -> pd.Series:
    return np.tanh(x)


def _mean_available(parts: list[pd.Series]) -> pd.Series:
    """Row-wise mean ignoring NaN; all-NaN rows -> 0 (neutral)."""
    if not parts:
        return None
    frame = pd.concat(parts, axis=1)
    return frame.mean(axis=1, skipna=True).fillna(0.0)


def compute_feature_frame(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature_frame, missing_inputs).

    feature_frame columns: copper_px, c_trend, c_inventory, c_demand,
    c_usd_rates, c_risk, c_news (each in [-1,1]). missing_inputs lists component
    inputs that were unavailable (reduced-confidence signal for publish).
    """
    wide = build_wide(clean)
    if wide.empty:
        return pd.DataFrame(), ["all"]

    copper = _pick_copper(wide)
    out = pd.DataFrame(index=wide.index)
    out["copper_px"] = copper
    missing: list[str] = []

    def have(col: str) -> bool:
        return col in wide and wide[col].notna().any()

    # ── trend (always available — copper price is mandatory) ─────────
    ret20 = copper.pct_change(20)
    ret60 = copper.pct_change(60)
    sma50 = copper.rolling(50, min_periods=20).mean()
    px_vs_sma = copper / sma50 - 1.0
    out["c_trend"] = _mean_available([
        _tanh(ret20 / SC_COPPER_RET20),
        _tanh(ret60 / SC_COPPER_RET60),
        _tanh(px_vs_sma / SC_COPPER_SMA),
    ])

    # ── inventory (Westmetall: stock down = bullish, backwardation = bullish)
    inv_parts = []
    if have("lme_cu_stock__westmetall"):
        stock_chg = wide["lme_cu_stock__westmetall"].pct_change(30)
        inv_parts.append(-_tanh(stock_chg / SC_STOCK_CHG))
    else:
        missing.append("lme_cu_stock")
    if have("lme_cu_backwardation__westmetall"):
        bw = wide["lme_cu_backwardation__westmetall"]
        inv_parts.append(_tanh(bw / SC_BACKWARD))
    else:
        missing.append("lme_cu_backwardation")
    out["c_inventory"] = _mean_available(inv_parts) if inv_parts else 0.0

    # ── demand (commodity currency + EM/China growth proxies) ────────
    dem_parts = []
    if have("audusd__yfinance"):
        dem_parts.append(_tanh(wide["audusd__yfinance"].pct_change(20) / SC_AUD_RET))
    else:
        missing.append("audusd")
    for proxy in ("fxi__stooq", "eem__stooq"):
        if have(proxy):
            dem_parts.append(_tanh(wide[proxy].pct_change(20) / SC_GROWTH_RET))
        else:
            missing.append(proxy.split("__")[0])
    out["c_demand"] = _mean_available(dem_parts) if dem_parts else 0.0

    # ── usd / rates (strong USD & rising real rates = bearish copper) ─
    ur_parts = []
    if have("usd_broad__fred"):
        ur_parts.append(-_tanh(wide["usd_broad__fred"].pct_change(20) / SC_USD_RET))
    else:
        missing.append("usd_broad")
    if have("real_10y__fred"):
        real_chg = wide["real_10y__fred"].diff(20)  # pct points
        ur_parts.append(-_tanh(real_chg / SC_REAL10Y_CHG))
    else:
        missing.append("real_10y")
    out["c_usd_rates"] = _mean_available(ur_parts) if ur_parts else 0.0

    # ── risk appetite (low/falling VIX bullish; SPY up bullish) ──────
    risk_parts = []
    if have("vix__fred"):
        risk_parts.append(-_tanh((wide["vix__fred"] - SC_VIX_CENTER) / SC_VIX_SCALE))
    else:
        missing.append("vix")
    if have("spy__stooq"):
        risk_parts.append(_tanh(wide["spy__stooq"].pct_change(20) / SC_SPY_RET))
    else:
        missing.append("spy")
    out["c_risk"] = _mean_available(risk_parts) if risk_parts else 0.0

    # ── news / disruption (stub until P6) ────────────────────────────
    out["c_news"] = 0.0
    missing.append("news")

    return out, missing
