"""Central config: paths, score weights/thresholds, source registry, tolerances.

Everything tunable lives here so no module hardcodes a path or a weight. The v1
score weights are FIXED A PRIORI (Plan rule 3) — they are not fitted on the same
history they grade. Changing them is a deliberate model-version bump, not tuning.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Route stdlib ssl (and thus requests) through the OS trust store so pulls work
# behind a TLS-intercepting proxy/AV. No-op if truststore isn't installed.
try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")  # FRED_API_KEY etc; no-op if absent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
MANUAL_DIR = DATA_DIR / "manual"
REGIME_DIR = DATA_DIR / "regime"
JOURNAL_DIR = BASE_DIR / "journal"
RUNS_DIR = JOURNAL_DIR / "runs"
SCORECARD_DIR = JOURNAL_DIR / "scorecards"

REGIME_FILE = REGIME_DIR / "copper_regime.json"
FRESHNESS_FILE = JOURNAL_DIR / "freshness.json"

for _d in (RAW_DIR, CLEAN_DIR, MANUAL_DIR, REGIME_DIR, RUNS_DIR, SCORECARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Build a CA bundle incl. the Windows store so curl_cffi (yfinance) trusts the
# TLS-intercepting root. Must run before yfinance imports curl_cffi.
try:
    from .certs import ensure_ca_bundle
    ensure_ca_bundle(DATA_DIR / "ca_bundle.pem")
except Exception:  # noqa: BLE001
    pass

# ── Secrets ─────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

# ── Source registry ─────────────────────────────────────────────────
# Each source declares its id, whether it needs a key, and the canonical copper
# price column it contributes to the cross-check (None = no comparable copper
# spot/continuous series, e.g. World Bank monthly is a coarse cross-check).
# FRED series ids: PCOPPUSDM copper $/mt, DTWEXBGS broad USD, DFII10 real 10y,
# T10YIE 10y breakeven, VIXCLS VIX.
FRED_SERIES = {
    "copper": "PCOPPUSDM",   # Global price of copper, $/metric ton, monthly
    "usd_broad": "DTWEXBGS",  # Nominal broad USD index, daily
    "real_10y": "DFII10",     # 10y TIPS yield, daily
    "breakeven_10y": "T10YIE",  # 10y breakeven inflation, daily
    "vix": "VIXCLS",          # CBOE VIX, daily
}

# Stooq symbols (free, no key). HG.F continuous copper future, DX.F dollar index.
STOOQ_SYMBOLS = {
    "copper": "HG.F",
    "dollar_index": "DX.F",
    "spy": "SPY.US",
    "eem": "EEM.US",   # EM equity = global growth / copper-demand proxy
    "fxi": "FXI.US",   # China large-cap = #1 copper consumer proxy
}

# yfinance tickers (gray ToS, fallback + 4th cross-check).
YFINANCE_TICKERS = {
    "copper": "HG=F",
    "audusd": "AUDUSD=X",  # commodity-currency / risk proxy
    "tnx": "^TNX",         # 10y note yield x10
}

# Westmetall LME copper table — the ONLY free inventory + cash/3m curve source.
# One page returns a 4-column table: date, cash settlement, 3-month, warehouse
# stock. We derive backwardation = cash - 3m. If the layout changes the source
# degrades to empty rather than crashing ingest.
WESTMETALL_URL = "https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash"
WESTMETALL_DATE_FORMAT = "%d. %B %Y"  # e.g. "19. June 2026"

# World Bank "Pink Sheet" monthly commodity prices (Excel). Coarse cross-check.
# The download URL embeds a doc hash that rotates each monthly release, so we
# discover the live link from the CMO landing page and fall back to this last-
# known URL if discovery fails.
WORLDBANK_CMO_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORLDBANK_PINKSHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
)

# Which sources contribute a copper price series to crosscheck.py.
CROSSCHECK_SOURCES = ("fred", "stooq", "yfinance", "worldbank")

# ── Cross-check tolerance ───────────────────────────────────────────
# Sources quote copper in different units (FRED $/mt, Stooq/yf $/lb). We compare
# normalized % changes, not levels. Flag when a source's recent %-move diverges
# from the cross-source median by more than this (fraction, e.g. 0.05 = 5pp).
CROSSCHECK_PCT_TOLERANCE = 0.05
# A source older than this many days is "stale" (calendar days).
SOURCE_STALE_DAYS = 5
# A source whose latest copper point is older than this is excluded from the
# cross-check (a lagging monthly feed shouldn't fake a divergence). Generous
# enough to keep a normally-current monthly source (e.g. World Bank).
CROSSCHECK_MAX_AGE_DAYS = 70

# ── Copper Score v1 (FIXED weights — Plan §"Copper Score") ──────────
SCORE_WEIGHTS = {
    "trend": 30,        # trend / momentum
    "inventory": 20,    # inventory tightness (LME stocks + backwardation)
    "demand": 20,       # industrial demand proxies
    "usd_rates": 15,    # USD / real-rates macro
    "risk": 10,         # risk appetite
    "news": 5,          # news / disruption score (stub until P6)
}
assert sum(SCORE_WEIGHTS.values()) == 100, "Copper Score weights must sum to 100"

# Each component is scored in [-1, +1]; weighted sum * 100 => [-100, +100].
SCORE_BULLISH_THRESHOLD = 60
SCORE_BEARISH_THRESHOLD = -60

# ── Label / forecast horizons ───────────────────────────────────────
HORIZONS_DAYS = (5, 21)  # forward trading-day direction horizons

# ── Overfit gate (Plan rule 3) — v2 promotes only if BOTH clear ─────
PBO_MAX = 0.5            # Probability of Backtest Overfitting must be < this
# RAISED 0.0 -> 1.645 on 2026-07-30, with the deflated_sharpe unit fix. Once the DSR is
# correct, `ratio > 0` is a MEDIAN test -- "better than the EXPECTED max of N noise
# trials" -- which best-of-8 pure noise clears 44.8% of the time. 1.645 is prob > 0.95.
DEFLATED_SHARPE_MIN = 1.645  # Deflated Sharpe must be > this

MODEL_VERSION_V1 = "v1-score"
MODEL_VERSION_V2 = "v2-rf"

# ── Regime flag-file fail-safe (Plan rule 4) ────────────────────────
# If the primary copper price is older than this many calendar days, the
# published regime is forced to neutral and stale=True. The webhook reader
# applies the same guard against the file's own age.
REGIME_STALE_DAYS = 4
