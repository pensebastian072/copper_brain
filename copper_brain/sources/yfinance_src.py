"""yfinance source — free, gray ToS. Fallback + 4th copper cross-check.

Copper HG=F, AUDUSD (commodity currency), ^TNX (10y x10). Keeps adjusted close.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from .. import config
from .base import Source


class YFinanceSource(Source):
    source_id = "yfinance"
    needs_key = False

    def _fetch_raw(self) -> pd.DataFrame:
        tickers = list(config.YFINANCE_TICKERS.values())
        rev = {v: k for k, v in config.YFINANCE_TICKERS.items()}
        raw = yf.download(
            tickers, period="max", interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "series", "value"])
        # Multi-ticker -> columns MultiIndex (field, ticker); take Close.
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):  # single ticker edge case
            close = close.to_frame(tickers[0])
        long = close.reset_index().melt(
            id_vars=close.index.name or "Date", var_name="ticker", value_name="value"
        )
        long.columns = ["date", "ticker", "value"]
        long["series"] = long["ticker"].map(rev)
        long = long.dropna(subset=["series"])
        return long[["date", "series", "value"]]
